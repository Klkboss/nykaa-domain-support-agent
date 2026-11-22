"""The support crew (spec Task 7): Retrieval Agent -> Lookup Agent -> Response Composer, run via kickoff().

Flow (sequential process):
  1. retrieval_task (Retrieval Agent, tool rag_lookup): grounded policy answer + retrieved context, as JSON.
  2. lookup_task (Lookup Agent, tool check_order_status): order facts as JSON. `context=[]`, so it sees only
     the customer question and never the policy output.
  3. compose_task (Response Composer, no tools): `context=[retrieval_task, lookup_task]`; returns a
     `SupportDraft` (`output_pydantic`, the crew-level `response_format` in crewai 1.15.33).
Every task description carries the session history (Phase 6). Only the Lookup Agent holds check_order_status
(least autonomy, spec Task 15): `enforce_crew_least_autonomy` refuses any other wiring when the crew is built.
A per-request `RequestBudget` (spec Task 15, Runtime layer) meters every mock LLM call when one is given.

v2, real provider (LLM_PROVIDER=ollama|openai_compatible with MOCK_LLM=false); the mock path is unchanged:
  - every agent gets a metered real LLM with native tool calling;
  - both tools use `result_as_answer`, so the next agent receives the tool's exact JSON, not a paraphrase;
  - the Composer's LLM is schema-constrained to SupportDraft, and a task guardrail validates the draft and lets
    CrewAI retry the Composer ONCE with the validation errors; a second failure raises (the pipeline then gives the
    polite error reply). The groundedness guardrail still checks every claim against the real tool outputs.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from crewai import Agent, Crew, Process, Task, TaskOutput
from crewai.llms.base_llm import BaseLLM
from pydantic import ValidationError

from app.core import config
from app.core.mock_llm import ORDER_FACT_FIELDS, ORDER_INTENT_RE, POLICY_CUE_RE, RECORD_ID_RE, MockCrewLLM
from app.governance.budget import RequestBudget
from app.governance.least_autonomy import enforce_crew_least_autonomy
from app.llm.crew_llm import build_crew_llm
from app.llm.settings import load_llm_settings
from app.rag.documents import split_sentences
from app.schemas import PolicyClaim, SupportDraft
from app.tools.crew_tools import CheckOrderStatusTool, RagLookupTool

NO_HISTORY = "(none)"
COMPOSER_RETRIES = 1  # v2: one retry with the validation errors, then the request fails safely
# v2, real models only (the mock reads the JSON context directly, and its prompts must stay unchanged).
REAL_COMPOSER_RULES = (
    "\nFill the draft only from the JSON tool results in your context:\n"
    "- policy_claims: copy each item of the rag_lookup result's \"claims\" list exactly (text and source_doc_ids). "
    "Leave it empty if rag_lookup returned grounded=false or an error.\n"
    "- policy_error: the rag_lookup \"error\" text if it has one, otherwise null.\n"
    "- order: copy record_id, status, order_value_inr, escalation_score, escalation_threshold and "
    "escalation_recommended exactly from the check_order_status result; null if there is none.\n"
    "- order_error: the check_order_status \"error\" text if it has one, otherwise null.\n"
    "- needs_order_id: true only if the customer asks about their own order and no order id was given.\n"
    "Never invent facts, ids or numbers."
)


def draft_guardrail(context_tasks: list[Task]):  # type: ignore[no-untyped-def]
    """Composer guardrail (real models): the raw answer must be a valid SupportDraft. CrewAI's retry REPLACES the
    task context with this error text, so the error re-attaches the context tasks' exact tool results; it names
    fields and rules, never the rejected values. (No return annotation: CrewAI 1.15.33 rejects a postponed one.)"""
    def valid_support_draft(output: TaskOutput):  # type: ignore[no-untyped-def]  # -> tuple[bool, Any]
        raw = output.raw
        try:
            data = json.loads(raw)
            # The one deterministic repair (counted in .repairs): once an order WAS looked up, "please share your
            # order id" is moot. Clearing the flag adds no content; the order facts are still verified downstream.
            if isinstance(data, dict) and data.get("needs_order_id") and (data.get("order") or data.get("order_error")):
                data["needs_order_id"] = False
                valid_support_draft.repairs.append("needs_order_id")  # type: ignore[attr-defined]
                raw = json.dumps(data)
        except ValueError:
            pass  # not JSON: the validation below reports it
        try:
            draft = SupportDraft.model_validate_json(raw)
        except ValidationError as e:
            problems = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'draft'}: {err['msg']}" for err in e.errors())
            facts = "\n\n".join(t.output.raw for t in context_tasks if t.output is not None)
            return False, (f"Return only a JSON object that is a valid SupportDraft. Problems: {problems}\n\n"
                           f"Use only these tool results:\n{facts}")
        return True, draft.model_dump_json()
    valid_support_draft.repairs = []  # type: ignore[attr-defined]
    return valid_support_draft


def apply_intent_rules(draft: SupportDraft, question: str, ids_given: set[str], notes: list[str],
                       order_output: dict[str, Any] | None = None) -> SupportDraft:
    """Real models: the capstone mock Composer's deterministic rules, applied to the model's validated draft. They
    never invent anything: order fields are copied from the order tool's own output, and otherwise content is only
    removed or the fixed "please share your order ID" line is set. The groundedness guardrail still checks the
    result. Each applied rule is noted (counted in reports).
      1. needs_order_id is decided in code: an order question (ORDER_INTENT_RE) with no order id given.
      2. Order facts / the order error are COPIED FROM THE ORDER TOOL'S OUTPUT, never transcribed by the model
         (measured: qwen3:4b left `order` empty although the tool returned the facts). No lookup -> no order fields.
      3. A pure order question (order part answered, no policy cue in the question) gets no policy text.
      4. Each policy claim is split into sentences, so two sentences a model merged are verified separately.
    """
    update: dict[str, Any] = {}
    needs_id = bool(ORDER_INTENT_RE.search(question)) and not ids_given
    if needs_id != draft.needs_order_id:
        notes.append("ask_for_order_id" if needs_id else "no_order_question")
    update["needs_order_id"] = needs_id
    order, order_error = None, None
    if order_output and not needs_id:
        if "error" in order_output:
            order_error = order_output["error"]
        else:
            order = {k: order_output[k] for k in ORDER_FACT_FIELDS}
    model_order = draft.order.model_dump() if draft.order else None
    if (model_order, draft.order_error) != (order, order_error):
        notes.append("order_from_tool")
    update.update(order=order, order_error=order_error)
    order_answered = needs_id or order is not None or order_error is not None
    if order_answered and not POLICY_CUE_RE.search(question) and (draft.policy_claims or draft.policy_error):
        notes.append("pure_order_question")
        update.update(policy_claims=[], policy_error=None)
    else:
        split = [PolicyClaim(text=s, source_doc_ids=c.source_doc_ids)
                 for c in draft.policy_claims for s in split_sentences(c.text)]
        if len(split) != len(draft.policy_claims):
            notes.append("split_claims")
        update["policy_claims"] = split[:5]  # SupportDraft allows at most 5 claims
    return SupportDraft.model_validate(draft.model_dump() | update)


@dataclass
class SupportCrew:
    crew: Crew
    rag_tool: RagLookupTool
    order_tool: CheckOrderStatusTool
    llms: dict[str, BaseLLM]
    real: bool = False  # v2: built for a real provider
    draft_repairs: list[str] = field(default_factory=list)  # v2: the Composer guardrail's counted repairs


@dataclass
class CrewRun:
    question: str
    raw: str
    structured: SupportDraft | None  # CrewOutput.pydantic, None if CrewAI could not convert the raw output
    task_outputs: dict[str, str]
    rag_invocations: list[dict[str, Any]]
    order_invocations: list[dict[str, Any]]
    rag_outputs: list[dict[str, Any]]
    order_outputs: list[dict[str, Any]]
    llm_calls: dict[str, int]
    llm_prompts: list[str]  # every message every agent's LLM received (PII tests inspect this)
    draft_repairs: list[str] = field(default_factory=list)  # v2, real models: counted Composer draft repairs


def build_support_crew(
    rag_tool: RagLookupTool | None = None, verbose: bool = False, composer_llm: BaseLLM | None = None,
    budget: RequestBudget | None = None, on_step: Callable[[str], None] | None = None,
) -> SupportCrew:
    settings = load_llm_settings()
    if not config.MOCK_LLM and not settings.is_real:
        raise NotImplementedError("MOCK_LLM=false needs a real provider: set LLM_PROVIDER=ollama or openai_compatible.")
    rag_tool = rag_tool or RagLookupTool()
    order_tool = CheckOrderStatusTool()
    if on_step:  # v2 chat UI: real stage events (a tool really starts), never a timer
        rag_tool.on_start, order_tool.on_start = (lambda: on_step("policies")), (lambda: on_step("order"))
    llms: dict[str, BaseLLM]
    if settings.is_real:
        llms = {"retrieval": build_crew_llm(settings, budget), "lookup": build_crew_llm(settings, budget),
                "composer": composer_llm or build_crew_llm(settings, budget, response_format=SupportDraft)}
        rag_tool.result_as_answer = order_tool.result_as_answer = True  # exact tool JSON is the task output
    else:
        llms = {"retrieval": MockCrewLLM(budget), "lookup": MockCrewLLM(budget),  # one per agent
                "composer": composer_llm or MockCrewLLM(budget)}
    # Real models: no agent-level exception retry (CrewAI's default 2 would triple every timeout and outage); the
    # Composer's single validation retry is its task guardrail. The mock agents keep CrewAI's defaults.
    agent_extra: dict[str, Any] = {"max_retry_limit": 0} if settings.is_real else {}

    retrieval_agent = Agent(
        role="Retrieval Agent",
        goal="Find the Nykaa policy text that answers the customer's question, using only the rag_lookup tool.",
        backstory="You search the simulated Nykaa policy knowledge base and never answer from memory.",
        tools=[rag_tool], llm=llms["retrieval"], allow_delegation=False, max_iter=3, verbose=verbose, **agent_extra,
    )
    lookup_agent = Agent(
        role="Lookup Agent",
        goal="Report the facts of the customer's order using only the check_order_status tool.",
        backstory="You are the only agent allowed to read order records. You never guess order details.",
        tools=[order_tool], llm=llms["lookup"], allow_delegation=False, max_iter=3, verbose=verbose, **agent_extra,
    )
    composer_agent = Agent(
        role="Response Composer",
        goal="Combine the policy findings and the order facts into one clear draft reply for the customer.",
        backstory="You write customer replies using only facts given to you by the other agents.",
        tools=[], llm=llms["composer"], allow_delegation=False, max_iter=2, verbose=verbose, **agent_extra,
    )

    history = "Conversation so far:\n{history}\n\n"
    retrieval_task = Task(
        description=history + "Customer question: {question}\nSearch the policy knowledge base for this question.",
        expected_output="The rag_lookup JSON result: grounded flag, answer, claims, sources and retrieved context.",
        agent=retrieval_agent,
    )
    with warnings.catch_warnings():
        # The per-request progress callback is a closure, which CrewAI warns it cannot checkpoint. This crew is built
        # fresh per request and never checkpointed, so that one warning is expected here.
        warnings.filterwarnings("ignore", message="A function callback cannot be saved in a checkpoint")
        lookup_task = Task(
            description=history + "Customer question: {question}\nIf the question or the conversation names an "
                                  "order id the customer is asking about, look that order up.",
            expected_output="The check_order_status JSON result, or a note that no order lookup was possible.",
            agent=lookup_agent,
            context=[],
            callback=(lambda _output: on_step("composing")) if on_step else None,  # the Composer task starts next
        )
    # Real models: the provider enforces the SupportDraft JSON schema (response_format on the Composer's LLM) and the
    # guardrail validates it, with one retry. output_pydantic is mock-only: with a tool-calling LLM CrewAI would
    # parse inside the agent's own call, where a validation error bypasses the guardrail's retry.
    guardrail = draft_guardrail([retrieval_task, lookup_task])
    compose_extra: dict[str, Any] = (
        {"guardrail": guardrail, "guardrail_max_retries": COMPOSER_RETRIES}
        if settings.is_real else {"output_pydantic": SupportDraft})
    compose_task = Task(
        description=history + "Customer question: {question}\nCombine the policy findings and the order facts from "
                              "your context into the structured draft. Use only facts from the context."
                    + (REAL_COMPOSER_RULES if settings.is_real else ""),
        expected_output="A SupportDraft JSON object: policy_claims (text + source_doc_ids), policy_error, order, "
                        "order_error, needs_order_id.",
        agent=composer_agent,
        context=[retrieval_task, lookup_task],
        **compose_extra,
    )
    enforce_crew_least_autonomy([retrieval_agent, lookup_agent, composer_agent])
    crew = Crew(
        agents=[retrieval_agent, lookup_agent, composer_agent],
        tasks=[retrieval_task, lookup_task, compose_task],
        process=Process.sequential,
        verbose=verbose,
    )
    return SupportCrew(crew, rag_tool, order_tool, llms, real=settings.is_real, draft_repairs=guardrail.repairs)


def run_support_crew(
    question: str,
    history: str = NO_HISTORY,
    rag_tool: RagLookupTool | None = None,
    verbose: bool = False,
    composer_llm: BaseLLM | None = None,
    budget: RequestBudget | None = None,
    on_step: Callable[[str], None] | None = None,
) -> CrewRun:
    support = build_support_crew(rag_tool, verbose, composer_llm, budget, on_step)  # fresh crew per run: no state leaks
    if support.real:  # v2: a real model may only look up an order id the customer actually wrote
        support.order_tool.allowed_ids = {m.upper() for m in RECORD_ID_RE.findall(f"{question}\n{history}")}
    out = support.crew.kickoff(inputs={"question": question, "history": history})
    names = ("retrieval", "lookup", "composer")
    calls = {n: getattr(support.llms[n], "calls", []) for n in names}
    structured = out.pydantic if isinstance(out.pydantic, SupportDraft) else None
    if support.real:  # the guardrail already validated it; a failure here means CrewAI changed, and fails safe
        structured = apply_intent_rules(SupportDraft.model_validate_json(out.raw), question,
                                        support.order_tool.allowed_ids or set(), support.draft_repairs,
                                        (support.order_tool.outputs or [None])[-1])
    return CrewRun(
        question=question,
        raw=out.raw,
        structured=structured,
        task_outputs={n: t.raw for n, t in zip(names, out.tasks_output)},
        rag_invocations=list(support.rag_tool.invocations),
        order_invocations=list(support.order_tool.invocations),
        rag_outputs=list(support.rag_tool.outputs),
        order_outputs=list(support.order_tool.outputs),
        llm_calls={n: len(c) for n, c in calls.items()},
        llm_prompts=[str(m.get("content", "")) for c in calls.values() for call in c for m in call["messages"]],
        draft_repairs=list(support.draft_repairs),
    )
