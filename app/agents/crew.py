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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM

from app.core import config
from app.core.mock_llm import MockCrewLLM
from app.governance.budget import RequestBudget
from app.governance.least_autonomy import enforce_crew_least_autonomy
from app.schemas import SupportDraft
from app.tools.crew_tools import CheckOrderStatusTool, RagLookupTool

NO_HISTORY = "(none)"


@dataclass
class SupportCrew:
    crew: Crew
    rag_tool: RagLookupTool
    order_tool: CheckOrderStatusTool
    llms: dict[str, BaseLLM]


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


def build_support_crew(
    rag_tool: RagLookupTool | None = None, verbose: bool = False, composer_llm: BaseLLM | None = None,
    budget: RequestBudget | None = None,
) -> SupportCrew:
    if not config.MOCK_LLM:
        raise NotImplementedError("Only MOCK_LLM=true is supported; no real LLM is wired in.")
    rag_tool = rag_tool or RagLookupTool()
    order_tool = CheckOrderStatusTool()
    llms: dict[str, BaseLLM] = {"retrieval": MockCrewLLM(budget), "lookup": MockCrewLLM(budget),  # one per agent
                                "composer": composer_llm or MockCrewLLM(budget)}

    retrieval_agent = Agent(
        role="Retrieval Agent",
        goal="Find the Nykaa policy text that answers the customer's question, using only the rag_lookup tool.",
        backstory="You search the simulated Nykaa policy knowledge base and never answer from memory.",
        tools=[rag_tool], llm=llms["retrieval"], allow_delegation=False, max_iter=3, verbose=verbose,
    )
    lookup_agent = Agent(
        role="Lookup Agent",
        goal="Report the facts of the customer's order using only the check_order_status tool.",
        backstory="You are the only agent allowed to read order records. You never guess order details.",
        tools=[order_tool], llm=llms["lookup"], allow_delegation=False, max_iter=3, verbose=verbose,
    )
    composer_agent = Agent(
        role="Response Composer",
        goal="Combine the policy findings and the order facts into one clear draft reply for the customer.",
        backstory="You write customer replies using only facts given to you by the other agents.",
        tools=[], llm=llms["composer"], allow_delegation=False, max_iter=2, verbose=verbose,
    )

    history = "Conversation so far:\n{history}\n\n"
    retrieval_task = Task(
        description=history + "Customer question: {question}\nSearch the policy knowledge base for this question.",
        expected_output="The rag_lookup JSON result: grounded flag, answer, claims, sources and retrieved context.",
        agent=retrieval_agent,
    )
    lookup_task = Task(
        description=history + "Customer question: {question}\nIf the question or the conversation names an order "
                              "id the customer is asking about, look that order up.",
        expected_output="The check_order_status JSON result, or a note that no order lookup was possible.",
        agent=lookup_agent,
        context=[],
    )
    compose_task = Task(
        description=history + "Customer question: {question}\nCombine the policy findings and the order facts from "
                              "your context into the structured draft. Use only facts from the context.",
        expected_output="A SupportDraft JSON object: policy_claims (text + source_doc_ids), policy_error, order, "
                        "order_error, needs_order_id.",
        agent=composer_agent,
        context=[retrieval_task, lookup_task],
        output_pydantic=SupportDraft,
    )
    enforce_crew_least_autonomy([retrieval_agent, lookup_agent, composer_agent])
    crew = Crew(
        agents=[retrieval_agent, lookup_agent, composer_agent],
        tasks=[retrieval_task, lookup_task, compose_task],
        process=Process.sequential,
        verbose=verbose,
    )
    return SupportCrew(crew, rag_tool, order_tool, llms)


def run_support_crew(
    question: str,
    history: str = NO_HISTORY,
    rag_tool: RagLookupTool | None = None,
    verbose: bool = False,
    composer_llm: BaseLLM | None = None,
    budget: RequestBudget | None = None,
) -> CrewRun:
    support = build_support_crew(rag_tool, verbose, composer_llm, budget)  # fresh crew per run: no state leaks
    out = support.crew.kickoff(inputs={"question": question, "history": history})
    names = ("retrieval", "lookup", "composer")
    calls = {n: getattr(support.llms[n], "calls", []) for n in names}
    return CrewRun(
        question=question,
        raw=out.raw,
        structured=out.pydantic if isinstance(out.pydantic, SupportDraft) else None,
        task_outputs={n: t.raw for n, t in zip(names, out.tasks_output)},
        rag_invocations=list(support.rag_tool.invocations),
        order_invocations=list(support.order_tool.invocations),
        rag_outputs=list(support.rag_tool.outputs),
        order_outputs=list(support.order_tool.outputs),
        llm_calls={n: len(c) for n, c in calls.items()},
        llm_prompts=[str(m.get("content", "")) for c in calls.values() for call in c for m in call["messages"]],
    )
