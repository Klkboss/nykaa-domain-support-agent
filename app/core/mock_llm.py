"""Deterministic CrewAI LLM used under MOCK_LLM (no API key, no network).

CrewAI runs tools through its ReAct text loop when supports_function_calling() is False.
After a tool runs, CrewAI appends the model's own turn plus "\\nObservation: <result>" as an
*assistant* message. CrewAI's system prompt also contains the word "Observation:", so the mock
reads loop state from assistant turns only, never from the whole conversation.

Tools are chosen by their declared args_schema fields, never by matching tool names.

An agent without tools (the Response Composer) writes its reply only from JSON tool results found in
the task-context section of its own prompt, so it cannot add facts the other agents did not return.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from pydantic import PrivateAttr

from crewai import BaseLLM
from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import llm_call_context

RECORD_ID_RE = re.compile(r"\bORD-[0-9]{4}\b", re.I)  # the tool itself normalises case
QUESTION_RE = re.compile(r"Customer question:\s*(.+)")
OBSERVATION_MARKER = "\nObservation:"
CONTEXT_MARKER = "This is the context you're working with:"  # CrewAI 1.15.33 header before prior task outputs
ORDER_INTENT_RE = re.compile(r"\b(order|status|track(ing)?|package|parcel|shipment)\b", re.I)
# Stems of policy topics (KB-01..KB-12). Used only to decide whether a question that has an order result
# also asks about policy; it stands in for the relevance judgement a real LLM composer would make.
POLICY_CUE_RE = re.compile(
    r"\b(return|refund|cancel|exchang|swap|size|warrant|guarantee|repair|deliver|dispatch|ship|pick ?up|courier|"
    r"point|loyalty|pay|cod\b|cash on delivery|damag|broke|claim|international|abroad|customs|escalat|complain|"
    r"polic|how long|how many days)\w*",
    re.I,
)
ASK_FOR_ORDER_ID = "Please share your order ID (format ORD-1234) so I can check it."
NO_LOOKUP_NEEDED = "No order lookup needed: the question does not refer to a specific order."


def json_objects(text: str) -> list[dict[str, Any]]:
    """Every JSON object embedded in `text`, in order."""
    decoder, found, i = json.JSONDecoder(), [], 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict):
            found.append(obj)
        i = end
    return found


ORDER_FACT_FIELDS = ("record_id", "status", "order_value_inr", "escalation_score", "escalation_threshold",
                     "escalation_recommended")


def compose_draft(prompt: str) -> dict[str, Any]:
    """Response Composer: a SupportDraft-shaped dict built only from tool results in the context section.

    The LAST context header is used: CrewAI appends prior task outputs at the end, so header text a customer
    typed earlier in the prompt cannot smuggle in fake facts.
    """
    context = prompt.rsplit(CONTEXT_MARKER, 1)[1] if CONTEXT_MARKER in prompt else ""
    facts = json_objects(context)
    rag = next((f for f in facts if f.get("tool") == "rag_lookup"), None)
    order = next((f for f in facts if f.get("tool") == "check_order_status"), None)
    question = (QUESTION_RE.findall(prompt) or [""])[-1]
    order_answered = order is not None or ASK_FOR_ORDER_ID in context
    if order_answered and not POLICY_CUE_RE.search(question):
        rag = None  # a pure order question: retrieved policy text would be off-topic padding
    draft: dict[str, Any] = {"policy_claims": [], "policy_error": None, "order": None, "order_error": None,
                             "needs_order_id": False}
    if rag and "error" in rag:
        draft["policy_error"] = rag["error"]
    elif rag and rag.get("grounded"):
        draft["policy_claims"] = [c for c in rag.get("claims", []) if c.get("source_doc_ids")]
    if order and "error" in order:
        draft["order_error"] = order["error"]
    elif order:
        draft["order"] = {k: order[k] for k in ORDER_FACT_FIELDS}
    elif ASK_FOR_ORDER_ID in context:
        draft["needs_order_id"] = True
    return draft


def extract_slot_values(prompt: str) -> dict[str, str]:
    """Values the mock can fill into tool arguments, in dispatch priority order."""
    values: dict[str, str] = {}
    if ids := RECORD_ID_RE.findall(prompt):
        values["record_id"] = ids[-1]  # most recent mention wins (question comes after history)
    if questions := QUESTION_RE.findall(prompt):
        values["query"] = questions[-1].strip()
    return values


def pick_tool(tools: list[Any], values: dict[str, str]) -> tuple[Any, dict[str, str]] | None:
    """Return the first tool whose required argument set is exactly one fillable slot."""
    for field, value in values.items():
        for tool in tools:
            schema = getattr(tool, "args_schema", None)
            if schema is None:
                continue
            required = {name for name, f in schema.model_fields.items() if f.is_required()}
            if required == {field}:
                return tool, {field: value}
    return None


class MockCrewLLM(BaseLLM):
    """CrewAI BaseLLM that answers from the prompt with fixed rules."""

    _calls: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _budget: Any = PrivateAttr(default=None)  # app.governance.budget.RequestBudget, or None (unmetered)

    def __init__(self, budget: Any = None, **data: Any) -> None:
        data.setdefault("model", "mock/deterministic")
        data.setdefault("provider", "mock")
        super().__init__(**data)
        self._budget = budget

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self._calls

    def call(
        self,
        messages: Any,
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
        **kwargs: Any,
    ) -> str:
        prompt_tokens = len(str(messages)) // 4
        if self._budget is not None:
            self._budget.charge(prompt_tokens)  # before the prompt is "sent": an over-budget call never happens
        with llm_call_context():
            self._emit_call_started_event(
                messages=messages, tools=tools, callbacks=callbacks,
                available_functions=available_functions, from_task=from_task, from_agent=from_agent,
            )
            text = self._respond(messages, list(getattr(from_agent, "tools", None) or []))
            if self._budget is not None:
                self._budget.charge(len(text) // 4)  # before the completion is returned
            self._calls.append({"messages": copy.deepcopy(messages), "response": text,
                                "response_model": response_model})
            usage = {"prompt_tokens": prompt_tokens, "completion_tokens": len(text) // 4}
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            self._track_token_usage_internal(usage)
            self._emit_call_completed_event(
                response=text, call_type=LLMCallType.LLM_CALL, from_task=from_task,
                from_agent=from_agent, messages=messages, usage=usage,
            )
            return text

    async def acall(self, *args: Any, **kwargs: Any) -> str:
        return self.call(*args, **kwargs)

    def _respond(self, messages: Any, agent_tools: list[Any]) -> str:
        msgs = [{"role": "user", "content": messages}] if isinstance(messages, str) else messages
        prompt = "\n".join(str(m.get("content", "")) for m in msgs if m.get("role") == "user")
        generated = [str(m.get("content", "")) for m in msgs if m.get("role") == "assistant"]
        observations = [g.split(OBSERVATION_MARKER, 1)[1].strip() for g in generated if OBSERVATION_MARKER in g]

        if observations:
            return f"Thought: I now know the final answer\nFinal Answer: {observations[-1]}"
        if not agent_tools:
            return f"Thought: I can write the reply from my context.\nFinal Answer: {json.dumps(compose_draft(prompt))}"
        if picked := pick_tool(agent_tools, extract_slot_values(prompt)):
            tool, args = picked
            return f"Thought: I should use {tool.name}.\nAction: {tool.name}\nAction Input: {json.dumps(args)}"
        # Judge order intent from the customer's question only: task instructions mention "order" too.
        question = (QUESTION_RE.findall(prompt) or [""])[-1]
        reply = ASK_FOR_ORDER_ID if ORDER_INTENT_RE.search(question) else NO_LOOKUP_NEEDED
        return f"Thought: None of my tools can be used with this request.\nFinal Answer: {reply}"

    def supports_function_calling(self) -> bool:
        return False  # keep CrewAI on the ReAct text loop

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 128_000
