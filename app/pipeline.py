"""The single request pipeline (Phases 6+): guardrails around the CrewAI crew, wrapped in session memory.

    raw question -> one-line normalisation -> PII masking -> injection check (blocked turns are never stored)
      -> budget pre-flight (oversized requests are refused before any model call and never stored; Phase 8)
      -> RunnableWithMessageHistory(session) -> crew.kickoff(question, history), metered by a RequestBudget
      -> SupportDraft re-validated in code -> groundedness check against the real tool outputs
      -> AutoGen review of the Composer's policy draft + retrieved context (Phase 8): a revision removes claims
      -> final groundedness check of the reviewed draft against the same tool outputs (Phase 8 review fix)
      -> reply rendered from that final checked draft -> SupportResponse

Session history therefore only ever holds masked questions and guardrail-checked replies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from app.agents.crew import NO_HISTORY, CrewRun, run_support_crew
from app.core.cache import GenerationCache
from app.governance.budget import (
    MAX_REQUEST_TOKENS,
    MAX_REQUEST_TOKENS_REAL,
    BudgetExceeded,
    RequestBudget,
    project_request_tokens,
)
from app.guardrails.groundedness import check_draft
from app.guardrails.injection import BLOCKED_REPLY, detect_injection
from app.guardrails.pii import mask_pii
from app.llm.errors import classify
from app.llm.settings import load_llm_settings
from app.memory.session_memory import get_session_history, has_session, with_session_memory
from app.rag.documents import split_sentences
from app.rag.generation import FALLBACK_ANSWER
from app.review.autogen_review import ReviewResult, review_policy_draft
from app.schemas import GuardrailReport, SupportDraft, SupportResponse, render_reply
from app.tools.crew_tools import RagLookupTool

MAX_HISTORY_MESSAGES = 10  # assumption: last 5 exchanges is enough context and bounds prompt size
ERROR_REPLY = "Sorry, I couldn't prepare a reliable answer just now. Please try again or contact support."
BUDGET_REPLY = ("This request is too large to process within the per-request budget. "
                "Please shorten it or split it into smaller questions.")
# v2 chat UI: the stages a request reports while it runs, in order ("order" only when the order tool is called).
STEPS = ("safety", "policies", "order", "composing", "reviewing")


@dataclass
class RequestTrace:
    """v2, opt-in, one per request: live stage events out (on_step) and this request's evidence back (for the chat
    UI's "How this answer was made"). Unlike the last_* attributes it cannot mix up simultaneous requests."""

    on_step: Callable[[str], None] | None = None
    crew_run: CrewRun | None = None
    review: ReviewResult | None = None
    budget: RequestBudget | None = None

    def step(self, name: str) -> None:
        if self.on_step:
            self.on_step(name)


def format_history(messages: list[BaseMessage]) -> str:
    lines = [f"{'customer' if m.type == 'human' else 'assistant'}: {' '.join(str(m.content).split())}"
             for m in messages[-MAX_HISTORY_MESSAGES:]]
    return "\n".join(lines) or NO_HISTORY


# v2: CrewAI raises plain exceptions for these; a fixed label is reported instead of the message (it can quote data)
_CREWAI_FAILURES = {"Task failed guardrail validation": "draft_invalid_after_retry",
                    "Invalid response from LLM call - None or empty": "empty_model_response"}


def _failure_kind(error: BaseException) -> str:
    """The typed provider error kind (v2), a fixed label, or the exception type name; never the message text."""
    provider = classify(error)
    if provider:
        return f"provider_{provider.kind}"
    return next((label for prefix, label in _CREWAI_FAILURES.items() if str(error).startswith(prefix)),
                type(error).__name__)


def _status(draft: SupportDraft, reply: str) -> str:
    if reply == FALLBACK_ANSWER:
        return "fallback"
    if draft.needs_order_id and not draft.policy_claims:
        return "needs_order_id"
    return "answered"


class SupportAssistant:
    """One assistant per process; sessions are kept apart by session_id."""

    def __init__(self, rag_tool_factory=RagLookupTool, composer_llm_factory=None,
                 max_request_tokens: int | None = None, generation_cache: GenerationCache | None = None) -> None:
        # v2: provider settings are validated once, at startup (a bad LLM_* value fails here with a clear error).
        self.llm_settings = load_llm_settings()
        self._rag_tool_factory = rag_tool_factory
        self._composer_llm_factory = composer_llm_factory
        default_cap = MAX_REQUEST_TOKENS_REAL if self.llm_settings.is_real else MAX_REQUEST_TOKENS
        self.max_request_tokens = default_cap if max_request_tokens is None else max_request_tokens
        # Task 16: shared by all sessions; it holds only validated policy text (see app/core/cache.py)
        self.generation_cache = generation_cache if generation_cache is not None else GenerationCache()
        self._chain = with_session_memory(RunnableLambda(self._answer))
        self.last_crew_run = None  # evidence hooks for tests and transcripts
        self.last_review = None
        self.last_budget = None

    def ask(self, session_id: str, question: str, trace: RequestTrace | None = None) -> SupportResponse:
        self.last_review = self.last_budget = None  # Phase 8 evidence hooks describe this request only
        trace = trace or RequestTrace()
        trace.step("safety")
        masked = mask_pii(" ".join(question.split()))  # one line: no forged multi-line prompt structure
        report = GuardrailReport(pii_masked={k: v for k, v in masked.counts.items() if v})
        injection = detect_injection(masked.text)
        if injection.detected:
            report.injection_detected, report.injection_categories = True, list(injection.categories)
            return SupportResponse(session_id=session_id, status="blocked", reply=BLOCKED_REPLY,
                                   masked_question=masked.text, guardrails=report)
        history = format_history(get_session_history(session_id).messages) if has_session(session_id) else ""
        if project_request_tokens(masked.text, history) > self.max_request_tokens:
            # Oversized: refused before any model call, and not stored, so it cannot inflate later turns.
            return SupportResponse(session_id=session_id, status="budget_exceeded", reply=BUDGET_REPLY,
                                   masked_question=masked.text, guardrails=report)
        out = self._chain.invoke({"question": masked.text, "report": report, "trace": trace},
                                 config={"configurable": {"session_id": session_id}})
        return out["response"].model_copy(update={"session_id": session_id})

    def _answer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        question, report, trace = inputs["question"], inputs["report"], inputs["trace"]
        budget = self.last_budget = trace.budget = RequestBudget(self.max_request_tokens)  # fresh per request
        composer_llm = self._composer_llm_factory(budget=budget) if self._composer_llm_factory else None
        rag_tool = self._rag_tool_factory()
        rag_tool.budget = budget  # the grounded-generation call is a model call too
        rag_tool.cache = self.generation_cache  # caches only that generation call; everything else still runs
        try:
            run = run_support_crew(question, format_history(inputs["history"]), rag_tool,
                                   composer_llm=composer_llm, budget=budget, on_step=trace.on_step)
            self.last_crew_run = trace.crew_run = run
            draft = SupportDraft.model_validate(run.structured.model_dump() if run.structured else run.raw)
        except BudgetExceeded:
            return self._over_budget(question, report)
        except (ValidationError, ValueError, TypeError) as e:
            return self._failure(question, report, f"schema: {_failure_kind(e)}")  # same type name unless a v2 label
        except Exception as e:  # the crew itself failed (e.g. CrewAI could not convert the output)
            return self._failure(question, report, f"crew: {_failure_kind(e)}")
        rag_output, order_output = (run.rag_outputs or [None])[-1], (run.order_outputs or [None])[-1]
        checked, grounding = check_draft(draft, rag_output, order_output)
        trace.step("reviewing")
        try:  # spec Task 14: the Composer's draft (before the guardrail) + the original retrieved context
            self.last_review = trace.review = review_policy_draft(draft, rag_output, budget)
        except BudgetExceeded:
            return self._over_budget(question, report)
        except Exception as e:  # v2: a real reviewer can fail (server down, timeout): never ship an unreviewed draft
            return self._failure(question, report, f"review: {_failure_kind(e)}")
        review = self.last_review
        if review and review.verdict.final_answer != review.draft:  # a revision (or an inconsistent approval)
            # The review may only REMOVE claims: keep the checked claims it kept. Its own text is never rendered,
            # so sentences it added (review.discarded_additions) and citations it wrote cannot reach the reply.
            kept = set(split_sentences(review.verdict.final_answer))
            checked = checked.model_copy(update={"policy_claims": [c for c in checked.policy_claims if c.text in kept]})
        # Final validation: whatever leaves the review stage is checked again against the same tool outputs.
        checked, final = check_draft(checked, rag_output, order_output)
        reply = render_reply(checked)
        report = report.model_copy(update={
            **grounding.model_dump(exclude={"pii_masked", "injection_detected", "injection_categories"}),
            "grounded": bool(grounding.grounded and final.grounded),
            "rejected_claims": grounding.rejected_claims + final.rejected_claims,
            "removed_citations": grounding.removed_citations + final.removed_citations,
            "rejected_order": grounding.rejected_order or final.rejected_order,
            "refused": reply == FALLBACK_ANSWER,
        })
        response = SupportResponse(session_id="", status=_status(checked, reply), reply=reply,
                                   masked_question=question, draft=checked, guardrails=report)
        return {"answer": reply, "response": response}

    @staticmethod
    def _over_budget(question: str, report: GuardrailReport) -> dict[str, Any]:
        response = SupportResponse(session_id="", status="budget_exceeded", reply=BUDGET_REPLY,
                                   masked_question=question, guardrails=report)
        return {"answer": BUDGET_REPLY, "response": response}

    @staticmethod
    def _failure(question: str, report: GuardrailReport, reason: str) -> dict[str, Any]:
        report = report.model_copy(update={"schema_valid": False, "rejected_claims": [reason]})
        response = SupportResponse(session_id="", status="error", reply=ERROR_REPLY, masked_question=question,
                                   guardrails=report)
        return {"answer": ERROR_REPLY, "response": response}
