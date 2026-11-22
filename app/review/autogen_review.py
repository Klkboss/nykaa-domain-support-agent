"""AutoGen review stage (spec Task 14): Policy-Compliance-Reviewer -> Final-Editor (structured Verdict).

The pipeline runs it after the crew: the Composer's policy draft plus the original retrieved context go in, and
the Final-Editor's `Verdict` either approves the draft unchanged or revises it.

Verdict semantics (spec Task 14: "either approves it unchanged or revises it"):
  approved=True   the Composer's draft is approved unchanged; final_answer is that draft.
  approved=False  the draft was not approved as written; final_answer is the editor's revision, reason says why.
`approved` judges the Composer's draft, not the final answer. The pipeline never renders final_answer: it only
uses it to decide which of the Composer's guardrail-checked claims survive, then re-checks the result.

v2, real provider: both agents use the metered OpenAI-compatible AutoGen client, with fuller instructions (the mock
keeps its own prompts). There is no up-front review reservation: each call reserves its own prompt estimate +
max_tokens against the request budget and settles on the reported usage. A provider failure is re-raised as its
typed LLMError (AutoGen itself only re-raises a RuntimeError string), so the pipeline can answer politely.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pydantic import BaseModel

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat

from app.core.mock_autogen import MockChatCompletionClient
from app.governance.budget import BudgetExceeded, RequestBudget, estimate_tokens
from app.governance.least_autonomy import enforce_reviewers_have_no_tools
from app.llm.autogen_client import MeteredChatCompletionClient, build_review_client
from app.llm.settings import LLMSettings, load_llm_settings
from app.rag.documents import split_sentences
from app.schemas import SupportDraft

# AutoGen agent names must be Python identifiers, so hyphens become underscores.
REVIEWER_NAME = "Policy_Compliance_Reviewer"
EDITOR_NAME = "Final_Editor"
# Upper bound of one review's token use: two prompts carry the task, the editor's also the reviewer's reply,
# and both completions are at most the draft. Tests check it against the clients' measured usage.
REVIEW_OVERHEAD_TOKENS = 200
# v2, real models only: the mock's terse system messages stay as they are (its prompts are budget-calibrated).
REAL_REVIEWER_SYSTEM = (
    "You check a customer-support DRAFT against the policy CONTEXT. For every draft sentence that the CONTEXT does "
    "not state, write one line 'UNSUPPORTED: <the sentence copied exactly>'. If every sentence is supported, write "
    "exactly 'ALL CLAIMS SUPPORTED'. Do not rewrite or add anything."
)
REAL_EDITOR_SYSTEM = (
    "You produce the final verdict on the DRAFT. If the reviewer flagged no sentence: approved=true and "
    "final_answer=the draft unchanged. Otherwise: approved=false, final_answer=the draft with the flagged sentences "
    "removed (keep every other sentence exactly as written; never add or reword text), and reason=what was removed."
)
RealClients = tuple[MeteredChatCompletionClient, MeteredChatCompletionClient]


class Verdict(BaseModel):
    approved: bool
    final_answer: str
    reason: str


@dataclass
class ReviewResult:
    draft: str
    context: str
    verdict: Verdict
    messages: list
    reserved_tokens: int
    spent_tokens: int  # metered reviewer + editor calls (<= reserved_tokens)
    discarded_additions: list[str]  # sentences the editor wrote that were not in the draft: never rendered


def build_review_team(budget: RequestBudget | None = None,
                      real_clients: RealClients | None = None) -> RoundRobinGroupChat:
    reviewer = AssistantAgent(
        REVIEWER_NAME,
        model_client=real_clients[0] if real_clients else MockChatCompletionClient("reviewer", REVIEWER_NAME, budget),
        system_message=REAL_REVIEWER_SYSTEM if real_clients else
        "List every draft sentence that the CONTEXT does not support.",
    )
    editor = AssistantAgent(
        EDITOR_NAME,
        model_client=real_clients[1] if real_clients else MockChatCompletionClient("editor", REVIEWER_NAME, budget),
        system_message=REAL_EDITOR_SYSTEM if real_clients else
        "Produce the final verdict. Remove unsupported claims flagged by the reviewer.",
        output_content_type=Verdict,
    )
    enforce_reviewers_have_no_tools([reviewer, editor])
    return RoundRobinGroupChat(
        [reviewer, editor], max_turns=2, custom_message_types=[StructuredMessage[Verdict]]
    )


def review_task(draft: str, context: str) -> str:
    return f"DRAFT:\n{draft}\n\nCONTEXT:\n{context}"


def review_token_reservation(task: str) -> int:
    return 4 * estimate_tokens(task) + REVIEW_OVERHEAD_TOKENS


async def review(draft: str, context: str, budget: RequestBudget | None = None,
                 real_clients: RealClients | None = None) -> tuple[Verdict, list]:
    """Run the team; return the validated verdict and the full message list."""
    result = await build_review_team(budget, real_clients).run(task=review_task(draft, context))
    last = result.messages[-1]
    if not (isinstance(last, StructuredMessage) and isinstance(last.content, Verdict)):
        raise TypeError(f"Final-Editor did not return a Verdict: {type(last).__name__}")
    return last.content, list(result.messages)


def review_policy_draft(draft: SupportDraft, rag_output: dict | None,
                        budget: RequestBudget | None = None) -> ReviewResult | None:
    """Review the Composer's policy claims against the retrieved context. None when the draft has no policy text
    (order-only or no-answer turns): order facts are checked against the real tool output by the guardrail."""
    text = " ".join(c.text for c in draft.policy_claims)
    if not text:
        return None
    context = "\n".join(p for e in (rag_output or {}).get("evidence", []) for p in e["passages"])
    settings = load_llm_settings()
    if settings.is_real:
        return _review_with_real_model(text, context, budget, settings)
    reserved = review_token_reservation(review_task(text, context))
    if budget is not None:
        budget.charge(reserved)  # reserved before the team starts: AutoGen would wrap an in-run failure
    meter = RequestBudget(reserved)  # every reviewer and editor call is charged against the reservation
    try:
        verdict, messages = asyncio.run(review(text, context, meter))
    except RuntimeError:
        if meter.exhausted:  # AutoGen re-raised our BudgetExceeded as a RuntimeError string
            raise BudgetExceeded(0, meter.used, meter.cap) from None
        raise
    finally:
        if budget is not None:
            budget.release(reserved - meter.used)
    draft_sentences = set(split_sentences(text))
    added = [s for s in split_sentences(verdict.final_answer) if s not in draft_sentences]
    return ReviewResult(text, context, verdict, messages, reserved, meter.used, added)


def _review_with_real_model(text: str, context: str, budget: RequestBudget | None,
                            settings: LLMSettings) -> ReviewResult:
    clients = (build_review_client(settings, budget), build_review_client(settings, budget))
    before = budget.used if budget is not None else 0

    async def review_then_close() -> tuple[Verdict, list]:
        try:
            return await review(text, context, budget, clients)
        finally:  # inside the loop: closing after asyncio.run() hits a closed loop and leaks the connections
            for client in clients:
                await client.close()

    try:
        verdict, messages = asyncio.run(review_then_close())
    except Exception:
        if budget is not None and budget.exhausted:  # AutoGen re-raised our BudgetExceeded as a RuntimeError
            raise BudgetExceeded(0, budget.used, budget.cap) from None
        error = clients[0].error or clients[1].error
        if error is not None:
            raise error from None
        raise
    spent = (budget.used - before) if budget is not None else 0
    draft_sentences = set(split_sentences(text))
    added = [s for s in split_sentences(verdict.final_answer) if s not in draft_sentences]
    # reserved_tokens = spent: with a real model each call reserves for itself; nothing is held for the whole review
    return ReviewResult(text, context, verdict, messages, spent, spent, added)
