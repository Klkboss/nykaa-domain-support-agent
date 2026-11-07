"""AutoGen review stage (spec Task 14): Policy-Compliance-Reviewer -> Final-Editor (structured Verdict).

The pipeline runs it after the crew: the Composer's policy draft plus the original retrieved context go in, and
the Final-Editor's `Verdict` either approves the draft unchanged or revises it.

Verdict semantics (spec Task 14: "either approves it unchanged or revises it"):
  approved=True   the Composer's draft is approved unchanged; final_answer is that draft.
  approved=False  the draft was not approved as written; final_answer is the editor's revision, reason says why.
`approved` judges the Composer's draft, not the final answer. The pipeline never renders final_answer: it only
uses it to decide which of the Composer's guardrail-checked claims survive, then re-checks the result.
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
from app.rag.documents import split_sentences
from app.schemas import SupportDraft

# AutoGen agent names must be Python identifiers, so hyphens become underscores.
REVIEWER_NAME = "Policy_Compliance_Reviewer"
EDITOR_NAME = "Final_Editor"
# Upper bound of one review's token use: two prompts carry the task, the editor's also the reviewer's reply,
# and both completions are at most the draft. Tests check it against the clients' measured usage.
REVIEW_OVERHEAD_TOKENS = 200


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


def build_review_team(budget: RequestBudget | None = None) -> RoundRobinGroupChat:
    reviewer = AssistantAgent(
        REVIEWER_NAME,
        model_client=MockChatCompletionClient("reviewer", REVIEWER_NAME, budget),
        system_message="List every draft sentence that the CONTEXT does not support.",
    )
    editor = AssistantAgent(
        EDITOR_NAME,
        model_client=MockChatCompletionClient("editor", REVIEWER_NAME, budget),
        system_message="Produce the final verdict. Remove unsupported claims flagged by the reviewer.",
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


async def review(draft: str, context: str, budget: RequestBudget | None = None) -> tuple[Verdict, list]:
    """Run the team; return the validated verdict and the full message list."""
    result = await build_review_team(budget).run(task=review_task(draft, context))
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
