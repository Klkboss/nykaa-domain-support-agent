"""Phase 8 evidence (spec Task 14): the AutoGen review stage approving and revising a draft, through the real
pipeline (SupportAssistant.ask: real crew kickoff, real retrieval, real AutoGen RoundRobinGroupChat).

Inputs were frozen in the Phase 8 checklist (PROJECT_STATUS.md) before any run:
  approve: "How long do I have to return shoes?" with the normal pipeline.
  revise:  the same question; the Composer appends the ungrounded claim below to its otherwise correct draft.

Usage (socket guard, as recorded in the transcript headers):
  .venv\\Scripts\\python.exe -c "import tests.conftest as g, runpy, sys; sys.argv=['x','approve']; runpy.run_path('scripts/run_review_demo.py', run_name='__main__'); print(g.blocked_attempts)"
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.getLogger("crewai.llms.base_llm").setLevel(logging.ERROR)

import app.core.config  # noqa: E402,F401
from app.core.mock_llm import MockCrewLLM  # noqa: E402
from app.pipeline import SupportAssistant  # noqa: E402

QUESTION = "How long do I have to return shoes?"
INJECTED_CLAIM = "Premium members get instant cash refunds at any store."


class AppendsUngroundedClaim(MockCrewLLM):
    """A deliberately misbehaving Composer: its normal draft plus one invented claim (cited KB-01)."""

    def _respond(self, messages, agent_tools):
        draft = json.loads(super()._respond(messages, agent_tools).split("Final Answer:", 1)[1])
        draft["policy_claims"].append({"text": INJECTED_CLAIM, "source_doc_ids": ["KB-01"]})
        return "Thought: drafted\nFinal Answer: " + json.dumps(draft)


def show(mode: str) -> None:
    assistant = SupportAssistant(composer_llm_factory=AppendsUngroundedClaim if mode == "revise" else None)
    response = assistant.ask(f"phase8-{mode}", QUESTION)
    run, review = assistant.last_crew_run, assistant.last_review
    print(f"MODE: {mode}\nquestion: {QUESTION}")
    print(f"crew tool invocations: rag_lookup={run.rag_invocations} check_order_status={run.order_invocations}")
    print(f"crew LLM calls per agent: {run.llm_calls}")
    print(f"\n--- AutoGen input (the Composer's policy draft + the original retrieved context) ---\n"
          f"DRAFT:\n{review.draft}\n\nCONTEXT:\n{review.context}")
    print("\n--- AutoGen RoundRobinGroupChat(max_turns=2) messages ---")
    for m in review.messages:
        content = m.content.model_dump_json() if hasattr(m.content, "model_dump_json") else m.content
        print(f"[{m.source}] {type(m).__name__}: {content}")
    print(f"\n--- Structured verdict ({type(review.verdict).__name__}) ---\n{review.verdict.model_dump_json(indent=2)}")
    print(f"\n--- Pipeline result ---\nstatus: {response.status}\nfinal reply: {response.reply}")
    print(f"groundedness guardrail: rejected_claims={response.guardrails.rejected_claims} "
          f"grounded={response.guardrails.grounded}")
    print(f"editor sentences not in the draft (discarded, never rendered): {review.discarded_additions}")
    print(f"request budget: used {assistant.last_budget.used} of {assistant.last_budget.cap} estimated tokens "
          f"(review reserved {review.reserved_tokens}, its reviewer + editor calls spent {review.spent_tokens}; "
          f"the rest was released)")


if __name__ == "__main__":
    show(sys.argv[1] if len(sys.argv) > 1 else "approve")
