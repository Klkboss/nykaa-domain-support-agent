"""Phase 5 evidence run (spec Task 7): real crew.kickoff() demos under MOCK_LLM with CrewAI's verbose log.

Usage: .venv\\Scripts\\python.exe scripts\\run_crew_demo.py
Uses the deployed collection (chroma_db/, policies_fixed_overlap); run scripts\\run_rag_evaluation.py first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.core.config  # noqa: E402,F401  (offline env first)
from app.agents.crew import run_support_crew  # noqa: E402

DEMOS = [
    ("Demo A: policy retrieval", "How long do I have to return shoes?"),
    ("Demo B: order lookup", "What is the status of my order ORD-1016?"),
    ("Demo C: combined policy + order", "My order ORD-1029 arrived late. Can I still return the shoes, and is my case being escalated?"),
    ("Edge: unknown order", "Where is my order ORD-9999?"),
    ("Edge: out-of-scope", "What is the best way to train for a marathon?"),
    ("Edge: order question without an id", "Can you check my order status?"),
]


def main() -> None:
    summary = []
    for title, question in DEMOS:
        print(f"\n{'#' * 78}\n# {title}\n# Customer question: {question}\n{'#' * 78}")
        r = run_support_crew(question, verbose=True)  # CrewAI prints its own agent/tool execution panels
        print(f"\n--- {title}: execution summary ---")
        print(f"rag_lookup invocations:         {r.rag_invocations}")
        print(f"check_order_status invocations: {r.order_invocations}")
        print(f"LLM calls per agent:            {r.llm_calls}")
        for name, raw in r.task_outputs.items():
            print(f"[{name} task output] {raw}")
        draft = r.structured.model_dump() if r.structured else None  # SupportDraft since Phase 6 (was a string)
        print(f"FINAL DRAFT (SupportDraft):\n{json.dumps(draft, indent=2, ensure_ascii=False)}")
        summary.append({"demo": title, "question": question, "rag_invocations": r.rag_invocations,
                        "order_invocations": r.order_invocations, "llm_calls": r.llm_calls, "draft": draft})
    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
