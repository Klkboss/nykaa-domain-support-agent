"""The documented Phase 5 crew demo (scripts/run_crew_demo.py) must keep running against the current CrewRun.

Regression: Phase 6 renamed CrewRun.draft to CrewRun.structured and the script crashed with AttributeError
(found by the final submission audit). The real script runs its six real kickoffs here, on the test index.
"""

from __future__ import annotations

import json

import scripts.run_crew_demo as demo
from app.agents.crew import run_support_crew
from app.rag.generation import RECOMMENDED_STRATEGY
from app.tools.crew_tools import RagLookupTool


def test_crew_demo_runs_end_to_end_and_reports_structured_drafts(collections, monkeypatch, capsys):
    monkeypatch.setattr(demo, "run_support_crew", lambda q, verbose: run_support_crew(
        q, rag_tool=RagLookupTool(collection=collections[RECOMMENDED_STRATEGY]), verbose=verbose))
    demo.main()
    summary = json.loads(capsys.readouterr().out.split("SUMMARY\n" + "=" * 78 + "\n", 1)[1])
    by = {d["demo"]: d for d in summary}
    assert [d["demo"] for d in summary] == [title for title, _ in demo.DEMOS]
    assert by["Demo A: policy retrieval"]["draft"]["policy_claims"][0]["source_doc_ids"] == ["KB-01"]
    assert by["Demo B: order lookup"]["draft"]["order"]["record_id"] == "ORD-1016"
    combined = by["Demo C: combined policy + order"]
    assert combined["rag_invocations"] and combined["order_invocations"] == [{"record_id": "ORD-1029"}]
    assert by["Edge: unknown order"]["draft"]["order"] is None and by["Edge: unknown order"]["draft"]["order_error"]
    assert by["Edge: order question without an id"]["draft"]["needs_order_id"] is True
