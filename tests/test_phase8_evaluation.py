"""Phase 8: the 15-query evaluation through the FULL pipeline (CrewAI + AutoGen + final groundedness check)."""

from __future__ import annotations

import json

from app.core.config import ROOT
from app.pipeline import SupportAssistant
from app.rag.generation import RECOMMENDED_STRATEGY
from app.tools.crew_tools import RagLookupTool

SAVED = json.loads((ROOT / "evaluation" / "phase8" / "full_pipeline_evaluation.json").read_text(encoding="utf-8"))
FROZEN = [q["id"] for q in json.loads((ROOT / "evaluation" / "test_queries.json").read_text(encoding="utf-8"))["queries"]]
KEYS = ("id", "status", "reply", "autogen", "final_check", "scores_rubric_v1", "scores_rubric_v3", "same_as_r2")


def test_full_pipeline_evaluation_is_reproducible_live(collections):
    import scripts.run_phase8_evaluation as phase8

    live = phase8.evaluate(SupportAssistant(
        rag_tool_factory=lambda: RagLookupTool(collection=collections[RECOMMENDED_STRATEGY])))
    assert [{k: r[k] for k in KEYS} for r in live["results"]] == [{k: r[k] for k in KEYS} for r in SAVED["results"]]
    assert live["averages"] == SAVED["averages"]


def test_autogen_ran_and_the_final_reply_passed_the_final_check():
    rows = {r["id"]: r for r in SAVED["results"]}
    assert list(rows) == FROZEN  # the same 15 frozen queries, in order
    ran = {i for i, r in rows.items() if r["autogen"]["ran"]}
    assert ran == {f"E{n:02d}" for n in range(1, 13)}  # every turn with policy text was reviewed
    for i, r in rows.items():
        if r["autogen"]["ran"]:
            assert r["autogen"]["message_sources"] == ["user", "Policy_Compliance_Reviewer", "Final_Editor"]
            assert r["autogen"]["message_types"][-1] == "StructuredMessage[Verdict]"
            assert r["autogen"]["spent_tokens"] <= r["autogen"]["reserved_tokens"]
        else:
            assert i in {"E13", "E14", "E15"} and r["autogen"]["why"]
        if r["status"] != "blocked":  # the reply is the rendered final draft, and it re-checks clean
            fc = r["final_check"]
            assert fc["unchanged_by_recheck"] and fc["reply_is_rendered_final_draft"]
            assert not fc["rejected_claims"] and not fc["removed_citations"] and not fc["rejected_order"]


def test_full_pipeline_scores_equal_r2_and_nothing_was_altered():
    assert all(r["same_as_r2"] for r in SAVED["results"])
    assert SAVED["averages"] == {"v1": {"accuracy": 4.7333, "grounding": 5, "completeness": 5, "safety": 5},
                                 "v3": {"accuracy": 5, "grounding": 5, "completeness": 5, "safety": 5}}


def test_history_left_in_an_eval_session_does_not_leak_into_a_new_run(collections):
    """Regression (found in Phase 8): run_case reuses fixed eval-<id> session ids, so a second evaluation in one
    process inherited the first run's history. E07's reply asks for an order id "(format ORD-1234)", and with
    that reply in the history the Lookup Agent looked up ORD-1234."""
    import scripts.run_phase8_evaluation as phase8
    from app.memory.session_memory import get_session_history

    saved = next(r for r in SAVED["results"] if r["id"] == "E07")
    history = get_session_history("eval-E07")
    history.add_user_message(saved["question"])  # exactly what an earlier run in this process leaves behind
    history.add_ai_message(saved["reply"])
    assert "ORD-1234" in saved["reply"]
    live = phase8.evaluate(SupportAssistant(
        rag_tool_factory=lambda: RagLookupTool(collection=collections[RECOMMENDED_STRATEGY])))
    e07 = next(r for r in live["results"] if r["id"] == "E07")
    assert e07["reply"] == saved["reply"] and "No order found" not in e07["reply"]
