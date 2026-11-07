"""Phase 7 evidence for E01, E05, E08, E12 on the current KB (KB-08 wording v2). Prints before/after retrieval and
answers, the routing table for the frozen escalation regression set, KB-08 wording v1 vs v2, E08 end to end
(retrieval scores, reply, citations, count checks), the E05 category-filter decisions, then runs the versioned
evaluation (scripts/run_evaluation.py, r2).

Usage (socket guard, as recorded in the transcript header):
  .venv\\Scripts\\python.exe -c "import tests.conftest as g, runpy; runpy.run_path('scripts/run_remediation_evidence.py', run_name='__main__'); print(g.blocked_attempts)"
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.getLogger("crewai.llms.base_llm").setLevel(logging.ERROR)

import app.core.config  # noqa: E402,F401
from app.core.config import ROOT  # noqa: E402
from app.judge import fact_present  # noqa: E402
from app.pipeline import SupportAssistant  # noqa: E402
from app.rag.chunking import STRATEGIES  # noqa: E402
import numpy as np  # noqa: E402

from app.rag.documents import load_knowledge_base, split_sentences  # noqa: E402
from app.rag.embeddings import embed  # noqa: E402
from app.rag.evaluation import dedupe  # noqa: E402
from app.rag.generation import (SIMILARITY_THRESHOLDS, MockGroundedGenerator, answer_query,  # noqa: E402
                                contiguous_passages, subject_categories)
from app.rag.query_analysis import analyse_query  # noqa: E402
from app.rag.retrieval import retrieve, search  # noqa: E402
from app.rag.vector_store import get_collection, persistent_client  # noqa: E402

E12 = "My order ORD-1022 was delayed. Will my complaint be escalated to a senior specialist?"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def print_sentence_decisions(col, query: str) -> None:
    """The generator's inputs for one query: each candidate sentence's score, margin and category decision."""
    analysis = analyse_query(query)
    chunks = [c for c in search(col, query, 3) if c.similarity >= SIMILARITY_THRESHOLDS["fixed"]]
    print(f"  retrieved: {[(c.chunk_id, round(c.similarity, 4)) for c in chunks]}; asked categories: "
          f"{list(analysis.categories)}")
    candidates = dedupe([s for p in contiguous_passages(chunks) for s in split_sentences(p)])
    vectors = np.array(embed([analysis.search_text] + candidates))
    scored = sorted(zip(vectors[1:] @ vectors[0], candidates), key=lambda t: -t[0])
    for sim, s in scored:  # margin is shown against the overall best; the generator applies it after the filter
        subject = subject_categories(s)
        kept = not subject or not analysis.categories or bool(subject & set(analysis.categories))
        margin = "in-margin " if sim >= scored[0][0] - MockGroundedGenerator().margin else "OUT-margin"
        print(f"    {sim:.4f} category:{'kept   ' if kept else 'DROPPED'} {margin} subject={sorted(subject) or '-'} "
              f"| {s[:85]}")


def main() -> None:
    client = persistent_client()
    original = {r["id"]: r for r in json.loads((ROOT / "evaluation" / "evaluation_results.json")
                                               .read_text(encoding="utf-8"))["results"]}

    section("E12 root cause and fix: plain dense retrieve() (pre-remediation path) vs shared search()")
    print(f"query: {E12}\nanalyse_query: topic={analyse_query(E12).topic!r}, search_text unchanged="
          f"{analyse_query(E12).search_text == E12}")
    for s in STRATEGIES:
        col = get_collection(client, s)
        print(f"  [{s}] before: {[(c.chunk_id, round(c.similarity, 4)) for c in retrieve(col, E12, 3)]}")
        print(f"  [{s}] after : {[(c.chunk_id, round(c.similarity, 4)) for c in search(col, E12, 3)]}")
        print(f"  [{s}] answer after: {answer_query(E12, col, SIMILARITY_THRESHOLDS[s]).answer}")

    section("Frozen escalation regression set (evaluation/rag_escalation_queries.json)")
    queries = json.loads((ROOT / "evaluation" / "rag_escalation_queries.json").read_text(encoding="utf-8"))["queries"]
    for q in queries:
        route = analyse_query(q["query"]).topic
        tops = {s: search(get_collection(client, s), q["query"], 1)[0].doc_id for s in STRATEGIES}
        print(f"  {q['id']:<3} route={route!s:<18} expected={q['expected_route']!s:<18} "
              f"{'OK ' if route == q['expected_route'] else 'BAD'} top1 fixed={tops['fixed']} sentence={tops['sentence']} "
              f"relevant={q['relevant_doc_ids']}  {q['query']}")

    section("E01 / E05 / E12 replies: original run (b7e2fe9) vs current pipeline (fixed collection)")
    col = get_collection(client, "fixed")
    for qid in ("E01", "E05", "E12"):
        q = original[qid]["question"]
        print(f"  {qid}: {q}\n    original : {original[qid]['reply']}\n    now      : "
              f"{answer_query(q, col, SIMILARITY_THRESHOLDS['fixed']).answer}")

    section("KB-08 clarification (owner-approved): wording v1 (history) vs wording v2 (current)")
    history = json.loads((ROOT / "data" / "knowledge_base" / "history" / "KB-08_wording_v1.json").read_text(encoding="utf-8"))
    kb08 = next(d for d in load_knowledge_base() if d.doc_id == "KB-08")
    for label, text in (("v1", history["document"]["text"]), ("v2", kb08.text)):
        for i, s in enumerate(split_sentences(text), 1):
            print(f"  KB-08 {label} s{i}: {s}")
    print(f"  reason: {history['reason_for_clarification']}")

    section("E08 end to end on KB-08 wording v2 (frozen question, persisted chroma_db re-indexed)")
    e08 = original["E08"]
    print(f"  E08 query: {e08['question']}")
    print(f"  expected facts: {e08['expected']['facts']}")
    for s in STRATEGIES:
        print(f"  [{s}] retrieved: {[(c.chunk_id, round(c.similarity, 4)) for c in search(get_collection(client, s), e08['question'], 3)]}")
    print_sentence_decisions(col, e08["question"])
    response = SupportAssistant().ask("evidence-e08", e08["question"])
    print(f"  original reply (KB-08 v1): {e08['reply']}")
    print(f"  pipeline reply (KB-08 v2): {response.reply}")
    print(f"  citations: {[c.source_doc_ids for c in response.draft.policy_claims]}  grounded={response.guardrails.grounded} "
          f"rejected_claims={response.guardrails.rejected_claims}")
    for probe in (response.reply, e08["reply"], "You can make 3 additional retries.", "You get 3 attempts in total.",
                  "You get four attempts in total.", "You can make 2 additional retries.", "You can make 4 additional retries.",
                  "You can retry the payment up to 3 more times."):
        print(f"  fact '3 times' (counts retries) v1={fact_present('3 times', probe, 'v1')} "
              f"v3={fact_present('3 times', probe, 'v3', 'retry')} | {probe[:110]}")

    section("E05: category-subject filter decisions (trimmer = Electronics)")
    kb05 = next(d for d in load_knowledge_base() if d.doc_id == "KB-05")
    for i, s in enumerate(split_sentences(kb05.text), 1):
        print(f"  KB-05 s{i}: {s}")
    print(f"  expected facts: {original['E05']['expected']}")
    print_sentence_decisions(col, original["E05"]["question"])

    section("Versioned evaluation (scripts/run_evaluation.py): original results are never rewritten")
    import runpy

    runpy.run_path(str(ROOT / "scripts" / "run_evaluation.py"), run_name="__main__")


if __name__ == "__main__":
    main()
