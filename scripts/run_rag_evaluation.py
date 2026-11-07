"""Phase 3 evidence run (spec Tasks 3-5): index both collections, calibrate the fallback threshold,
demonstrate grounded generation and compare chunking strategies by document-level precision/recall.

Usage: .venv\\Scripts\\python.exe scripts\\run_rag_evaluation.py
Writes evaluation/rag_threshold_calibration.json and evaluation/chunking_comparison.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.core.config  # noqa: E402,F401  (offline env first)
from app.core.config import ROOT  # noqa: E402
from app.rag.chunking import FIXED_CHUNK_WORDS, FIXED_OVERLAP_WORDS, SENTENCES_PER_CHUNK, STRATEGIES  # noqa: E402
from app.rag.documents import load_knowledge_base  # noqa: E402
from app.rag.embeddings import EMBEDDING_DIM, MODEL_DIR  # noqa: E402
from app.rag.evaluation import choose_threshold, precision_recall  # noqa: E402
from app.rag.generation import (  # noqa: E402
    DEFAULT_TOP_K, SIMILARITY_THRESHOLDS, MockGroundedGenerator, answer_query,
)
from app.rag.query_analysis import analyse_query  # noqa: E402
from app.rag.retrieval import retrieve, search  # noqa: E402
from app.rag.vector_store import COLLECTION_NAMES, get_collection, index_all, persistent_client  # noqa: E402

EVAL_DIR = ROOT / "evaluation"
SAMPLE_QUERY = "How long do I have to return shoes?"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    docs = load_knowledge_base()
    queries = json.loads((EVAL_DIR / "rag_queries.json").read_text(encoding="utf-8"))
    client = persistent_client()
    revision = (MODEL_DIR / "REVISION.txt").read_text(encoding="utf-8").strip()

    section("1. Indexing (collection.upsert, run twice to show re-ingestion creates no duplicates)")
    print(f"Embedding model: {revision} | dimension {EMBEDDING_DIM} | device cpu | normalised")
    print(f"Fixed: {FIXED_CHUNK_WORDS} words, overlap {FIXED_OVERLAP_WORDS} | Sentence: {SENTENCES_PER_CHUNK} sentence/chunk")
    first, second = index_all(client, docs), index_all(client, docs)
    collections = {s: get_collection(client, s) for s in STRATEGIES}
    for s in STRATEGIES:
        c = collections[s]
        print(f"  {c.name:<26} space={c.configuration['hnsw']['space']}  chunks: run1={first[s]} "
              f"run2={second[s]}  collection.count()={c.count()}")

    section(f"2. Sample retrieval on both collections (k={DEFAULT_TOP_K}): {SAMPLE_QUERY!r}")
    for s in STRATEGIES:
        print(f"  [{COLLECTION_NAMES[s]}]")
        for r in search(collections[s], SAMPLE_QUERY, DEFAULT_TOP_K):
            print(f"    {r.similarity:.4f}  {r.chunk_id:<20} {r.text[:80]}...")

    section("3. Fallback threshold calibration: top-1 cosine similarity per query (search() pipeline)")
    cal = queries["calibration"]
    calibration: dict = {"model": revision, "retrieval": "search(): metadata-aware query analysis + dense top-k",
                         "method": "threshold = (min in-scope top-1 + max out-of-scope top-1) / 2 on UNROUNDED "
                                   "similarities, then rounded to 3 decimals", "strategies": {}}
    for s in STRATEGIES:
        rows = []
        for group, items in (("in_scope", cal["in_scope"]), ("out_of_scope", cal["out_of_scope"])):
            for item in items:
                top = search(collections[s], item["query"], 1)[0]
                rows.append({"scope": group, "query": item["query"], "expected_doc_id": item.get("expected_doc_id"),
                             "routed_topic": analyse_query(item["query"]).topic,
                             "top1_doc_id": top.doc_id, "top1_similarity": top.similarity})
        decision = choose_threshold([r["top1_similarity"] for r in rows if r["scope"] == "in_scope"],
                                    [r["top1_similarity"] for r in rows if r["scope"] == "out_of_scope"])
        unrounded_midpoint = (decision["min_in_scope"] + decision["max_out_of_scope"]) / 2
        calibration["strategies"][s] = {"collection": COLLECTION_NAMES[s], "measurements": rows,
                                        "unrounded_midpoint": unrounded_midpoint, **decision}
        print(f"  [{COLLECTION_NAMES[s]}]")
        for r in rows:
            mark = "" if r["expected_doc_id"] in (None, r["top1_doc_id"]) else f"  (expected {r['expected_doc_id']})"
            print(f"    {r['scope']:<12} {r['top1_similarity']:.4f}  top1={r['top1_doc_id']}  routed={r['routed_topic']}"
                  f"  {r['query']}{mark}")
        print(f"    -> ({decision['min_in_scope']:.8f} + {decision['max_out_of_scope']:.8f}) / 2 = "
              f"{unrounded_midpoint:.8f} -> threshold {decision['threshold']} | gap {decision['gap']:.8f}")
        if SIMILARITY_THRESHOLDS[s] != decision["threshold"]:
            print(f"    NOTE: app/rag/generation.py SIMILARITY_THRESHOLDS[{s!r}] = {SIMILARITY_THRESHOLDS[s]} "
                  f"differs from this calibration; update it.")
    (EVAL_DIR / "rag_threshold_calibration.json").write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")

    section("4. Grounded generation (answers use ONLY retrieved chunks above the threshold)")
    demo = queries["evaluation"] + [{"id": "OOS", **queries["out_of_scope_demo"], "relevant_doc_ids": []}]
    for s in STRATEGIES:
        threshold = calibration["strategies"][s]["threshold"]
        generator = MockGroundedGenerator()
        print(f"  [{COLLECTION_NAMES[s]}] threshold={threshold}")
        for q in demo:
            a = answer_query(q["query"], collections[s], threshold, DEFAULT_TOP_K, generator)
            print(f"    {q['id']}: {q['query']}\n        top1={a.top_similarity:.4f} grounded={a.grounded} "
                  f"routed={a.routed_topic} sources={a.sources}\n        answer: {a.answer}")
        print(f"    generator calls: {generator.calls} (fallbacks never call the generator)")

    section(f"5. Document-level precision / recall (top-{DEFAULT_TOP_K} chunks -> parent docs, deduped)")
    comparison: dict = {"k": DEFAULT_TOP_K, "queries": queries["evaluation"], "strategies": {}}
    for s in STRATEGIES:
        per_query = []
        print(f"  [{COLLECTION_NAMES[s]}]")
        for q in queries["evaluation"]:
            chunks = search(collections[s], q["query"], DEFAULT_TOP_K)
            pr = precision_recall([c.doc_id for c in chunks], q["relevant_doc_ids"])
            per_query.append({"id": q["id"], "retrieved_chunk_ids": [c.chunk_id for c in chunks], **pr})
            print(f"    {q['id']}: chunks {[c.chunk_id for c in chunks]}\n"
                  f"        docs {pr['retrieved_docs']} relevant {pr['relevant_docs']} "
                  f"-> P = {pr['precision_arithmetic']}  R = {pr['recall_arithmetic']}")
        n = len(per_query)
        avg_p = sum(r["precision"] for r in per_query) / n
        avg_r = sum(r["recall"] for r in per_query) / n
        f1 = 2 * avg_p * avg_r / (avg_p + avg_r)
        comparison["strategies"][s] = {"collection": COLLECTION_NAMES[s], "per_query": per_query,
                                       "avg_precision": round(avg_p, 4), "avg_recall": round(avg_r, 4),
                                       "f1_of_averages": round(f1, 4)}
        print(f"    AVERAGE: precision = ({' + '.join(f'{r['precision']:.3f}' for r in per_query)}) / {n} = {avg_p:.4f}"
              f" | recall = ({' + '.join(f'{r['recall']:.3f}' for r in per_query)}) / {n} = {avg_r:.4f}"
              f" | F1 = 2*{avg_p:.4f}*{avg_r:.4f}/({avg_p:.4f}+{avg_r:.4f}) = {f1:.4f}")
    comparison["retrieval"] = "search(): metadata-aware query analysis + dense top-k"
    (EVAL_DIR / "chunking_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")

    section("6. Return-vs-warranty regression: plain dense retrieve() BEFORE vs search() AFTER")
    intent_set = json.loads((EVAL_DIR / "rag_intent_queries.json").read_text(encoding="utf-8"))["queries"]
    regression: dict = {"k": DEFAULT_TOP_K, "strategies": {}}
    for s in STRATEGIES:
        rows = []
        print(f"  [{COLLECTION_NAMES[s]}]")
        for q in [{"id": "Q3", **queries["evaluation"][2], "expected_intent": "warranty"}] + intent_set:
            before = retrieve(collections[s], q["query"], DEFAULT_TOP_K)
            after = search(collections[s], q["query"], DEFAULT_TOP_K)
            analysis = analyse_query(q["query"])
            row = {"id": q["id"], "query": q["query"], "relevant_doc_ids": q["relevant_doc_ids"],
                   "expected_intent": q["expected_intent"], "routed_topic": analysis.topic,
                   "categories": list(analysis.categories),
                   "before": [[c.chunk_id, round(c.similarity, 4)] for c in before],
                   "after": [[c.chunk_id, round(c.similarity, 4)] for c in after],
                   "before_top1_relevant": before[0].doc_id in q["relevant_doc_ids"],
                   "after_top1_relevant": after[0].doc_id in q["relevant_doc_ids"]}
            rows.append(row)
            print(f"    {q['id']} routed={analysis.topic} categories={list(analysis.categories)}: {q['query']}\n"
                  f"        before {row['before']}\n        after  {row['after']}")
        regression["strategies"][s] = {
            "rows": rows,
            "top1_relevant_before": sum(r["before_top1_relevant"] for r in rows),
            "top1_relevant_after": sum(r["after_top1_relevant"] for r in rows),
            "routing_matches_expected": sum(r["routed_topic"] == r["expected_intent"] for r in rows),
        }
        print(f"    top-1 relevant: before {regression['strategies'][s]['top1_relevant_before']}/{len(rows)}, "
              f"after {regression['strategies'][s]['top1_relevant_after']}/{len(rows)}; routing as expected "
              f"{regression['strategies'][s]['routing_matches_expected']}/{len(rows)}")
    (EVAL_DIR / "rag_intent_regression.json").write_text(json.dumps(regression, indent=2) + "\n", encoding="utf-8")
    print("\nWrote evaluation/rag_threshold_calibration.json, evaluation/chunking_comparison.json, "
          "evaluation/rag_intent_regression.json")


if __name__ == "__main__":
    main()
