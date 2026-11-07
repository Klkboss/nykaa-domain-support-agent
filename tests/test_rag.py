"""Spec Tasks 3-5 with the REAL all-MiniLM-L6-v2 model (no dummy embeddings), network blocked."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from app.core.config import ROOT
from app.rag import embeddings
from app.rag.chunking import (
    FIXED_CHUNK_WORDS, FIXED_OVERLAP_WORDS, STRATEGIES, chunk_documents, fixed_size_chunks, sentence_chunks,
)
from app.rag.documents import KBDocument, load_knowledge_base, split_sentences
from app.rag.embeddings import EMBEDDING_DIM, MODEL_DIR, ModelNotDownloadedError, embed
from app.rag.evaluation import choose_threshold, dedupe, precision_recall
from app.rag.generation import (
    DEFAULT_TOP_K, FALLBACK_ANSWER, RECOMMENDED_STRATEGY, SIMILARITY_THRESHOLDS, MockGroundedGenerator,
    answer_query, contiguous_passages,
)
from app.rag.query_analysis import analyse_query
from app.rag.retrieval import RetrievedChunk, distance_to_similarity, retrieve, search
from app.rag.vector_store import COLLECTION_NAMES, get_collection, index_all, index_documents

DOCS = load_knowledge_base()
QUERIES = json.loads((ROOT / "evaluation" / "rag_queries.json").read_text(encoding="utf-8"))
INTENT_QUERIES = json.loads((ROOT / "evaluation" / "rag_intent_queries.json").read_text(encoding="utf-8"))["queries"]
Q3 = QUERIES["evaluation"][2]["query"]
SHORT_DOC = KBDocument(doc_id="KB-98", title="Short", topic="short_doc", categories=["All"],
                       text="Gift cards never expire. They work on every order.")

# `client` / `collections` fixtures (real MiniLM, both strategies indexed) live in tests/conftest.py.

# --- Embeddings: real model, local, offline -------------------------------------------------

def test_model_is_downloaded_locally_and_hub_is_offline():
    # Fails (never skips) if scripts/download_model.py was not run.
    assert (MODEL_DIR / "modules.json").exists() and (MODEL_DIR / "model.safetensors").exists()
    assert (MODEL_DIR / "REVISION.txt").read_text(encoding="utf-8").startswith("sentence-transformers/all-MiniLM-L6-v2@")
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_real_embeddings_have_expected_shape_norm_and_are_finite():
    vectors = np.array(embed(["COD refund timeline", "Return window for shoes"]))
    assert vectors.shape == (2, EMBEDDING_DIM) == (2, 384)
    assert np.isfinite(vectors).all() and np.abs(vectors).sum() > 0
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_embeddings_are_deterministic_and_semantic():
    a, b = embed(["When will my refund arrive?"]), embed(["When will my refund arrive?"])
    np.testing.assert_allclose(a, b, atol=1e-6)
    q, related, unrelated = np.array(embed(["When will I get my money back?", "refund timeline", "bicycle tyre repair"]))
    assert q @ related > q @ unrelated + 0.2


def test_embed_rejects_blank_input():
    with pytest.raises(ValueError):
        embed([])
    with pytest.raises(ValueError):
        embed(["ok", "  "])


def test_missing_model_raises_instead_of_skipping(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "MODEL_DIR", tmp_path / "absent")
    embeddings.get_model.cache_clear()
    try:
        with pytest.raises(ModelNotDownloadedError, match="download_model.py"):
            embeddings.get_model()
    finally:
        monkeypatch.undo()
        embeddings.get_model.cache_clear()


# --- Chunking --------------------------------------------------------------------------------

@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.doc_id)
def test_fixed_chunks_overlap_and_reconstruct_document(doc):
    chunks = fixed_size_chunks(doc)
    words = doc.text.split()
    assert all(0 < len(c.text.split()) <= FIXED_CHUNK_WORDS for c in chunks)
    for prev, cur in zip(chunks, chunks[1:]):
        assert prev.text.split()[-FIXED_OVERLAP_WORDS:] == cur.text.split()[:FIXED_OVERLAP_WORDS]
        assert cur.metadata["word_start"] == prev.metadata["word_end"] - FIXED_OVERLAP_WORDS
    rebuilt = chunks[0].text.split() + [w for c in chunks[1:] for w in c.text.split()[FIXED_OVERLAP_WORDS:]]
    assert rebuilt == words and chunks[-1].metadata["word_end"] == len(words)


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.doc_id)
def test_sentence_chunks_are_exactly_the_sentences(doc):
    chunks = sentence_chunks(doc)
    assert [c.text for c in chunks] == split_sentences(doc.text)
    assert [c.metadata["sentence_start"] for c in chunks] == list(range(len(chunks)))


def test_sentence_chunks_can_group_sentences():
    chunks = sentence_chunks(DOCS[0], per_chunk=2)  # 5 sentences -> 2 + 2 + 1
    assert [len(split_sentences(c.text)) for c in chunks] == [2, 2, 1]


def test_short_document_handling():
    assert len(fixed_size_chunks(SHORT_DOC)) == 1  # 9 words < 40
    assert fixed_size_chunks(SHORT_DOC)[0].text == SHORT_DOC.text
    assert len(sentence_chunks(SHORT_DOC)) == 2


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_chunks_preserve_parent_and_metadata_and_are_unique(strategy):
    chunks = chunk_documents(DOCS, strategy)
    by_id = {d.doc_id: d for d in DOCS}
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert {c.doc_id for c in chunks} == set(by_id)  # no document lost
    for c in chunks:
        assert c.text.strip() and c.chunk_id.startswith(f"{c.doc_id}::{strategy}::")
        assert by_id[c.doc_id].chroma_metadata().items() <= c.metadata.items()
        assert c.metadata["strategy"] == strategy and c.text in by_id[c.doc_id].text


def test_chunk_counts_and_determinism():
    assert len(chunk_documents(DOCS, "fixed")) == 42
    assert len(chunk_documents(DOCS, "sentence")) == 55
    assert chunk_documents(DOCS, "fixed") == chunk_documents(DOCS, "fixed")


@pytest.mark.parametrize("kwargs", [{"size": 0}, {"size": 10, "overlap": 10}, {"overlap": -1}])
def test_fixed_chunk_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        fixed_size_chunks(DOCS[0], **kwargs)


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        chunk_documents(DOCS, "paragraph")  # type: ignore[arg-type]


# --- ChromaDB collections --------------------------------------------------------------------

def test_two_separate_cosine_collections_with_real_embeddings(collections):
    assert COLLECTION_NAMES == {"fixed": "policies_fixed_overlap", "sentence": "policies_sentence_based"}
    assert collections["fixed"].count() == 42 and collections["sentence"].count() == 55
    for s, col in collections.items():
        assert col.name == COLLECTION_NAMES[s] and col.configuration["hnsw"]["space"] == "cosine"
        stored = col.get(limit=1, include=["embeddings", "metadatas"])
        assert len(stored["embeddings"][0]) == EMBEDDING_DIM and np.isfinite(stored["embeddings"][0]).all()
        assert stored["metadatas"][0]["strategy"] == s
    fixed_ids = set(collections["fixed"].get(include=[])["ids"])
    assert fixed_ids.isdisjoint(collections["sentence"].get(include=[])["ids"])


def test_stored_chunk_keeps_ids_and_metadata(collections):
    got = collections["sentence"].get(ids=["KB-09::sentence::04"], include=["metadatas", "documents"])
    meta = got["metadatas"][0]
    assert meta["doc_id"] == "KB-09" and meta["topic"] == "size_exchange" and meta["chunk_index"] == 4
    assert got["documents"][0] == "Beauty, Home and Electronics products are not eligible for size exchange."


def test_reingestion_does_not_duplicate_and_removes_stale_chunks(client, collections):
    index_all(client, DOCS)
    assert collections["fixed"].count() == 42 and collections["sentence"].count() == 55
    shorter = DOCS[1].model_copy(update={"text": "COD refunds go to the wallet. Bank refunds take longer."})
    try:
        index_documents(client, [shorter], "sentence")
        assert collections["sentence"].count() == 55 - 4 + 2
        assert collections["sentence"].get(ids=["KB-02::sentence::03"])["ids"] == []
    finally:
        index_documents(client, [DOCS[1]], "sentence")
    assert collections["sentence"].count() == 55


# --- Retrieval and similarity ----------------------------------------------------------------

@pytest.mark.parametrize("strategy", STRATEGIES)
def test_retrieval_from_both_collections(collections, strategy):
    results = retrieve(collections[strategy], "When will my cash on delivery refund arrive?", k=3)
    assert len(results) == 3 and results[0].doc_id == "KB-02"
    assert [r.similarity for r in results] == sorted((r.similarity for r in results), reverse=True)
    assert all(isinstance(r, RetrievedChunk) and r.metadata["strategy"] == strategy for r in results)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_similarity_equals_independent_cosine(collections, strategy):
    query = "Can I exchange my shoes for a bigger size?"
    q = np.array(embed([query])[0])
    for r in retrieve(collections[strategy], query, k=3):
        assert r.similarity == pytest.approx(float(q @ np.array(embed([r.text])[0])), abs=1e-4)


def test_distance_to_similarity():
    assert distance_to_similarity(0.0) == 1.0
    assert distance_to_similarity(0.25) == 0.75
    assert distance_to_similarity(1.0) == 0.0


def test_retrieve_rejects_bad_input(collections):
    with pytest.raises(ValueError):
        retrieve(collections["fixed"], "refund", k=0)
    with pytest.raises(ValueError):
        retrieve(collections["fixed"], "   ")


def test_retrieval_is_deterministic(collections):
    query = "How do I raise a damaged item claim?"
    assert retrieve(collections["fixed"], query) == retrieve(collections["fixed"], query)


# --- Threshold calibration -------------------------------------------------------------------

@pytest.mark.parametrize("strategy", STRATEGIES)
def test_threshold_reproduces_from_live_measurements(collections, strategy):
    # Same pipeline that answers queries (search), on UNROUNDED similarities.
    top1 = {group: [search(collections[strategy], q["query"], 1)[0].similarity for q in items]
            for group, items in QUERIES["calibration"].items()}
    decision = choose_threshold(top1["in_scope"], top1["out_of_scope"])
    assert decision["threshold"] == SIMILARITY_THRESHOLDS[strategy]
    assert max(top1["out_of_scope"]) < SIMILARITY_THRESHOLDS[strategy] < min(top1["in_scope"])
    saved = json.loads((ROOT / "evaluation" / "rag_threshold_calibration.json").read_text(encoding="utf-8"))
    assert saved["strategies"][strategy]["threshold"] == SIMILARITY_THRESHOLDS[strategy]


def test_calibration_inputs_meet_spec_minimums():
    assert len(QUERIES["calibration"]["in_scope"]) >= 3 and len(QUERIES["calibration"]["out_of_scope"]) >= 2
    assert SIMILARITY_THRESHOLDS["fixed"] not in (0.5, 0.6, 0.7)  # not an untested tutorial preset


def test_choose_threshold_rejects_overlap_and_too_few_points():
    with pytest.raises(ValueError, match="overlap"):
        choose_threshold([0.4, 0.5, 0.6], [0.45, 0.1])
    with pytest.raises(ValueError):
        choose_threshold([0.6, 0.7], [0.1, 0.2])
    assert choose_threshold([0.5, 0.6, 0.7], [0.1, 0.3])["threshold"] == 0.4


# --- Grounded generation and fallback ------------------------------------------------------------

@pytest.mark.parametrize("strategy", STRATEGIES)
def test_grounded_answer_uses_only_retrieved_text(collections, strategy):
    gen = MockGroundedGenerator()
    a = answer_query("When do my loyalty points expire?", collections[strategy], SIMILARITY_THRESHOLDS[strategy],
                     generator=gen)
    assert a.grounded and gen.calls == 1 and "KB-07" in a.sources
    supporting = " ".join(contiguous_passages([c for c in a.chunks if c.similarity >= a.threshold]))
    for sentence in split_sentences(a.answer):
        assert sentence in supporting  # extractive: nothing outside the retrieved context
    assert "12 months" in a.answer


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_out_of_scope_triggers_fallback_without_generation(collections, strategy):
    gen = MockGroundedGenerator()
    a = answer_query(QUERIES["out_of_scope_demo"]["query"], collections[strategy],
                     SIMILARITY_THRESHOLDS[strategy], generator=gen)
    assert a.answer == FALLBACK_ANSWER and not a.grounded and a.sources == []
    assert a.top_similarity < a.threshold and gen.calls == 0


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_all_demo_queries_in_scope_are_answered(collections, strategy):
    for q in QUERIES["evaluation"]:
        a = answer_query(q["query"], collections[strategy], SIMILARITY_THRESHOLDS[strategy])
        assert a.grounded and a.answer != FALLBACK_ANSWER, q["id"]


def test_e08_retrieves_and_answers_with_the_kb08_retry_rule(collections):
    q = "My UPI payment failed twice. How many times can I retry?"
    chunks = search(collections["fixed"], q, DEFAULT_TOP_K)
    assert [c.doc_id for c in chunks] == ["KB-08"] * 3 and chunks[-1].similarity >= SIMILARITY_THRESHOLDS["fixed"]
    a = answer_query(q, collections["fixed"], SIMILARITY_THRESHOLDS["fixed"])
    assert "retry the payment up to 3 more times within 30 minutes" in a.answer and "all 3 retries" in a.answer
    assert a.sources == ["KB-08"] and "failed attempts" not in a.answer


def test_generation_is_deterministic(collections):
    col, t = collections[RECOMMENDED_STRATEGY], SIMILARITY_THRESHOLDS[RECOMMENDED_STRATEGY]
    q = QUERIES["evaluation"][3]["query"]
    assert answer_query(q, col, t).answer == answer_query(q, col, t).answer


def test_contiguous_passages_merges_overlapping_windows():
    doc_chunks = fixed_size_chunks(DOCS[7])  # KB-08: 123 words -> windows 0-40, 30-70, 60-100, 90-123
    as_retrieved = [RetrievedChunk(c.chunk_id, c.doc_id, c.text, 0.5, c.metadata) for c in doc_chunks]
    assert len(as_retrieved) == 4
    assert contiguous_passages(list(reversed(as_retrieved))) == [DOCS[7].text]  # order-independent
    # Windows 0-40 and 60-100 do not touch: words 40-60 were not retrieved, so they stay separate.
    assert contiguous_passages([as_retrieved[2], as_retrieved[0]]) == [doc_chunks[0].text, doc_chunks[2].text]


# --- Precision / recall ----------------------------------------------------------------------------

def test_precision_recall_arithmetic():
    r = precision_recall(["KB-06", "KB-06", "KB-01", "KB-07"], ["KB-06", "KB-07"])
    assert r["retrieved_docs"] == ["KB-06", "KB-01", "KB-07"]
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == 1.0
    assert r["precision_arithmetic"] == "2/3 = 0.667" and r["recall_arithmetic"] == "2/2 = 1.000"
    assert precision_recall([], ["KB-01"])["precision"] == 0.0
    with pytest.raises(ValueError):
        precision_recall(["KB-01"], [])
    assert dedupe(["b", "a", "b"]) == ["b", "a"]


def test_saved_chunking_comparison_matches_live_retrieval(collections):
    saved = json.loads((ROOT / "evaluation" / "chunking_comparison.json").read_text(encoding="utf-8"))
    assert saved["k"] == DEFAULT_TOP_K and len(saved["queries"]) >= 5
    for s in STRATEGIES:
        live = [precision_recall([c.doc_id for c in search(collections[s], q["query"], DEFAULT_TOP_K)],
                                 q["relevant_doc_ids"]) for q in QUERIES["evaluation"]]
        stored = saved["strategies"][s]
        assert [r["retrieved_docs"] for r in live] == [r["retrieved_docs"] for r in stored["per_query"]]
        assert stored["avg_precision"] == round(sum(r["precision"] for r in live) / len(live), 4)
        assert stored["avg_recall"] == round(sum(r["recall"] for r in live) / len(live), 4)


def test_recommended_strategy_is_backed_by_the_numbers():
    # Precision and recall disagree after the Q3 fix, so the documented criterion is F1 of the averages.
    saved = json.loads((ROOT / "evaluation" / "chunking_comparison.json").read_text(encoding="utf-8"))["strategies"]
    other = "sentence" if RECOMMENDED_STRATEGY == "fixed" else "fixed"
    for s in STRATEGIES:
        p, r = saved[s]["avg_precision"], saved[s]["avg_recall"]
        assert saved[s]["f1_of_averages"] == pytest.approx(2 * p * r / (p + r), abs=1e-4)
    assert saved[RECOMMENDED_STRATEGY]["f1_of_averages"] > saved[other]["f1_of_averages"]


# --- Return-vs-warranty regression (Phase 3 review fix) ---------------------------------------------

@pytest.mark.parametrize("q", INTENT_QUERIES, ids=lambda q: q["id"])
def test_query_analysis_routes_only_warranty_intent(q):
    assert analyse_query(q["query"]).topic == q["expected_intent"]


CALIBRATION_ESCALATION = "How do I reach a senior support specialist about an unresolved complaint?"


@pytest.mark.parametrize("query", [
    *[x["query"] for x in QUERIES["calibration"]["in_scope"] + QUERIES["calibration"]["out_of_scope"]
      if x["query"] != CALIBRATION_ESCALATION],  # routed to escalation since the Phase 7 remediation (below)
    QUERIES["out_of_scope_demo"]["query"],
    *[q["query"] for q in QUERIES["evaluation"] if q["id"] != "Q3"],
    "Is my car under warranty?",                                 # no Nykaa product category
    "My hair dryer stopped working after five days.",            # days = return/damage window, not warranty
    "The payment page stopped working after two minutes.",       # no product
    "My hair dryer arrived broken. What do I do?",               # damaged on arrival, no elapsed use
])
def test_query_analysis_does_not_route_other_queries(query):
    assert analyse_query(query).topic is None


# --- Escalation-vs-damaged-item regression (Phase 7 remediation, E12) -------------------------------

ESCALATION_QUERIES = json.loads((ROOT / "evaluation" / "rag_escalation_queries.json").read_text(encoding="utf-8"))["queries"]
E12 = "My order ORD-1022 was delayed. Will my complaint be escalated to a senior specialist?"


@pytest.mark.parametrize("q", ESCALATION_QUERIES, ids=lambda q: q["id"])
def test_escalation_routing_matches_frozen_expectation(q):
    assert analyse_query(q["query"]).topic == q["expected_route"]


# Documented, pre-existing misses of the NON-deployed sentence collection on unrouted queries (plain dense
# retrieval, unchanged by the remediation): D1 top-1 KB-11 (0.5062 vs KB-10 0.4931), D2 top-1 KB-01.
KNOWN_SENTENCE_MISSES = {"D1", "D2"}


@pytest.mark.parametrize("q", ESCALATION_QUERIES, ids=lambda q: q["id"])
def test_escalation_regression_top1_is_relevant_in_deployed_collection(collections, q):
    assert search(collections[RECOMMENDED_STRATEGY], q["query"], 1)[0].doc_id in q["relevant_doc_ids"]


def test_sentence_collection_misses_are_only_the_documented_unrouted_ones(collections):
    misses = {q["id"] for q in ESCALATION_QUERIES
              if search(collections["sentence"], q["query"], 1)[0].doc_id not in q["relevant_doc_ids"]}
    assert misses == KNOWN_SENTENCE_MISSES
    for q in ESCALATION_QUERIES:
        if q["id"] in KNOWN_SENTENCE_MISSES:  # not routed, so retrieval is exactly the pre-remediation dense search
            assert analyse_query(q["query"]).topic is None
            assert search(collections["sentence"], q["query"]) == retrieve(collections["sentence"], q["query"])


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_e12_retrieves_only_the_escalation_matrix(collections, strategy):
    assert analyse_query(E12).topic == "escalation_matrix" and analyse_query(E12).search_text == E12  # no expansion
    assert {c.doc_id for c in search(collections[strategy], E12, DEFAULT_TOP_K)} == {"KB-12"}
    a = answer_query(E12, collections[strategy], SIMILARITY_THRESHOLDS[strategy])
    assert a.grounded and a.sources == ["KB-12"] and "Level 2" in a.answer and "damaged" not in a.answer


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_routed_calibration_query_keeps_its_similarity(collections, strategy):
    # Escalation routing adds no expansion: the top chunk and its similarity are those of plain dense retrieval,
    # so the calibrated thresholds (re-derived live in test_threshold_reproduces_from_live_measurements) hold.
    assert analyse_query(CALIBRATION_ESCALATION).topic == "escalation_matrix"
    routed, plain = search(collections[strategy], CALIBRATION_ESCALATION, 1)[0], retrieve(collections[strategy],
                                                                                          CALIBRATION_ESCALATION, 1)[0]
    assert routed.chunk_id == plain.chunk_id and routed.similarity == pytest.approx(plain.similarity, abs=1e-9)


@pytest.mark.parametrize("query", [
    "My order ORD-1029 arrived late. Can I still return the shoes, and is my case being escalated?",  # Demo C
    "My damaged-item claim was rejected. Can I escalate it?",
    "Will my refund be escalated if it is late?",
])
def test_mixed_intent_escalation_questions_are_not_restricted(query):
    assert analyse_query(query).topic is None


# --- Category-specific answers (Phase 7 remediation, E01) --------------------------------------------

@pytest.mark.parametrize("query, keep, drop", [
    ("I bought a dress 20 days ago and haven't worn it. Can I still return it?", "Apparel can be returned",
     ["Beauty products", "Home items"]),
    ("Can I return an unopened lipstick?", "Beauty products can be returned", ["Apparel can", "Home items"]),
    ("How many days do I have to return a bedsheet?", "Home items", ["Beauty products", "Apparel can"]),
    ("Can I return a hair dryer that is faulty?", "Electronics such as hair dryers", ["Beauty products", "Home items"]),
])
def test_category_specific_return_answers_exclude_other_categories(collections, query, keep, drop):
    a = answer_query(query, collections["fixed"], SIMILARITY_THRESHOLDS["fixed"])
    assert keep in a.answer and not any(d in a.answer for d in drop), a.answer


def test_questions_without_a_category_are_not_filtered(collections):
    query = "How long do I have to return an item?"
    assert analyse_query(query).categories == ()
    a = answer_query(query, collections["fixed"], SIMILARITY_THRESHOLDS["fixed"])
    assert a.grounded and a.answer.startswith("Home items")  # a category-subject sentence survives: no filter


def test_query_analysis_detects_categories_and_expands_only_when_routed():
    a = analyse_query(Q3)
    assert a.categories == ("Electronics",) and a.topic == "warranty" and a.where == {"topic": "warranty"}
    assert a.search_text == f"{Q3} Electronics warranty"
    lipstick = analyse_query(QUERIES["evaluation"][0]["query"])
    assert lipstick.categories == ("Beauty",) and lipstick.search_text == lipstick.query and lipstick.where is None


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_plain_dense_retrieval_misranks_q3_root_cause(collections, strategy):
    # Documents the root cause the fix addresses: surface wording "hair dryer" pulls KB-01 to the top.
    assert retrieve(collections[strategy], Q3, 3)[0].doc_id == "KB-01"


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_q3_retrieves_the_electronics_warranty(collections, strategy):
    results = search(collections[strategy], Q3, DEFAULT_TOP_K)
    assert {r.doc_id for r in results} == {"KB-05"}
    assert "12-month manufacturer warranty" in results[0].text and results[0].text.startswith("Electronics")
    assert results[0].similarity > SIMILARITY_THRESHOLDS[strategy]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_q3_answer_states_12_month_warranty_not_return_window(collections, strategy):
    a = answer_query(Q3, collections[strategy], SIMILARITY_THRESHOLDS[strategy])
    assert a.grounded and a.routed_topic == "warranty" and a.sources == ["KB-05"]
    assert split_sentences(a.answer)[0].startswith("Electronics come with a 12-month manufacturer warranty")
    assert "7 days" not in a.answer and "returned within" not in a.answer


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("q", INTENT_QUERIES, ids=lambda q: q["id"])
def test_intent_regression_top1_is_relevant(collections, strategy, q):
    assert search(collections[strategy], q["query"], 1)[0].doc_id in q["relevant_doc_ids"]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_unrouted_queries_use_plain_dense_retrieval(collections, strategy):
    for q in QUERIES["evaluation"]:
        if q["id"] != "Q3":
            assert search(collections[strategy], q["query"]) == retrieve(collections[strategy], q["query"])
