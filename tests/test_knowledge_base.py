"""Spec Task 2: knowledge-base coverage, structure, quality and RAG-ingestion compatibility."""

from __future__ import annotations

import itertools
import json
import re

import pydantic
import pytest

from app.rag.documents import (
    KB_PATH, MAX_SENTENCES, MIN_SENTENCES, REQUIRED_TOPICS, KBDocument, KnowledgeBase,
    load_knowledge_base, split_sentences,
)

DOCS = load_knowledge_base()
BY_ID = {d.doc_id: d for d in DOCS}
GIVEN_CATEGORIES = ("Apparel", "Electronics", "Home", "Footwear", "Beauty")


# --- Loading, count, coverage -------------------------------------------------------------

def test_loads_programmatically_with_at_least_12_documents():
    assert all(isinstance(d, KBDocument) for d in DOCS)
    assert len(DOCS) >= 12


def test_every_required_topic_is_covered_and_no_unknown_topics():
    topics = {d.topic for d in DOCS}
    assert set(REQUIRED_TOPICS) <= topics, f"missing: {set(REQUIRED_TOPICS) - topics}"
    assert topics <= set(REQUIRED_TOPICS), f"unexpected: {topics - set(REQUIRED_TOPICS)}"


def test_doc_ids_unique_and_stable():
    ids = [d.doc_id for d in DOCS]
    assert len(set(ids)) == len(ids)
    # Phase 3 maps chunks back to these ids for precision/recall; keep them fixed.
    assert ids == [f"KB-{i:02d}" for i in range(1, 13)]


# --- Sentence counts ----------------------------------------------------------------------

@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.doc_id)
def test_two_to_five_sentences(doc):
    sentences = split_sentences(doc.text)
    assert MIN_SENTENCES <= len(sentences) <= MAX_SENTENCES
    # Cross-check the splitter: count sentence-final punctuation independently.
    assert len(re.findall(r"[.!?](?=\s|$)", doc.text)) == len(sentences)
    assert all(s[-1] in ".!?" for s in sentences)


@pytest.mark.parametrize("text, expected", [
    ("Refunds take 2.5 days. Points expire later.", 2),
    ("Use any method, e.g. the app. Then wait.", 2),
    ("It costs ₹100 to ship. ₹50 is refunded.", 2),
    ("One sentence only", 1),
    ("Is it late? Yes! Contact us.", 3),
])
def test_sentence_splitter(text, expected):
    assert len(split_sentences(text)) == expected


# --- Metadata and content quality --------------------------------------------------------

@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.doc_id)
def test_required_metadata_present_and_non_empty(doc):
    assert doc.doc_id and doc.title.strip() and doc.topic and doc.categories and doc.text.strip()
    assert doc.simulated is True
    assert set(doc.categories) <= set(GIVEN_CATEGORIES) | {"All"}
    assert doc.categories == ["All"] or "All" not in doc.categories


def test_file_carries_simulated_disclaimer():
    kb = KnowledgeBase.model_validate_json(KB_PATH.read_text(encoding="utf-8"))
    assert "SIMULATED" in kb.disclaimer and "not official" in kb.disclaimer


def test_no_duplicate_or_near_duplicate_documents():
    assert len({d.title.lower() for d in DOCS}) == len(DOCS)
    assert len({d.text for d in DOCS}) == len(DOCS)

    def words(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3}

    worst = max(
        (len(words(a.text) & words(b.text)) / len(words(a.text) | words(b.text)), a.doc_id, b.doc_id)
        for a, b in itertools.combinations(DOCS, 2)
    )
    assert worst[0] < 0.35, f"near-duplicate pair: {worst}"


def test_category_specific_documents_name_every_given_category():
    for doc_id in ("KB-01", "KB-05"):  # return window and warranty are "by category" in the spec
        doc = BY_ID[doc_id]
        assert set(doc.categories) == set(GIVEN_CATEGORIES)
        assert all(c in doc.text for c in GIVEN_CATEGORIES), doc_id
    assert BY_ID["KB-09"].categories == ["Apparel", "Footwear"]


# --- Cross-document consistency (facts stated in more than one document must agree) ----

@pytest.mark.parametrize("fact, doc_ids", [
    ("within 30 days of delivery", ["KB-01"]),
    ("30-day window from delivery", ["KB-09"]),             # size exchange = apparel/footwear return window
    ("5 to 7 business days", ["KB-02", "KB-06", "KB-08"]),  # bank / original-method refunds
    ("within 48 hours of delivery", ["KB-10", "KB-11"]),    # damaged-item claim window
    ("within 24 hours", ["KB-06", "KB-07"]),                # loyalty points restored
    ("₹10,000", ["KB-08", "KB-12"]),
])
def test_shared_facts_are_stated_consistently(fact, doc_ids):
    for doc_id in doc_ids:
        assert fact in BY_ID[doc_id].text, (fact, doc_id)


HISTORY_KB08 = json.loads((KB_PATH.parent / "history" / "KB-08_wording_v1.json").read_text(encoding="utf-8"))


def test_kb08_retry_and_attempt_counts_are_consistent():
    """Owner-approved clarification: the first attempt is separate from retries; 3 retries; at most 4 attempts."""
    text = BY_ID["KB-08"].text
    assert "If the first payment attempt fails" in text and "retry the payment up to 3 more times within 30 minutes" in text
    assert "Once all 3 retries have also failed, the customer should switch to another payment method" in text
    assert "The first attempt plus 3 retries means at most 4 payment attempts in total." in text
    assert set(re.findall(r"(\d+) (?:more times|retries)", text)) == {"3"}  # every retry count is 3
    assert set(re.findall(r"(\d+) (?:payment )?attempts", text)) == {"4"}  # every attempt total is 4
    assert "failed attempts" not in text  # the v1 phrase that counted attempts, not retries


def test_kb08_clarification_changed_only_the_count_rules():
    old, new = HISTORY_KB08["document"], BY_ID["KB-08"].model_dump()
    assert {k: v for k, v in old.items() if k != "text"} == {k: v for k, v in new.items() if k != "text"}
    old_s, new_s = split_sentences(old["text"]), split_sentences(new["text"])
    assert old_s[1] == new_s[1] and old_s[3] == new_s[3]  # debit reversal and COD limit: verbatim
    assert HISTORY_KB08["wording_version"] == 1 and "four total attempts" in HISTORY_KB08["reason_for_clarification"]


def test_no_conflicting_damaged_claim_window():
    windows = {m for d in DOCS for m in re.findall(r"within (\d+) hours of delivery", d.text)}
    assert windows == {"48"}


def test_size_exchange_excludes_categories_without_sizes():
    text = BY_ID["KB-09"].text
    assert "Beauty, Home and Electronics products are not eligible" in text


# --- Validation rejects bad documents -------------------------------------------------------

VALID = {"doc_id": "KB-99", "title": "Test Policy", "topic": "test_topic", "categories": ["All"],
         "text": "First sentence here. Second sentence here."}


@pytest.mark.parametrize("change", [
    {"text": "Only one sentence."},
    {"text": "One. Two. Three. Four. Five. Six."},
    {"text": ""},
    {"doc_id": "DOC-1"},
    {"title": ""},
    {"topic": "Bad Topic"},
    {"categories": []},
    {"categories": ["Toys"]},
    {"unexpected": "field"},
])
def test_kbdocument_rejects_invalid_input(change):
    with pytest.raises(pydantic.ValidationError):
        KBDocument.model_validate(VALID | change)


def test_loader_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "kb.json"
    path.write_text(json.dumps({"disclaimer": "SIMULATED test data, not official.",
                                "documents": [VALID, VALID]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate doc_id"):
        load_knowledge_base(path)


# --- Compatibility with the Phase 3 ingestion interface ---------------------------------------

def test_chroma_metadata_is_flat_scalars_with_doc_id():
    for d in DOCS:
        meta = d.chroma_metadata()
        assert meta["doc_id"] == d.doc_id
        assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())


def test_documents_upsert_into_a_real_chroma_collection():
    """Offline: explicit dummy embeddings, so Chroma's default model is never downloaded."""
    import chromadb

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection("kb_ingestion_check", embedding_function=None)
    collection.upsert(
        ids=[d.doc_id for d in DOCS],
        documents=[d.text for d in DOCS],
        metadatas=[d.chroma_metadata() for d in DOCS],
        embeddings=[[float(i), 1.0, 0.0] for i in range(len(DOCS))],
    )
    got = collection.get(ids=["KB-09"], include=["metadatas", "documents"])
    assert got["metadatas"][0]["topic"] == "size_exchange"
    assert got["documents"][0] == BY_ID["KB-09"].text
    assert collection.count() == len(DOCS)
    client.delete_collection("kb_ingestion_check")
