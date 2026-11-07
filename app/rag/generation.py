"""Grounded generation over retrieved chunks (spec Task 4). Separate from retrieval, CrewAI and the API.

Under MOCK_LLM the generator is extractive and deterministic. It can only output sentences that occur
in the retrieved chunks that passed the threshold, so it cannot invent policy text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from chromadb.api.models.Collection import Collection

from app.rag.chunking import Strategy
from app.rag.documents import split_sentences
from app.rag.embeddings import embed
from app.rag.evaluation import dedupe
from app.rag.query_analysis import PRODUCT_TERMS, analyse_query
from app.rag.retrieval import RetrievedChunk, search

FALLBACK_ANSWER = "I don't have sufficient information in my knowledge base to answer that question."
DEFAULT_TOP_K = 3

# Calibrated in Phase 3 from measured top-1 cosine similarities: midpoint of the gap between the
# in-scope and out-of-scope clusters, per collection. Source: evaluation/rag_threshold_calibration.json.
#   fixed:    min in-scope 0.4247, max out-of-scope 0.2360 -> 0.330
#   sentence: min in-scope 0.4120, max out-of-scope 0.1910 -> 0.301
SIMILARITY_THRESHOLDS: dict[Strategy, float] = {"fixed": 0.33, "sentence": 0.301}
# Task 5 (evaluation/chunking_comparison.json, search() pipeline): fixed P 0.8056 R 1.0000 F1 0.8923;
# sentence P 0.8333 R 0.9167 F1 0.8730. Decided on F1 because precision and recall disagree.
RECOMMENDED_STRATEGY: Strategy = "fixed"


@dataclass(frozen=True)
class GroundedAnswer:
    query: str
    answer: str
    grounded: bool
    top_similarity: float
    threshold: float
    sources: list[str]
    chunks: list[RetrievedChunk]
    routed_topic: str | None = None


def contiguous_passages(chunks: list[RetrievedChunk]) -> list[str]:
    """Rejoin overlapping fixed-size windows of one document (via word offsets) so the overlap is not
    repeated and fewer sentences are cut. Sentence chunks pass through unchanged."""
    passages: list[str] = []
    windows: dict[str, list[RetrievedChunk]] = {}
    for c in chunks:
        if "word_start" in c.metadata:
            windows.setdefault(c.doc_id, []).append(c)
        else:
            passages.append(c.text)
    for doc_chunks in windows.values():
        doc_chunks.sort(key=lambda c: c.metadata["word_start"])
        words, end = doc_chunks[0].text.split(), doc_chunks[0].metadata["word_end"]
        for c in doc_chunks[1:]:
            start = c.metadata["word_start"]
            if start <= end:
                words += c.text.split()[end - start:]
            else:  # a gap: the words in between were not retrieved, so keep the passages apart
                passages.append(" ".join(words))
                words = c.text.split()
            end = max(end, c.metadata["word_end"])
        passages.append(" ".join(words))
    return passages


def _is_complete(sentence: str) -> bool:
    return sentence[0].isupper() and sentence[-1] in ".!?"


_SUBJECT_RE = {
    category: re.compile(r"\b(" + "|".join([re.escape(category.lower())] + [re.escape(t) for t in terms]) + r")s?\b")
    for category, terms in PRODUCT_TERMS.items()
}


def subject_categories(sentence: str) -> set[str]:
    """Categories named in a sentence's first four words: its subject in the KB's '<Category> ...' rules."""
    head = " ".join(sentence.lower().split()[:4])
    return {c for c, pattern in _SUBJECT_RE.items() if pattern.search(head)}


class MockGroundedGenerator:
    """MOCK_LLM generator: returns retrieved sentences most similar to the query, verbatim.

    Only sentences scoring within `margin` of the best are eligible, so an unrelated neighbour is never
    padded in. Among eligible ones, complete sentences are preferred over the fragments that fixed-size
    windows cut at their edges; a relevant fragment still beats an irrelevant complete sentence.
    When the question names a product category, sentences whose subject is a DIFFERENT category are dropped
    (Phase 7 remediation, E01: a dress question no longer gets the Beauty 15-day rule); if that would leave
    nothing, all candidates are kept.
    """

    def __init__(self, max_sentences: int = 2, margin: float = 0.10) -> None:
        self.max_sentences = max_sentences
        self.margin = margin
        self.calls = 0  # Phase 13 uses this counter as cache-hit evidence

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        self.calls += 1
        candidates = dedupe([s for p in contiguous_passages(chunks) for s in split_sentences(p)])
        if asked := set(analyse_query(query).categories):
            on_topic = [s for s in candidates if not (subject := subject_categories(s)) or subject & asked]
            candidates = on_topic or candidates
        vectors = np.array(embed([query] + candidates))
        scores = vectors[1:] @ vectors[0]
        ranked = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))  # index breaks ties
        eligible = [i for i in ranked if scores[i] >= scores[ranked[0]] - self.margin]
        complete = [i for i in eligible if _is_complete(candidates[i])]
        return " ".join(candidates[i] for i in (complete or eligible)[: self.max_sentences])


def answer_query(
    query: str,
    collection: Collection,
    threshold: float,
    k: int = DEFAULT_TOP_K,
    generator: MockGroundedGenerator | None = None,
) -> GroundedAnswer:
    analysis = analyse_query(query)
    chunks = search(collection, query, k)
    top = chunks[0].similarity if chunks else 0.0
    supporting = [c for c in chunks if c.similarity >= threshold]
    if not supporting:
        return GroundedAnswer(query, FALLBACK_ANSWER, False, top, threshold, [], chunks, analysis.topic)
    # Sentences are ranked against the same (possibly expanded) text the chunks were retrieved with.
    text = (generator or MockGroundedGenerator()).generate(analysis.search_text, supporting)
    return GroundedAnswer(query, text, True, top, threshold, dedupe([c.doc_id for c in supporting]), chunks,
                          analysis.topic)
