"""Spec Task 16: in-memory cache for the grounded-generation step ONLY (owner decision, Phase 0).

`CachedGenerator` wraps the generator call inside `answer_query`, so retrieval, the similarity threshold and every
guardrail around the crew still run on every request; a hit replaces only the one generation call.

Key: SHA-256 of the normalised query text + a fingerprint of the retrieved supporting context (chunk ids, texts,
metadata) + the generator configuration + the embedding model. A changed policy text or a different chunk set
therefore gives a different key, so a stale answer cannot be returned. Only the digest is stored, never the query.

Never cached: queries with an order id or PII (or a PII mask token), anything that is not generated (out-of-scope
fallback, refusals), and generated text that does not pass `claim_supported` against the retrieved context.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import Counter, OrderedDict

from app.guardrails.groundedness import claim_supported, evidence_by_doc
from app.guardrails.pii import CARD_LAST4_TOKEN, CARD_TOKEN, PHONE_TOKEN, mask_pii
from app.rag.documents import split_sentences
from app.rag.embeddings import MODEL_NAME
from app.rag.generation import MockGroundedGenerator
from app.rag.retrieval import RetrievedChunk

MAX_ENTRIES = 256  # bounds memory in a long-running API process; least recently used entries go first
_ORDER_ID = re.compile(r"ORD-?[0-9]{4}", re.I)  # wider than the tool's own pattern on purpose
_PII_TOKENS = (PHONE_TOKEN, CARD_TOKEN, CARD_LAST4_TOKEN)


def normalise_query(text: str) -> str:
    """Whitespace collapsed, lower-cased: the embedding model is uncased and query_analysis is case-insensitive."""
    return " ".join(text.split()).lower()


def cacheable_query(query: str) -> bool:
    """False for anything customer-specific: an order id, PII, or PII that was already masked."""
    return not (_ORDER_ID.search(query) or any(t in query for t in _PII_TOKENS) or mask_pii(query).masked)


def cache_key(query: str, chunks: list[RetrievedChunk], generator: MockGroundedGenerator) -> str:
    payload = {
        "query": normalise_query(query),
        "context": [[c.chunk_id, c.doc_id, c.text, c.metadata] for c in chunks],
        "generator": [type(generator).__qualname__, generator.max_sentences, generator.margin],
        "embedding_model": MODEL_NAME,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def is_supported_by(text: str, chunks: list[RetrievedChunk]) -> bool:
    """The groundedness guardrail's own test, applied before anything is stored."""
    evidence = [p for passages in evidence_by_doc(chunks).values() for p in passages]
    sentences = split_sentences(text) if text.strip() else []
    return bool(sentences) and all(claim_supported(s, evidence) for s in sentences)


class GenerationCache:
    """Thread-safe (the API answers in a thread pool), LRU-bounded map of key digest -> validated generated text."""

    def __init__(self, max_entries: int = MAX_ENTRIES) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()
        self.stats: Counter[str] = Counter()  # hit / miss / skipped / rejected / error: the before/after evidence

    def get(self, key: str) -> str | None:
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key]
            return None

    def put(self, key: str, text: str) -> None:
        with self._lock:
            self._entries[key] = text
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return dict(self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class CachedGenerator(MockGroundedGenerator):
    """Answers from the cache when the same normalised query meets the same retrieved context; otherwise calls the
    inner generator, validates its text against that context and stores it. A cache error falls back to generation."""

    def __init__(self, inner: MockGroundedGenerator, cache: GenerationCache, config: MockGroundedGenerator) -> None:
        super().__init__(config.max_sentences, config.margin)
        self._inner, self._cache, self._config = inner, cache, config
        self.event: str | None = None  # None: the generator was never called (e.g. nothing above the threshold)

    def _note(self, event: str) -> None:
        self.event = event
        self._cache.stats[event] += 1

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        if not cacheable_query(query):
            self._note("skipped")
            return self._inner.generate(query, chunks)
        key = None
        try:
            key = cache_key(query, chunks, self._config)
            if (text := self._cache.get(key)) is not None:
                self._note("hit")
                return text
        except Exception:  # a broken cache must never break an answer
            key = None
        text = self._inner.generate(query, chunks)  # BudgetExceeded from here is NOT caught
        if key is None:
            self._note("error")
            return text
        try:
            if is_supported_by(text, chunks):
                self._cache.put(key, text)
                self._note("miss")
            else:
                self._note("rejected")
        except Exception:
            self._note("error")
        return text
