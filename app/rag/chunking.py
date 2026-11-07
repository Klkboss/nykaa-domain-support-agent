"""Two chunking strategies over KBDocument (spec Task 3).

fixed:    word windows of FIXED_CHUNK_WORDS with FIXED_OVERLAP_WORDS shared between neighbours.
sentence: SENTENCES_PER_CHUNK consecutive sentences, using the shared split_sentences().
Chunk ids are deterministic (`KB-01::fixed::00`), so re-indexing with upsert never duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.rag.documents import KBDocument, split_sentences

Strategy = Literal["fixed", "sentence"]
STRATEGIES: tuple[Strategy, ...] = ("fixed", "sentence")

# Documents are 88-126 words; sentences median 22 words (measured in Phase 3). A 40-word window is
# about two sentences, far below MiniLM's 256 word-piece limit. A 10-word (25%) overlap keeps any
# clause of up to 10 words intact in at least one chunk.
FIXED_CHUNK_WORDS = 40
FIXED_OVERLAP_WORDS = 10
SENTENCES_PER_CHUNK = 1


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    strategy: Strategy
    index: int
    text: str
    metadata: dict[str, str | int | bool] = field(hash=False)


def _chunk(doc: KBDocument, strategy: Strategy, index: int, text: str, span: dict[str, int]) -> Chunk:
    metadata = doc.chroma_metadata() | {"strategy": strategy, "chunk_index": index} | span
    return Chunk(f"{doc.doc_id}::{strategy}::{index:02d}", doc.doc_id, strategy, index, text, metadata)


def fixed_size_chunks(
    doc: KBDocument, size: int = FIXED_CHUNK_WORDS, overlap: int = FIXED_OVERLAP_WORDS
) -> list[Chunk]:
    if size < 1 or not 0 <= overlap < size:
        raise ValueError(f"need size >= 1 and 0 <= overlap < size, got size={size}, overlap={overlap}")
    words = doc.text.split()
    chunks: list[Chunk] = []
    start = 0
    while True:
        end = min(start + size, len(words))
        chunks.append(_chunk(doc, "fixed", len(chunks), " ".join(words[start:end]),
                             {"word_start": start, "word_end": end}))
        if end == len(words):  # a document shorter than `size` yields exactly one chunk
            return chunks
        start = end - overlap


def sentence_chunks(doc: KBDocument, per_chunk: int = SENTENCES_PER_CHUNK) -> list[Chunk]:
    if per_chunk < 1:
        raise ValueError(f"per_chunk must be >= 1, got {per_chunk}")
    sentences = split_sentences(doc.text)
    return [
        _chunk(doc, "sentence", i // per_chunk, " ".join(sentences[i:i + per_chunk]),
               {"sentence_start": i, "sentence_end": min(i + per_chunk, len(sentences))})
        for i in range(0, len(sentences), per_chunk)
    ]


def chunk_documents(docs: list[KBDocument], strategy: Strategy) -> list[Chunk]:
    if strategy == "fixed":
        return [c for d in docs for c in fixed_size_chunks(d)]
    if strategy == "sentence":
        return [c for d in docs for c in sentence_chunks(d)]
    raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
