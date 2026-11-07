"""Top-k retrieval with cosine similarity (spec Task 4). Independent of CrewAI and the API.

`retrieve` is plain dense search. `search` is the pipeline every caller uses (answering, calibration and
precision/recall): metadata-aware query analysis, then dense search.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chromadb.api.models.Collection import Collection

from app.rag.embeddings import embed
from app.rag.query_analysis import analyse_query


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    similarity: float
    metadata: dict = field(hash=False)


def distance_to_similarity(distance: float) -> float:
    """Chroma's cosine space returns distance = 1 - cosine similarity."""
    return 1.0 - distance


def retrieve(collection: Collection, query: str, k: int = 3, where: dict | None = None) -> list[RetrievedChunk]:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not query.strip():
        raise ValueError("query must not be blank")
    result = collection.query(
        query_embeddings=embed([query]), n_results=k, where=where, include=["documents", "metadatas", "distances"]
    )
    return [
        RetrievedChunk(chunk_id, meta["doc_id"], text, distance_to_similarity(dist), meta)
        for chunk_id, text, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def search(collection: Collection, query: str, k: int = 3) -> list[RetrievedChunk]:
    """Retrieval pipeline: route by detected intent (metadata filter + expansion), then dense top-k."""
    analysis = analyse_query(query)
    return retrieve(collection, analysis.search_text, k, analysis.where)
