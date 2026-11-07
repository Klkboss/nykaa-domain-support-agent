"""One ChromaDB collection per chunking strategy, cosine space, filled with collection.upsert()."""

from __future__ import annotations

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from app.core.config import ROOT
from app.rag.chunking import STRATEGIES, Strategy, chunk_documents
from app.rag.documents import KBDocument
from app.rag.embeddings import embed

COLLECTION_NAMES: dict[Strategy, str] = {
    "fixed": "policies_fixed_overlap",
    "sentence": "policies_sentence_based",
}
CHROMA_DIR = ROOT / "chroma_db"


def persistent_client(path=CHROMA_DIR) -> ClientAPI:
    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def get_collection(client: ClientAPI, strategy: Strategy) -> Collection:
    # embedding_function=None: we always pass our own MiniLM vectors, so Chroma never loads its default model.
    return client.get_or_create_collection(
        COLLECTION_NAMES[strategy], embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
    )


def index_documents(client: ClientAPI, docs: list[KBDocument], strategy: Strategy) -> int:
    """Upsert every chunk of `docs`; drop stale chunks if a document now has fewer. Returns chunk count."""
    chunks = chunk_documents(docs, strategy)
    collection = get_collection(client, strategy)
    new_ids = {c.chunk_id for c in chunks}
    existing = collection.get(where={"doc_id": {"$in": [d.doc_id for d in docs]}}, include=[])["ids"]
    if stale := [i for i in existing if i not in new_ids]:
        collection.delete(ids=stale)
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
        embeddings=embed([c.text for c in chunks]),
    )
    return len(chunks)


def index_all(client: ClientAPI, docs: list[KBDocument]) -> dict[Strategy, int]:
    return {s: index_documents(client, docs, s) for s in STRATEGIES}
