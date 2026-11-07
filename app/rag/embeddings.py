"""Local SentenceTransformers embeddings (spec Task 3). CPU inference; no network at runtime.

The model is fetched once by `scripts/download_model.py` into models/all-MiniLM-L6-v2 and loaded
from that folder with HF offline mode forced on (app/core/config.py). A missing model raises.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.core.config import ROOT

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = ROOT / "models" / "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class ModelNotDownloadedError(FileNotFoundError):
    pass


@lru_cache(maxsize=1)
def get_model():
    if not (MODEL_DIR / "modules.json").exists():
        raise ModelNotDownloadedError(
            f"Embedding model not found at {MODEL_DIR}. Run once with internet: "
            r".venv\Scripts\python.exe scripts\download_model.py"
        )
    from sentence_transformers import SentenceTransformer  # heavy import, deferred until needed
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()  # keeps transcripts readable
    return SentenceTransformer(str(MODEL_DIR), device="cpu")


def embed(texts: list[str]) -> list[list[float]]:
    """L2-normalised embeddings, so cosine similarity equals the dot product."""
    if not texts or any(not t.strip() for t in texts):
        raise ValueError("embed() needs a non-empty list of non-blank texts")
    vectors = get_model().encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    if vectors.shape != (len(texts), EMBEDDING_DIM) or not np.isfinite(vectors).all():
        raise ValueError(f"unexpected embeddings: shape {vectors.shape}, finite={np.isfinite(vectors).all()}")
    return vectors.tolist()
