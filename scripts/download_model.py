"""One-time, explicit model download (needs internet). Everything else runs offline.

Usage: .venv\\Scripts\\python.exe scripts\\download_model.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.core.config  # noqa: E402,F401  (forces offline mode; lifted below for this script only)

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

from app.rag.embeddings import EMBEDDING_DIM, MODEL_DIR, MODEL_NAME  # noqa: E402

# Only what SentenceTransformers needs (the repo also ships ONNX/OpenVINO/TF weights we never use).
ALLOW = ["*.json", "vocab.txt", "model.safetensors", "1_Pooling/*"]
# v2: the revision the retrieval thresholds were calibrated on, so Docker/CI builds embed identically.
# MODEL_REVISION=latest fetches the newest one instead (then re-run scripts/run_rag_evaluation.py).
PINNED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def main() -> None:
    revision = os.environ.get("MODEL_REVISION", PINNED_REVISION)
    if revision == "latest":
        revision = HfApi().model_info(MODEL_NAME).sha
    path = snapshot_download(MODEL_NAME, revision=revision, local_dir=MODEL_DIR, allow_patterns=ALLOW)
    (MODEL_DIR / "REVISION.txt").write_text(f"{MODEL_NAME}@{revision}\n", encoding="utf-8")
    size_mb = sum(f.stat().st_size for f in MODEL_DIR.rglob("*") if f.is_file()) / 1e6
    print(f"Downloaded {MODEL_NAME}@{revision} to {path} ({size_mb:.1f} MB)")

    os.environ["HF_HUB_OFFLINE"] = "1"  # verify the saved copy loads with the hub disabled
    from app.rag.embeddings import embed

    vector = embed(["offline load check"])[0]
    assert len(vector) == EMBEDDING_DIM
    print(f"Offline load OK: dimension {len(vector)}")


if __name__ == "__main__":
    main()
