"""Process-wide settings. Import this module before crewai, chromadb or autogen."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Forced (not setdefault) so a stray shell/.env value cannot re-enable outbound telemetry.
_OFFLINE_ENV = {
    "CREWAI_DISABLE_TELEMETRY": "true",
    "CREWAI_TRACING_ENABLED": "false",
    "OTEL_SDK_DISABLED": "true",
    "CREWAI_DISABLE_VERSION_CHECK": "true",  # verbose crews otherwise query pypi.org on every kickoff
    "ANONYMIZED_TELEMETRY": "False",  # chromadb
    "CREWAI_STORAGE_DIR": str(ROOT / ".crewai_storage"),
    # The embedding model is loaded from models/ (fetched once by scripts/download_model.py).
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
os.environ.update(_OFFLINE_ENV)

MOCK_LLM: bool = os.getenv("MOCK_LLM", "true").strip().lower() == "true"
