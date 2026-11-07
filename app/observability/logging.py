"""Structured JSON-Lines request logging (spec Task 12).

One JSON object per line, ELK-style. Every string value is passed through the same `mask_pii` the model input
gets before it is written, so a fixed-format PII field never reaches disk in the clear, even if a caller
passes something unexpected (e.g. a phone number used as a session id). Callers log masked text and exception
TYPE names only; raw request bodies, prompts and exception messages are never logged.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import ROOT
from app.guardrails.pii import mask_pii

DEFAULT_LOG_PATH = ROOT / "logs" / "requests.jsonl"


def new_trace_id() -> str:
    return uuid.uuid4().hex


def _sanitise(value: Any) -> Any:
    if isinstance(value, str):
        return mask_pii(value).text
    if isinstance(value, dict):
        return {str(k): _sanitise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitise(v) for v in value]
    return value


class JsonlLogger:
    def __init__(self, path: Path = DEFAULT_LOG_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, **fields: Any) -> dict[str, Any]:
        record = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **_sanitise(fields)}
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return record
