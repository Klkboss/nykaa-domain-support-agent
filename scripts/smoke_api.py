"""v2 smoke test of a RUNNING API (docker compose, CI): waits until it is up, then asks one policy question and
one order question in the same session over HTTP and checks both were answered.

Usage: python scripts/smoke_api.py [base_url]     (default http://127.0.0.1:8000; exit code 1 on failure)
"""

from __future__ import annotations

import sys
import time
import uuid

import httpx

QUESTIONS = ["How long do I have to return shoes?", "What is the status of order ORD-1016?"]


def wait_until_up(base: str, timeout_s: float = 240) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            httpx.get(f"{base}/openapi.json", timeout=5).raise_for_status()
            return
        except httpx.HTTPError:
            if time.monotonic() > deadline:
                raise
            time.sleep(2)


def main(base: str = "http://127.0.0.1:8000") -> int:
    wait_until_up(base)
    session_id, ok = f"smoke-{uuid.uuid4().hex[:8]}", True
    for question in QUESTIONS:
        body = httpx.post(f"{base}/ask", json={"question": question, "session_id": session_id},
                          timeout=300).raise_for_status().json()
        print(f"{question!r} -> {body['status']}: {body['reply'][:100]!r}")
        ok &= body["status"] == "answered" and bool(body["reply"])
    print("SMOKE", "PASSED" if ok else "FAILED", f"(session {session_id})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
