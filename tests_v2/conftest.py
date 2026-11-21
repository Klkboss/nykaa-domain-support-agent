"""v2 tests reuse the capstone's outbound-network guard (tests/conftest.py, unchanged).

When tests/ is collected first, pytest has already installed the guard; run alone, tests_v2 imports it. The
guard is found through the installed socket function itself (pytest swaps "conftest" in sys.modules between
directories), and every installed guard layer's record is checked, so a swallowed attempt fails the test.
Provider tests never use a real server: every client gets an httpx.MockTransport, and test base URLs are
non-loopback, so a transport that is not mocked hits the guard instead of a local Ollama.
"""

from __future__ import annotations

import socket

if getattr(socket.socket.connect, "__name__", "") != "_guarded_connect":
    import tests.conftest  # noqa: F401  (installs the guard)

import pytest  # noqa: E402


def _records() -> list[list[str]]:
    records, fn = [], socket.socket.connect
    while getattr(fn, "__name__", "") == "_guarded_connect":
        records.append(fn.__globals__["blocked_attempts"])
        fn = fn.__globals__["_real_connect"]
    return records


assert _records(), "network guard not installed"


@pytest.fixture(autouse=True)
def no_network_attempts_v2():
    yield
    attempts = [a for record in _records() for a in record]
    assert attempts == [], f"outbound network attempted: {attempts}"
