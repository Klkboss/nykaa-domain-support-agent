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
# The capstone's in-memory Chroma fixtures (real KB, real MiniLM embeddings), from the same guard module.
client = socket.socket.connect.__globals__["client"]
collections = socket.socket.connect.__globals__["collections"]

REAL_ENV = {"LLM_PROVIDER": "openai_compatible", "LLM_BASE_URL": "http://llm.test/v1", "LLM_MODEL": "fake-model",
            "LLM_MAX_RETRIES": "0"}


@pytest.fixture(autouse=True)
def no_network_attempts_v2():
    yield
    attempts = [a for record in _records() for a in record]
    assert attempts == [], f"outbound network attempted: {attempts}"


@pytest.fixture(autouse=True)
def no_ambient_llm_settings(monkeypatch):
    """A developer's shell LLM_* / JUDGE_PROVIDER values must never leak into a test."""
    for name in (*REAL_ENV, "LLM_API_KEY", "OLLAMA_BASE_URL", "LLM_TEMPERATURE", "LLM_TIMEOUT_S", "LLM_MAX_TOKENS",
                 "JUDGE_PROVIDER"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def real_provider(monkeypatch):
    """MOCK_LLM off and a real (fake-server) provider configured; returns a function that routes every model client
    (crew LLMs, AutoGen clients) to a given fake. Generators are injected through the rag tool factory."""
    import app.agents.crew as crew_module
    import app.review.autogen_review as review_module
    from app.core import config

    monkeypatch.setattr(config, "MOCK_LLM", False)
    for name, value in REAL_ENV.items():
        monkeypatch.setenv(name, value)

    def route(fake):
        crew_build, review_build = crew_module.build_crew_llm, review_module.build_review_client
        monkeypatch.setattr(crew_module, "build_crew_llm",
                            lambda s, b=None, **kw: crew_build(s, b, http_client=fake.sync_client(), **kw))
        monkeypatch.setattr(review_module, "build_review_client",
                            lambda s, b=None: review_build(s, b, http_client=fake.async_client()))
        return fake

    return route
