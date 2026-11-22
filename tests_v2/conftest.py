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


@pytest.fixture(scope="session")
def collections(tmp_path_factory):
    """tests_v2's OWN index (real KB, real MiniLM embeddings) in an on-disk Chroma store in a temp dir.

    Not the capstone's session-wide in-memory index: Chroma 1.5.9's in-memory store can break mid-process
    ("Error finding id", PROJECT_STATUS.md) and then every later search returns nothing, which once failed 12
    tests here in a full run. The on-disk client is a separate store (the deployed API uses the same kind), so v2
    tests no longer share that state with the capstone tests.
    """
    from app.rag.chunking import STRATEGIES
    from app.rag.documents import load_knowledge_base
    from app.rag.retrieval import search
    from app.rag.vector_store import get_collection, index_all, persistent_client

    store = persistent_client(tmp_path_factory.mktemp("v2_chroma"))
    index_all(store, load_knowledge_base())
    cols = {s: get_collection(store, s) for s in STRATEGIES}
    for s, col in cols.items():  # fail here, clearly, rather than as many unrelated-looking test failures
        assert search(col, "How long do I have to return shoes?"), f"v2 test index {s} returns no results"
    return cols

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
    for name in (*REAL_ENV, "LLM_API_KEY", "LLM_API_KEY_ENV", "LLM_REASONING_EFFORT", "OLLAMA_BASE_URL",
                 "LLM_TEMPERATURE", "LLM_TIMEOUT_S", "LLM_MAX_TOKENS", "LLM_FALLBACK_BASE_URL", "LLM_FALLBACK_MODEL",
                 "LLM_FALLBACK_API_KEY", "LLM_FALLBACK_API_KEY_ENV", "LLM_FALLBACK_REASONING_EFFORT",
                 "RATE_LIMIT_SITE_PER_DAY",
                 "JUDGE_PROVIDER", "DATABASE_URL"):
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
                            lambda s, b=None, **kw: review_build(s, b, http_client=fake.async_client(), **kw))
        return fake

    return route
