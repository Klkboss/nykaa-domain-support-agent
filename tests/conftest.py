"""Test harness: offline env + an outbound-network guard installed before any library import.

Loopback stays allowed because asyncio on Windows builds its event-loop self-pipe with a
127.0.0.1 socket pair. Every other connect / DNS lookup is refused AND recorded, so an
attempt that a library swallows (telemetry does) still fails the test.
"""

from __future__ import annotations

import socket

import app.core.config  # noqa: F401  (sets offline env vars first)

LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}
blocked_attempts: list[str] = []

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_getaddrinfo = socket.getaddrinfo


def _host(address: object) -> str:
    return str(address[0]) if isinstance(address, tuple) else str(address)


def _guarded_connect(self: socket.socket, address: object) -> None:
    if _host(address) not in LOOPBACK:
        blocked_attempts.append(f"connect {address}")
        raise ConnectionRefusedError(f"network blocked in tests: {address}")
    return _real_connect(self, address)


def _guarded_connect_ex(self: socket.socket, address: object) -> int:
    if _host(address) not in LOOPBACK:
        blocked_attempts.append(f"connect_ex {address}")
        raise ConnectionRefusedError(f"network blocked in tests: {address}")
    return _real_connect_ex(self, address)


def _guarded_getaddrinfo(host: object, *args: object, **kwargs: object):
    if str(host) not in LOOPBACK:
        blocked_attempts.append(f"getaddrinfo {host}")
        raise socket.gaierror(f"DNS blocked in tests: {host}")
    return _real_getaddrinfo(host, *args, **kwargs)


socket.socket.connect = _guarded_connect  # type: ignore[method-assign]
socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[method-assign]
socket.getaddrinfo = _guarded_getaddrinfo  # type: ignore[assignment]

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def client():
    """In-memory Chroma with both strategies indexed from the real KB with real MiniLM embeddings."""
    import chromadb

    from app.rag.chunking import STRATEGIES
    from app.rag.documents import load_knowledge_base
    from app.rag.vector_store import COLLECTION_NAMES, index_all

    c = chromadb.EphemeralClient()
    for s in STRATEGIES:  # start clean: the in-process ephemeral store is shared
        if COLLECTION_NAMES[s] in [col.name for col in c.list_collections()]:
            c.delete_collection(COLLECTION_NAMES[s])
    index_all(c, load_knowledge_base())
    return c


@pytest.fixture(scope="session")
def collections(client):
    from app.rag.chunking import STRATEGIES
    from app.rag.vector_store import get_collection

    return {s: get_collection(client, s) for s in STRATEGIES}


@pytest.fixture(autouse=True)
def no_network_attempts():
    # Not cleared on entry: an attempt during import or a session fixture also fails the next test.
    yield blocked_attempts
    assert blocked_attempts == [], f"outbound network attempted: {blocked_attempts}"
