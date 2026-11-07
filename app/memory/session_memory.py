"""Per-session conversation memory via LangChain RunnableWithMessageHistory."""

from __future__ import annotations

from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory

_store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    return _store.setdefault(session_id, InMemoryChatMessageHistory())


def has_session(session_id: str) -> bool:
    return session_id in _store


def clear_sessions() -> None:
    _store.clear()


def with_session_memory(pipeline: Runnable[dict[str, Any], dict[str, Any]]) -> RunnableWithMessageHistory:
    """Wrap a pipeline taking {"question", "history"} and returning {"answer", ...}."""
    return RunnableWithMessageHistory(
        pipeline,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
        output_messages_key="answer",
    )
