"""Per-session conversation memory via LangChain RunnableWithMessageHistory.

v2: when DATABASE_URL is set, history lives in PostgreSQL (app.db.repositories.SQLChatMessageHistory) and survives
restarts; otherwise it is the capstone's in-process store, unchanged.
"""

from __future__ import annotations

from typing import Any

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.db import database_url

_store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if database_url():
        from app.db.repositories import SQLChatMessageHistory

        return SQLChatMessageHistory(session_id)
    return _store.setdefault(session_id, InMemoryChatMessageHistory())


def has_session(session_id: str) -> bool:
    if database_url():
        from app.db.repositories import session_exists

        return session_exists(session_id)
    return session_id in _store


def clear_sessions() -> None:
    """In-process sessions only: never deletes a database's chat history."""
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
