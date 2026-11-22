"""v2 repositories: the only code that reads or writes the database. Callers decide WHETHER to use it
(DATABASE_URL set); these functions assume it is configured.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.db import session
from app.db.models import ChatMessage, ChatSession, Order, RequestLog

_ROLES = {"human": HumanMessage, "ai": AIMessage}


def get_order(record_id: str) -> dict[str, Any] | None:
    """One order as the same dict shape as data.dataset.ORDERS, or None."""
    with session() as s:
        row = s.get(Order, record_id)
        if row is None:
            return None
        return {"record_id": row.record_id, "category": row.category, "status": row.status,
                "order_value_inr": row.order_value_inr, "days_since_created": row.days_since_created,
                "delayed_shipment": row.delayed_shipment}


def session_exists(session_id: str) -> bool:
    with session() as s:
        return s.get(ChatSession, session_id) is not None


class SQLChatMessageHistory(BaseChatMessageHistory):
    """A session's history in chat_sessions / chat_messages, for LangChain's RunnableWithMessageHistory.

    The pipeline only ever stores masked questions and guardrail-checked replies, so that is all this holds.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        with session() as s:
            rows = s.execute(select(ChatMessage.role, ChatMessage.content)
                             .where(ChatMessage.session_id == self.session_id).order_by(ChatMessage.id)).all()
        return [_ROLES[role](content=content) for role, content in rows]

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        with session() as s, s.begin():
            # One atomic upsert, not check-then-insert: simultaneous first writes to a new session must all succeed.
            upsert = pg_insert if s.get_bind().dialect.name == "postgresql" else sqlite_insert
            stmt = upsert(ChatSession).values(id=self.session_id)
            s.execute(stmt.on_conflict_do_update(index_elements=[ChatSession.id], set_={"updated_at": func.now()}))
            s.add_all([ChatMessage(session_id=self.session_id, role=m.type, content=str(m.content))
                       for m in messages])

    def clear(self) -> None:
        with session() as s, s.begin():
            s.execute(delete(ChatSession).where(ChatSession.id == self.session_id))


def log_record(record: dict[str, Any]) -> None:
    """Store one already-sanitised JsonlLogger record."""
    with session() as s, s.begin():
        s.add(RequestLog(type=str(record.get("type", "unknown"))[:32], trace_id=record.get("trace_id"),
                         record=record))
