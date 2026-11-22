"""v2 database schema (SQLAlchemy 2). The Alembic migration in migrations/versions/ creates exactly this schema
(tests_v2/test_v2_db.py compares the two). Every column a person could type into holds MASKED text only.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from data.dataset import MAX_DAYS, REQUIRED_CATEGORIES, REQUIRED_STATUSES

NAMING = {"ix": "ix_%(column_0_label)s", "uq": "uq_%(table_name)s_%(column_0_name)s",
          "ck": "ck_%(table_name)s_%(constraint_name)s",
          "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s", "pk": "pk_%(table_name)s"}
# BIGINT identity on Postgres; SQLite (unit tests) only autoincrements an INTEGER primary key.
BigId = BigInteger().with_variant(Integer, "sqlite")


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)


class Order(Base):
    """The 50 simulated orders (data/dataset.py, same seed), read only by check_order_status."""

    __tablename__ = "orders"
    __table_args__ = (
        CheckConstraint(_in("category", REQUIRED_CATEGORIES), name="category"),
        CheckConstraint(_in("status", REQUIRED_STATUSES), name="status"),
        CheckConstraint("order_value_inr >= 0", name="order_value_inr"),
        CheckConstraint(f"days_since_created BETWEEN 0 AND {MAX_DAYS}", name="days_since_created"),
    )

    record_id: Mapped[str] = mapped_column(String(9), primary_key=True)
    category: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(10))
    order_value_inr: Mapped[int] = mapped_column(Integer)
    days_since_created: Mapped[int] = mapped_column(Integer)
    delayed_shipment: Mapped[bool] = mapped_column(Boolean)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # app.api.schemas.SESSION_ID_PATTERN: 1-64 chars
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ChatMessage(Base):
    """One turn of a session's history, in insertion order (id). Content is the masked question / checked reply."""

    __tablename__ = "chat_messages"
    __table_args__ = (CheckConstraint("role IN ('human', 'ai')", name="role"),)

    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(5))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RequestLog(Base):
    """The JsonlLogger record (already masked and sanitised), so logs survive a host without a persistent disk."""

    __tablename__ = "request_log"

    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    type: Mapped[str] = mapped_column(String(32))
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True)
    record: Mapped[dict[str, Any]] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
