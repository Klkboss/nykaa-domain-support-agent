"""Initial schema: orders, chat_sessions, chat_messages, request_log.

Revision ID: 0001
Revises:
Create Date: 2026-11-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

BIG_ID = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
CATEGORIES = "category IN ('Apparel', 'Electronics', 'Home', 'Footwear', 'Beauty')"
STATUSES = "status IN ('Placed', 'Shipped', 'Delivered', 'Returned', 'Refunded')"


def upgrade() -> None:
    op.create_table(
        "orders",
        sa.Column("record_id", sa.String(9), nullable=False),
        sa.Column("category", sa.String(20), nullable=False),
        sa.Column("status", sa.String(10), nullable=False),
        sa.Column("order_value_inr", sa.Integer(), nullable=False),
        sa.Column("days_since_created", sa.Integer(), nullable=False),
        sa.Column("delayed_shipment", sa.Boolean(), nullable=False),
        sa.CheckConstraint(CATEGORIES, name=op.f("ck_orders_category")),
        sa.CheckConstraint(STATUSES, name=op.f("ck_orders_status")),
        sa.CheckConstraint("order_value_inr >= 0", name=op.f("ck_orders_order_value_inr")),
        sa.CheckConstraint("days_since_created BETWEEN 0 AND 30", name=op.f("ck_orders_days_since_created")),
        sa.PrimaryKeyConstraint("record_id", name=op.f("pk_orders")),
    )
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chat_sessions")),
    )
    op.create_table(
        "chat_messages",
        sa.Column("id", BIG_ID, autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(64), nullable=False),
        sa.Column("role", sa.String(5), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("role IN ('human', 'ai')", name=op.f("ck_chat_messages_role")),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"], ondelete="CASCADE",
                                name=op.f("fk_chat_messages_session_id_chat_sessions")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chat_messages")),
    )
    op.create_index(op.f("ix_chat_messages_session_id"), "chat_messages", ["session_id"])
    op.create_table(
        "request_log",
        sa.Column("id", BIG_ID, autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("trace_id", sa.String(64), nullable=True),
        sa.Column("record", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_request_log")),
    )
    op.create_index(op.f("ix_request_log_trace_id"), "request_log", ["trace_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_request_log_trace_id"), table_name="request_log")
    op.drop_table("request_log")
    op.drop_index(op.f("ix_chat_messages_session_id"), table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
    op.drop_table("orders")
