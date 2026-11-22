"""v2 Stage C: integration tests against a REAL PostgreSQL server. Skipped unless TEST_DATABASE_URL is set and
reachable. They DROP and recreate the schema, so they use TEST_DATABASE_URL, never DATABASE_URL.

Local run (Docker):
  docker run -d --name nykaa-pg-test -e POSTGRES_USER=test -e POSTGRES_PASSWORD=test -e POSTGRES_DB=test \
      -p 55432:5432 postgres:17-alpine
  $env:TEST_DATABASE_URL = "postgresql://test:test@localhost:55432/test"
  .venv\\Scripts\\python.exe -m pytest -q tests_v2
"""

from __future__ import annotations

import json
import os
import uuid

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text

from tests_v2.test_v2_db import TABLES, alembic_config

RAW_URL = os.environ.get("TEST_DATABASE_URL", "")


def _reachable() -> bool:
    if not RAW_URL:
        return False
    from app.db import get_engine

    url = RAW_URL.replace("postgresql://", "postgresql+psycopg://", 1)
    try:
        with get_engine(url).connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason="set TEST_DATABASE_URL to a reachable PostgreSQL database")


@pytest.fixture
def pg(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", RAW_URL)
    from app.db import database_url

    url = database_url()
    command.downgrade(alembic_config(url), "base")
    command.upgrade(alembic_config(url), "head")
    return url


def test_alembic_upgrade_head_creates_the_schema_on_postgres(pg):
    from app.db import get_engine
    from app.db.models import Base

    with get_engine(pg).connect() as conn:
        assert TABLES <= set(inspect(conn).get_table_names())
        assert compare_metadata(MigrationContext.configure(conn), Base.metadata) == []
        assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == "0001"
        record_type = conn.execute(text("SELECT data_type FROM information_schema.columns "
                                        "WHERE table_name='request_log' AND column_name='record'")).scalar()
    assert record_type == "jsonb"


def test_seed_and_lookup_on_postgres(pg):
    from app.tools.order_lookup import check_order_status
    from data.dataset import ORDERS
    from scripts.seed_db import seed

    assert seed() == seed() == 50
    assert check_order_status("ORD-1016")["status"] == next(o for o in ORDERS if o["record_id"] == "ORD-1016")["status"]


def test_check_constraints_are_enforced_on_postgres(pg):
    from sqlalchemy.exc import IntegrityError

    from app.db import session
    from app.db.models import Order
    from data.dataset import ORDERS

    with pytest.raises(IntegrityError), session() as s, s.begin():
        s.add(Order(**{**ORDERS[0], "record_id": "ORD-0002", "days_since_created": 31}))


def test_pipeline_history_and_request_log_on_postgres(pg, collections, tmp_path):
    from app.db import session
    from app.db.models import ChatMessage, RequestLog
    from app.memory import session_memory
    from app.observability.logging import JsonlLogger
    from app.pipeline import SupportAssistant
    from app.rag.generation import RECOMMENDED_STRATEGY
    from app.tools.crew_tools import RagLookupTool

    sid = f"pg-{uuid.uuid4().hex[:8]}"
    assistant = SupportAssistant(rag_tool_factory=lambda: RagLookupTool(collection=collections[RECOMMENDED_STRATEGY]))
    assistant.ask(sid, "My card 4111 1111 1111 1111 was charged. How long do I have to return shoes?")
    session_memory._store.clear()
    assert session_memory.has_session(sid) and len(session_memory.get_session_history(sid).messages) == 2
    with session() as s:
        stored = " ".join(m.content for m in s.query(ChatMessage).filter_by(session_id=sid))
    assert "4111" not in stored
    JsonlLogger(tmp_path / "log.jsonl").log(type="http", trace_id=sid, masked_question=stored)
    with session() as s:
        row = s.query(RequestLog).filter_by(trace_id=sid).one()
    assert "4111" not in json.dumps(row.record)
