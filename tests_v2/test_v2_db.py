"""v2 Stage C: persistence on a real SQL database. These run on SQLite (no server needed); the same code paths run
against PostgreSQL in test_v2_postgres.py when a test database is configured.
"""

from __future__ import annotations

import json
import uuid

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from app.core.config import ROOT
from app.db import database_url, get_engine, session
from app.db.models import Base, ChatMessage, Order, RequestLog
from app.tools.order_lookup import InvalidRecordIdError, OrderNotFoundError, check_order_status
from data.dataset import ORDERS
from scripts.seed_db import seed

TABLES = {"orders", "chat_sessions", "chat_messages", "request_log"}


def alembic_config(url: str) -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.attributes.update(url=url, skip_logging_config=True)
    return cfg


@pytest.fixture
def db(tmp_path, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'v2.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    command.upgrade(alembic_config(url), "head")
    return url


def test_the_migration_creates_exactly_the_models_schema(db):
    with get_engine(db).connect() as conn:
        assert TABLES <= set(inspect(conn).get_table_names())
        assert compare_metadata(MigrationContext.configure(conn), Base.metadata) == []


def test_downgrade_removes_every_table(db):
    command.downgrade(alembic_config(db), "base")
    with get_engine(db).connect() as conn:
        assert not (TABLES & set(inspect(conn).get_table_names()))


def test_seeding_is_idempotent_and_equals_the_dataset(db):
    assert seed() == seed() == 50
    with session() as s:
        rows = {o.record_id: o for o in s.query(Order).all()}
    assert {k: {c: getattr(v, c) for c in ORDERS[0]} for k, v in rows.items()} == {o["record_id"]: o for o in ORDERS}


def test_order_lookup_from_the_database_equals_the_in_memory_lookup(db, monkeypatch):
    seed()
    from_db = {o["record_id"]: check_order_status(o["record_id"]) for o in ORDERS}
    monkeypatch.delenv("DATABASE_URL")
    assert from_db == {o["record_id"]: check_order_status(o["record_id"]) for o in ORDERS}


def test_lookup_errors_are_unchanged_with_a_database(db):
    seed()
    with pytest.raises(OrderNotFoundError, match="ORD-9999"):
        check_order_status("ORD-9999")
    with pytest.raises(InvalidRecordIdError):
        check_order_status("'; DROP TABLE orders; --")


def test_the_schema_rejects_invalid_orders(db):
    bad = {**ORDERS[0], "record_id": "ORD-0001", "status": "Lost"}
    with pytest.raises(IntegrityError), session() as s, s.begin():
        s.add(Order(**bad))


def test_chat_history_survives_a_restart_and_holds_masked_text_only(db, collections):
    from app.memory import session_memory
    from app.pipeline import SupportAssistant
    from app.rag.generation import RECOMMENDED_STRATEGY
    from app.tools.crew_tools import RagLookupTool

    def make() -> SupportAssistant:
        return SupportAssistant(rag_tool_factory=lambda: RagLookupTool(collection=collections[RECOMMENDED_STRATEGY]))

    sid = f"db-{uuid.uuid4().hex[:8]}"
    first = make().ask(sid, "My number is 9876543210. How long do I have to return shoes?")
    session_memory._store.clear()  # a restart: nothing left in the process
    assert session_memory.has_session(sid)
    history = session_memory.get_session_history(sid).messages
    assert [m.type for m in history] == ["human", "ai"] and history[1].content == first.reply
    with session() as s:
        stored = " ".join(m.content for m in s.query(ChatMessage).filter_by(session_id=sid))
    assert "9876543210" not in stored and "[PHONE]" in stored
    make().ask(sid, "And for apparel?")  # the second turn sees the first one from the database
    assert len(session_memory.get_session_history(sid).messages) == 4
    session_memory.get_session_history(sid).clear()  # deleting a session cascades to its messages
    with session() as s:
        assert s.query(ChatMessage).filter_by(session_id=sid).count() == 0


def test_request_log_rows_are_the_masked_jsonl_records(db, tmp_path):
    from app.observability.logging import JsonlLogger

    logger = JsonlLogger(tmp_path / "log.jsonl")
    logger.log(type="http", trace_id="t1", masked_question="call me on 9876543210")
    with session() as s:
        row = s.query(RequestLog).one()
    assert row.type == "http" and row.trace_id == "t1" and "9876543210" not in json.dumps(row.record)
    assert json.loads((tmp_path / "log.jsonl").read_text(encoding="utf-8")) == row.record


def test_a_database_failure_never_breaks_logging(tmp_path, monkeypatch):
    from app.observability.logging import JsonlLogger

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'no-tables.db').as_posix()}")  # never migrated
    logger = JsonlLogger(tmp_path / "log.jsonl")
    logger.log(type="http", trace_id="t2")
    lines = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[0]["trace_id"] == "t2" and lines[1] == {**lines[1], "type": "log_sink_error"}
    assert lines[1]["error_type"] == "OperationalError" and "no such table" not in json.dumps(lines)


def test_without_database_url_everything_stays_in_memory():
    from app.memory import session_memory

    assert database_url() is None
    sid = f"mem-{uuid.uuid4().hex[:8]}"
    session_memory.get_session_history(sid)
    assert session_memory.has_session(sid) and sid in session_memory._store


@pytest.mark.parametrize("given, expected", [
    ("postgres://u:p@h:5432/d", "postgresql+psycopg://u:p@h:5432/d"),
    ("postgresql://u:p@h/d", "postgresql+psycopg://u:p@h/d"),
    ("postgresql+psycopg://u:p@h/d", "postgresql+psycopg://u:p@h/d"),
    ("sqlite:///x.db", "sqlite:///x.db"),
])
def test_database_urls_use_the_psycopg3_driver(monkeypatch, given, expected):
    monkeypatch.setenv("DATABASE_URL", given)
    assert database_url() == expected


def test_sqlite_test_databases_enforce_foreign_keys(db):
    with get_engine(db).connect() as conn:
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1
