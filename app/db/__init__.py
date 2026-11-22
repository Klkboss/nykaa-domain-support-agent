"""v2 persistence: PostgreSQL when DATABASE_URL is set, otherwise the capstone's in-memory behaviour (unchanged).

`database_url()` reads DATABASE_URL on demand (never at import). postgres:// and postgresql:// URLs are mapped to
the psycopg 3 driver. Engines are cached per URL.
"""

from __future__ import annotations

import os

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

_engines: dict[str, Engine] = {}


def database_url() -> str | None:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        return None
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def get_engine(url: str | None = None) -> Engine:
    url = url or database_url()
    if url is None:
        raise RuntimeError("DATABASE_URL is not set")
    if url not in _engines:
        engine = create_engine(url, pool_pre_ping=True)  # pre_ping: survives a database that scaled to zero
        if engine.dialect.name == "sqlite":  # unit tests: SQLite enforces foreign keys only when asked
            event.listen(engine, "connect", lambda conn, _: conn.execute("PRAGMA foreign_keys=ON"))
        _engines[url] = engine
    return _engines[url]


def session(url: str | None = None) -> Session:
    return sessionmaker(get_engine(url), expire_on_commit=False)()
