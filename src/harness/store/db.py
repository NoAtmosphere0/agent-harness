"""Engine and session setup.

SQLite by default; the schema only uses portable types so the same code runs on
Postgres by changing ``DATABASE_URL``. Tables are created at startup; there are no
migrations (PLAN §10).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from harness.store.tables import Base

SQLITE_BUSY_TIMEOUT_MS = 5000


def create_engine(database_url: str) -> AsyncEngine:
    url = make_url(database_url)
    is_sqlite = url.get_backend_name() == "sqlite"
    in_memory = is_sqlite and url.database in (None, "", ":memory:")
    kwargs: dict[str, Any] = {}
    if in_memory:
        # An in-memory database lives inside one connection; share it across sessions.
        kwargs = {"poolclass": StaticPool, "connect_args": {"check_same_thread": False}}
    engine = create_async_engine(url, **kwargs)
    if is_sqlite:
        event.listen(engine.sync_engine, "connect", _enable_sqlite_foreign_keys)
    if is_sqlite and not in_memory:
        event.listen(engine.sync_engine, "connect", _enable_sqlite_wal)
    return engine


def _enable_sqlite_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
    # SQLite ignores foreign keys unless asked, per connection.
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def _enable_sqlite_wal(dbapi_connection: Any, _record: Any) -> None:
    # WAL lets API reads proceed while a run is writing; busy_timeout makes a second
    # writer wait for the lock instead of failing at once with "database is locked".
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    cursor.close()


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    # Records are converted to pydantic models before the session closes, so
    # expiring attributes on commit would only cost extra queries.
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
