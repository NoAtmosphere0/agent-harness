from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from harness.store.db import SQLITE_BUSY_TIMEOUT_MS


async def _pragma(sessions: async_sessionmaker[AsyncSession], name: str) -> object:
    async with sessions() as session:
        return (await session.execute(text(f"PRAGMA {name}"))).scalar()


async def test_db_file_sqlite_uses_wal_and_busy_timeout(
    file_sessions: async_sessionmaker[AsyncSession],
):
    assert await _pragma(file_sessions, "journal_mode") == "wal"
    assert await _pragma(file_sessions, "busy_timeout") == SQLITE_BUSY_TIMEOUT_MS
    assert await _pragma(file_sessions, "foreign_keys") == 1


async def test_db_memory_sqlite_keeps_default_journal(sessions: async_sessionmaker[AsyncSession]):
    assert await _pragma(sessions, "journal_mode") == "memory"
    assert await _pragma(sessions, "foreign_keys") == 1
