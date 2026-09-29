from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from harness.clock import FakeClock
from harness.observability.tracer import Tracer
from harness.store.db import create_engine, create_session_factory, init_db
from harness.store.repository import Repository, SqlIncidentStore
from tests.fakes import InMemoryIncidentStore, RecordingEvents

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def data_dir() -> Path:
    return REPO_ROOT / "data"


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def events() -> RecordingEvents:
    return RecordingEvents()


@pytest.fixture
def incident_store(clock: FakeClock) -> InMemoryIncidentStore:
    return InMemoryIncidentStore(clock)


# ------------------------------------------------------------------ database


@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await init_db(engine)
    yield engine
    await engine.dispose()


@pytest.fixture
def sessions(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return create_session_factory(engine)


@pytest.fixture
async def file_sessions(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A file-backed database for concurrency tests.

    In-memory SQLite shares one connection (and so one transaction) across all
    sessions, which cannot model two callers racing. A file gives each session its
    own connection, as in production.
    """
    engine = create_engine(f"sqlite+aiosqlite:///{tmp_path / 'harness.db'}")
    await init_db(engine)
    yield create_session_factory(engine)
    await engine.dispose()


@pytest.fixture
def repo(sessions: async_sessionmaker[AsyncSession], clock: FakeClock) -> Repository:
    return Repository(sessions, clock)


@pytest.fixture
def sql_incident_store(
    sessions: async_sessionmaker[AsyncSession], clock: FakeClock
) -> SqlIncidentStore:
    return SqlIncidentStore(sessions, clock)


@pytest.fixture
def tracer(repo: Repository) -> Tracer:
    return Tracer(repo)
