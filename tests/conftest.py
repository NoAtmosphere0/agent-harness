from pathlib import Path

import pytest

from harness.clock import FakeClock
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
