from datetime import UTC, datetime, timedelta

import pytest

from harness.clock import FakeClock, SystemClock


async def test_fake_clock_sleep_records_and_advances_without_waiting():
    clock = FakeClock()
    start_mono, start_utc = clock.monotonic(), clock.now_utc()

    await clock.sleep(1.5)
    await clock.sleep(0.25)

    assert clock.sleeps == [1.5, 0.25]
    assert clock.monotonic() == start_mono + 1.75
    assert clock.now_utc() == start_utc + timedelta(seconds=1.75)


def test_fake_clock_advance_moves_time_without_recording_sleep():
    clock = FakeClock(start_monotonic=100.0)

    clock.advance(30)

    assert clock.monotonic() == 130.0
    assert clock.sleeps == []


def test_fake_clock_advance_rejects_negative():
    clock = FakeClock()

    with pytest.raises(ValueError, match="backwards"):
        clock.advance(-1)


def test_fake_clock_rejects_naive_start():
    with pytest.raises(ValueError, match="timezone-aware"):
        FakeClock(start_utc=datetime(2026, 1, 1))  # noqa: DTZ001


async def test_system_clock_is_monotonic_and_utc():
    clock = SystemClock()

    first = clock.monotonic()
    await clock.sleep(0)
    second = clock.monotonic()

    assert second >= first
    assert clock.now_utc().tzinfo is UTC
