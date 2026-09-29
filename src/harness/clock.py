"""Injectable time source.

Core code never reads the system clock or sleeps directly. Routing time through
``Clock`` makes backoff delays (P5) and active-time accounting (P8) deterministic
in tests: ``FakeClock.sleep`` returns immediately but moves time forward.

Awaited-call timeouts are the exception: they use ``asyncio.timeout()`` with real
time, so timeout tests use very small values (PLAN §6.6).
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import Protocol

_DEFAULT_FAKE_START = datetime(2026, 1, 1, tzinfo=UTC)


class Clock(Protocol):
    """The only way core code observes or spends time."""

    def monotonic(self) -> float:
        """Seconds from an arbitrary origin; only differences are meaningful."""
        ...

    def now_utc(self) -> datetime:
        """Timezone-aware wall-clock time, used for persisted timestamps."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Suspend the caller for ``seconds`` (e.g. a retry backoff)."""
        ...


class SystemClock:
    """Real time, used at runtime."""

    def monotonic(self) -> float:
        return time.monotonic()

    def now_utc(self) -> datetime:
        return datetime.now(UTC)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class FakeClock:
    """Deterministic clock for tests.

    ``sleep`` does not wait: it records the requested duration in ``sleeps`` and
    advances time, so a test can assert on backoff delays without slowing down.
    ``advance`` moves time explicitly, e.g. to push a run past its time limit.
    """

    def __init__(
        self,
        start_utc: datetime = _DEFAULT_FAKE_START,
        start_monotonic: float = 0.0,
    ) -> None:
        if start_utc.tzinfo is None:
            raise ValueError("start_utc must be timezone-aware")
        self._start_utc = start_utc
        self._start_monotonic = start_monotonic
        self._elapsed = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self._start_monotonic + self._elapsed

    def now_utc(self) -> datetime:
        return self._start_utc + timedelta(seconds=self._elapsed)

    def advance(self, seconds: float) -> None:
        """Move time forward by ``seconds``; time never goes backwards."""
        if seconds < 0:
            raise ValueError("cannot move time backwards")
        self._elapsed += seconds

    async def sleep(self, seconds: float) -> None:
        self.advance(seconds)
        self.sleeps.append(seconds)
        # Still yield to the event loop so a sleep remains a suspension point,
        # as it is with the real clock.
        await asyncio.sleep(0)
