import asyncio
import random
from typing import Any

import pytest
from pydantic import BaseModel

from harness.clock import FakeClock
from harness.domain.errors import ToolPermanentError, ToolTransientError
from harness.domain.models import EventType
from harness.tools.executor import ToolExecutor
from harness.tools.spec import RetryPolicy, ToolContext, ToolHandler, ToolSpec
from tests.fakes import RecordingEvents


class _In(BaseModel):
    x: int = 0


class _Out(BaseModel):
    value: int


def _spec(handler: ToolHandler, *, timeout_s: float = 1.0, max_retries: int = 2) -> ToolSpec:
    return ToolSpec(
        name="stub",
        description="stub tool",
        input_model=_In,
        output_model=_Out,
        handler=handler,
        timeout_s=timeout_s,
        retry=RetryPolicy(max_retries=max_retries, base_delay_s=0.5, max_delay_s=4.0),
    )


def _ctx(deadline_remaining_s: float = 60.0) -> ToolContext:
    return ToolContext(run_id="run-1", step=1, deadline_remaining_s=deadline_remaining_s)


class _Script:
    """Handler that plays back a list of outcomes (dict to return, or exception to raise)."""

    def __init__(self, *outcomes: dict[str, Any] | Exception) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    async def __call__(self, args: _In, ctx: ToolContext) -> dict[str, Any]:
        outcome = self._outcomes[min(self.calls, len(self._outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def executor(clock: FakeClock, events: RecordingEvents) -> ToolExecutor:
    return ToolExecutor(clock, events, rng=random.Random(7))


async def test_executor_first_try_success(
    executor: ToolExecutor, clock: FakeClock, events: RecordingEvents
):
    handler = _Script({"value": 1})

    result = await executor.run(_spec(handler), _In(), _ctx())

    assert result.ok is True
    assert result.data == {"value": 1}
    assert result.attempts == 1
    assert clock.sleeps == []
    assert events.types() == [EventType.TOOL_SUCCEEDED]


async def test_executor_transient_twice_then_success_records_two_backoff_sleeps(
    executor: ToolExecutor, clock: FakeClock, events: RecordingEvents
):
    handler = _Script(
        ToolTransientError("stub", "503"), ToolTransientError("stub", "503"), {"value": 2}
    )

    result = await executor.run(_spec(handler), _In(), _ctx())

    assert result.ok is True
    assert result.attempts == 3
    assert len(clock.sleeps) == 2
    assert 0 <= clock.sleeps[0] <= 0.5
    assert 0 <= clock.sleeps[1] <= 1.0
    assert result.duration_ms == round(sum(clock.sleeps) * 1000)
    assert events.types() == [
        EventType.TOOL_ATTEMPT_FAILED,
        EventType.TOOL_ATTEMPT_FAILED,
        EventType.TOOL_SUCCEEDED,
    ]
    assert events.events[0].payload["will_retry"] is True


async def test_executor_gives_up_after_max_retries(
    executor: ToolExecutor, clock: FakeClock, events: RecordingEvents
):
    handler = _Script(ToolTransientError("stub", "503"))

    result = await executor.run(_spec(handler, max_retries=2), _In(), _ctx())

    assert result.ok is False
    assert result.attempts == 3
    assert handler.calls == 3
    assert result.error is not None
    assert result.error.type == "ToolTransientError"
    assert len(clock.sleeps) == 2
    assert events.types()[-1] == EventType.TOOL_FAILED


async def test_executor_timeout_is_retried_then_reported(executor: ToolExecutor):
    async def hangs(args: _In, ctx: ToolContext) -> dict[str, Any]:
        await asyncio.sleep(1)
        return {"value": 0}

    result = await executor.run(_spec(hangs, timeout_s=0.02, max_retries=1), _In(), _ctx())

    assert result.ok is False
    assert result.attempts == 2
    assert result.error is not None
    assert result.error.type == "ToolTimeoutError"
    assert result.error.retryable is True


async def test_executor_timeout_capped_by_run_deadline(executor: ToolExecutor):
    async def hangs(args: _In, ctx: ToolContext) -> dict[str, Any]:
        await asyncio.sleep(1)
        return {"value": 0}

    result = await executor.run(
        _spec(hangs, timeout_s=5.0, max_retries=0), _In(), _ctx(deadline_remaining_s=0.02)
    )

    assert result.error is not None
    assert result.error.type == "ToolTimeoutError"
    assert "0.02s" in result.error.message


async def test_executor_permanent_error_not_retried(executor: ToolExecutor, clock: FakeClock):
    handler = _Script(ToolPermanentError("stub", "unknown service"))

    result = await executor.run(_spec(handler), _In(), _ctx())

    assert result.ok is False
    assert result.attempts == 1
    assert handler.calls == 1
    assert clock.sleeps == []
    assert result.error is not None
    assert result.error.retryable is False


async def test_executor_output_validation_failure_not_retried(
    executor: ToolExecutor, clock: FakeClock
):
    handler = _Script({"unexpected": "shape"})

    result = await executor.run(_spec(handler), _In(), _ctx())

    assert result.ok is False
    assert result.attempts == 1
    assert handler.calls == 1
    assert clock.sleeps == []
    assert result.error is not None
    assert result.error.type == "ToolOutputValidationError"
    assert "value: Field required" in result.error.message


async def test_executor_lets_unexpected_exceptions_propagate(executor: ToolExecutor):
    handler = _Script(RuntimeError("bug in handler"))

    with pytest.raises(RuntimeError, match="bug in handler"):
        await executor.run(_spec(handler), _In(), _ctx())
