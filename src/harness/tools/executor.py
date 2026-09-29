"""Runs one validated tool call with timeout, retries and output validation (PLAN §8.5).

This is the only place tool calls are retried (P5). Every outcome, including
failure, comes back as a ``ToolResult`` that the loop turns into an observation
for the model (P7); tool errors never escape as exceptions.
"""

from __future__ import annotations

import asyncio
import random

from pydantic import BaseModel, ValidationError

from harness.clock import Clock
from harness.domain.errors import (
    ToolError,
    ToolOutputValidationError,
    ToolTimeoutError,
    describe_validation_error,
)
from harness.domain.models import EventType, ToolResult
from harness.observability.tracer import EventSink
from harness.tools.faults import FaultInjector
from harness.tools.spec import ToolContext, ToolSpec


class ToolExecutor:
    """Executes tools. Stateless across calls apart from the fault counters."""

    def __init__(
        self,
        clock: Clock,
        events: EventSink,
        faults: FaultInjector | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._clock = clock
        self._events = events
        self._faults = faults or FaultInjector()
        self._rng = rng or random.Random()

    async def run(self, spec: ToolSpec, args: BaseModel, ctx: ToolContext) -> ToolResult:
        handler = self._faults.wrap(spec)
        started = self._clock.monotonic()
        attempt = 0
        while True:
            attempt += 1
            try:
                timeout_s = min(spec.timeout_s, ctx.deadline_remaining_s)
                try:
                    async with asyncio.timeout(timeout_s):
                        raw = await handler(args, ctx)
                except TimeoutError:
                    raise ToolTimeoutError(
                        spec.name, f"no response within {timeout_s:g}s"
                    ) from None
                data = self._validate_output(spec, raw)
            except ToolError as error:
                # P5: retry only transient failures, and only up to the policy's limit.
                if error.retryable and attempt <= spec.retry.max_retries:
                    delay_s = spec.retry.backoff_delay(attempt, self._rng)
                    await self._events.emit(
                        ctx.run_id,
                        EventType.TOOL_ATTEMPT_FAILED,
                        ctx.step,
                        tool=spec.name,
                        attempt=attempt,
                        error_type=error.error_type,
                        message=error.message,
                        will_retry=True,
                        delay_s=round(delay_s, 3),
                    )
                    await self._clock.sleep(delay_s)
                    continue
                result = ToolResult.failed(
                    error, attempts=attempt, duration_ms=self._elapsed_ms(started)
                )
                await self._events.emit(
                    ctx.run_id,
                    EventType.TOOL_FAILED,
                    ctx.step,
                    tool=spec.name,
                    attempts=attempt,
                    error_type=error.error_type,
                    message=error.message,
                    retryable=error.retryable,
                )
                return result

            result = ToolResult.succeeded(
                spec.name, data, attempts=attempt, duration_ms=self._elapsed_ms(started)
            )
            await self._events.emit(
                ctx.run_id,
                EventType.TOOL_SUCCEEDED,
                ctx.step,
                tool=spec.name,
                attempts=attempt,
                duration_ms=result.duration_ms,
            )
            return result

    @staticmethod
    def _validate_output(spec: ToolSpec, raw: object) -> dict[str, object]:
        """Hold tools to their declared output schema; a violation is not retried."""
        try:
            return spec.output_model.model_validate(raw).model_dump(mode="json")
        except ValidationError as exc:
            raise ToolOutputValidationError(
                spec.name, f"tool returned invalid output: {describe_validation_error(exc)}"
            ) from exc

    def _elapsed_ms(self, started: float) -> int:
        return round((self._clock.monotonic() - started) * 1000)
