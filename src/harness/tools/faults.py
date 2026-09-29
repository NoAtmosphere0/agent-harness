"""Fault injection for the mock tools (PLAN §7.4).

Faults are configured per run (from ``MOCK_FAULTS`` or ``config_overrides.faults``)
and counted per ``(run_id, tool_name)``, so one run's flaky tool does not change
another run's behaviour.

Sleeps here use real time (``asyncio.sleep``), not the injected ``Clock``: the
point of ``slow`` and ``timeout_after_commit`` is to trip the executor's real
``asyncio.timeout()``.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from harness.domain.errors import ToolPermanentError, ToolTransientError
from harness.tools.spec import ToolContext, ToolHandler, ToolSpec


class _FaultBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FlakyFault(_FaultBase):
    """Fail with a transient error on the first ``fail_times`` calls, then behave."""

    mode: Literal["flaky"]
    fail_times: int = Field(default=1, ge=1)


class SlowFault(_FaultBase):
    """Sleep ``delay_s`` before answering; exceeds the timeout when larger."""

    mode: Literal["slow"]
    delay_s: float = Field(gt=0)


class ErrorFault(_FaultBase):
    """Always fail with a permanent error."""

    mode: Literal["error"]
    message: str = "upstream returned HTTP 500 (injected fault)"


class TimeoutAfterCommitFault(_FaultBase):
    """Commit the side effect, then hang past the timeout (``create_incident`` only)."""

    mode: Literal["timeout_after_commit"]
    times: int = Field(default=1, ge=1)


Fault = Annotated[
    FlakyFault | SlowFault | ErrorFault | TimeoutAfterCommitFault,
    Field(discriminator="mode"),
]
_FAULTS_ADAPTER: TypeAdapter[dict[str, Fault]] = TypeAdapter(dict[str, Fault])

_SIDE_EFFECTING_TOOLS = frozenset({"create_incident"})


def parse_faults(raw: Mapping[str, Any]) -> dict[str, Fault]:
    """Validate a ``{tool_name: fault}`` mapping. Raises ``ValueError`` on bad input."""
    faults = _FAULTS_ADAPTER.validate_python(dict(raw))
    for tool_name, fault in faults.items():
        if isinstance(fault, TimeoutAfterCommitFault) and tool_name not in _SIDE_EFFECTING_TOOLS:
            raise ValueError(
                f"timeout_after_commit only applies to create_incident, not {tool_name}"
            )
    return faults


class FaultInjector:
    """Wraps tool handlers with the faults configured for the calling run."""

    def __init__(self) -> None:
        self._run_faults: dict[str, dict[str, Fault]] = {}
        self._calls: Counter[tuple[str, str]] = Counter()

    def set_run_faults(self, run_id: str, faults: Mapping[str, Fault]) -> None:
        self._run_faults[run_id] = dict(faults)

    def clear_run(self, run_id: str) -> None:
        """Forget a finished run's faults and call counters so they don't accumulate."""
        self._run_faults.pop(run_id, None)
        for key in [k for k in self._calls if k[0] == run_id]:
            del self._calls[key]

    def wrap(self, spec: ToolSpec) -> ToolHandler:
        """Return ``spec.handler``, with the run's fault (if any) applied around it."""

        async def handler(args: Any, ctx: ToolContext) -> dict[str, Any]:
            fault = self._run_faults.get(ctx.run_id, {}).get(spec.name)
            if fault is None:
                return await spec.handler(args, ctx)

            key = (ctx.run_id, spec.name)
            self._calls[key] += 1
            call_number = self._calls[key]

            match fault:
                case FlakyFault(fail_times=n) if call_number <= n:
                    raise ToolTransientError(
                        spec.name, f"upstream temporarily unavailable (injected, {call_number}/{n})"
                    )
                case SlowFault(delay_s=delay):
                    await asyncio.sleep(delay)
                case ErrorFault(message=message):
                    raise ToolPermanentError(spec.name, message)
                case TimeoutAfterCommitFault(times=n) if call_number <= n:
                    result = await spec.handler(args, ctx)  # the side effect is committed...
                    await asyncio.sleep(spec.timeout_s + 1)  # ...but the caller never hears back
                    return result
            return await spec.handler(args, ctx)

        return handler
