import random
from typing import Any

import pytest
from pydantic import BaseModel

from harness.clock import FakeClock
from harness.domain.errors import ToolPermanentError, ToolTransientError
from harness.tools.executor import ToolExecutor
from harness.tools.faults import (
    ErrorFault,
    FaultInjector,
    FlakyFault,
    SlowFault,
    TimeoutAfterCommitFault,
    parse_faults,
)
from harness.tools.incidents import CreateIncidentInput, make_create_incident_tool
from harness.tools.spec import RetryPolicy, ToolContext, ToolSpec
from tests.fakes import InMemoryIncidentStore, RecordingEvents


class _In(BaseModel):
    pass


class _Out(BaseModel):
    ok: bool


async def _ok_handler(args: _In, ctx: ToolContext) -> dict[str, Any]:
    return {"ok": True}


def _spec(timeout_s: float = 1.0) -> ToolSpec:
    return ToolSpec(
        name="stub",
        description="stub",
        input_model=_In,
        output_model=_Out,
        handler=_ok_handler,
        timeout_s=timeout_s,
    )


def _ctx(run_id: str = "run-1", **kwargs: Any) -> ToolContext:
    return ToolContext(
        run_id=run_id, step=1, deadline_remaining_s=kwargs.pop("deadline", 60.0), **kwargs
    )


def test_parse_faults_builds_typed_faults():
    faults = parse_faults(
        {
            "get_service_status": {"mode": "flaky", "fail_times": 2},
            "search_knowledge_base": {"mode": "slow", "delay_s": 0.1},
            "create_incident": {"mode": "timeout_after_commit"},
        }
    )

    assert faults["get_service_status"] == FlakyFault(mode="flaky", fail_times=2)
    assert faults["search_knowledge_base"] == SlowFault(mode="slow", delay_s=0.1)
    assert faults["create_incident"] == TimeoutAfterCommitFault(mode="timeout_after_commit")


@pytest.mark.parametrize(
    "raw",
    [
        {"get_service_status": {"mode": "explode"}},
        {"get_service_status": {"mode": "flaky", "fail_times": 0}},
        {"get_service_status": {"mode": "slow"}},
        {"get_service_status": {"mode": "timeout_after_commit"}},
    ],
)
def test_parse_faults_rejects_invalid_config(raw: dict[str, Any]):
    with pytest.raises(ValueError):  # noqa: PT011 - pydantic's ValidationError is a ValueError
        parse_faults(raw)


async def test_fault_injector_passes_through_when_no_fault_configured():
    handler = FaultInjector().wrap(_spec())

    assert await handler(_In(), _ctx()) == {"ok": True}


async def test_flaky_fault_fails_first_calls_then_recovers_per_run():
    injector = FaultInjector()
    injector.set_run_faults("run-1", {"stub": FlakyFault(mode="flaky", fail_times=2)})
    injector.set_run_faults("run-2", {"stub": FlakyFault(mode="flaky", fail_times=2)})
    handler = injector.wrap(_spec())

    for _ in range(2):
        with pytest.raises(ToolTransientError):
            await handler(_In(), _ctx("run-1"))
    assert await handler(_In(), _ctx("run-1")) == {"ok": True}
    # Counters are per (run_id, tool): run-2 still starts from its first call.
    with pytest.raises(ToolTransientError):
        await handler(_In(), _ctx("run-2"))


async def test_error_fault_always_raises_permanent_error():
    injector = FaultInjector()
    injector.set_run_faults("run-1", {"stub": ErrorFault(mode="error")})
    handler = injector.wrap(_spec())

    for _ in range(3):
        with pytest.raises(ToolPermanentError, match="injected"):
            await handler(_In(), _ctx())


async def test_slow_fault_past_timeout_surfaces_as_tool_timeout(
    clock: FakeClock, events: RecordingEvents
):
    injector = FaultInjector()
    injector.set_run_faults("run-1", {"stub": SlowFault(mode="slow", delay_s=0.5)})
    spec = _spec(timeout_s=0.02).model_copy(update={"retry": RetryPolicy(max_retries=0)})

    result = await ToolExecutor(clock, events, injector).run(spec, _In(), _ctx())

    assert result.error is not None
    assert result.error.type == "ToolTimeoutError"


async def test_slow_fault_within_timeout_still_succeeds(clock: FakeClock, events: RecordingEvents):
    injector = FaultInjector()
    injector.set_run_faults("run-1", {"stub": SlowFault(mode="slow", delay_s=0.01)})

    result = await ToolExecutor(clock, events, injector).run(_spec(), _In(), _ctx())

    assert result.ok is True


async def test_timeout_after_commit_retry_returns_original_incident(
    clock: FakeClock, events: RecordingEvents, incident_store: InMemoryIncidentStore
):
    injector = FaultInjector()
    injector.set_run_faults(
        "run-1", {"create_incident": TimeoutAfterCommitFault(mode="timeout_after_commit")}
    )
    spec = make_create_incident_tool(incident_store, RetryPolicy(max_retries=2))
    args = CreateIncidentInput(
        title="payment-gateway: full outage",
        description="Payment gateway error rate 96.8%, checkout failing.",
        severity="SEV1",
    )
    ctx = _ctx(idempotency_key="run-1:appr-1", deadline=0.05)

    result = await ToolExecutor(clock, events, injector, random.Random(1)).run(spec, args, ctx)

    assert result.ok is True
    assert result.attempts == 2
    assert result.data is not None
    assert result.data["deduplicated"] is True
    assert len(incident_store.records) == 1
