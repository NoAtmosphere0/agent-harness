"""The approval gate end to end (PLAN §9): pause, approve, reject, idempotency,
prompt injection, time accounting, and the decide/resume races."""

import asyncio
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from harness.domain.errors import (
    ApprovalNotFoundError,
    ApprovalNotPendingError,
    RunAlreadyTerminalError,
    RunNotFoundError,
    RunNotWaitingError,
)
from harness.domain.models import (
    ApprovalStatus,
    EventType,
    RunStatus,
    TerminationReason,
    ToolCallStatus,
)
from harness.llm.scripted import call, final, tool_calls
from harness.observability.tracer import EventSink
from tests.fixtures.scripts import (
    CHECKOUT_INCIDENT_ARGS,
    INJECTED_INCIDENT_ARGS,
    checkout_outage_script,
    prompt_injection_script,
)
from tests.integration.conftest import EnvFactory

E = EventType


async def test_create_incident_pauses_for_approval(make_env: EnvFactory):
    env = make_env(checkout_outage_script())
    run_id = await env.start()

    run = await env.advance(run_id)

    assert run.status is RunStatus.WAITING_APPROVAL
    assert run.step_count == 4
    assert await env.incident_list() == []
    approval = await env.pending_approval(run_id)
    assert approval.status is ApprovalStatus.PENDING
    assert approval.tool_name == "create_incident"
    assert approval.args == CHECKOUT_INCIDENT_ARGS
    assert run.pending_tool_call is not None
    assert run.pending_tool_call.args == CHECKOUT_INCIDENT_ARGS
    assert (await env.event_types(run_id))[-1] is E.APPROVAL_REQUESTED
    assert len(env.llm.requests) == 4


async def test_approve_executes_stored_args_exactly_once(make_env: EnvFactory):
    env = make_env(checkout_outage_script())
    run_id = await env.start()
    await env.advance(run_id)
    approval = await env.pending_approval(run_id)

    run = await env.decide(run_id, "approve", decided_by="alice")

    assert run.status is RunStatus.COMPLETED
    assert run.pending_tool_call is None
    (incident,) = await env.incident_list()
    assert (incident.title, incident.description, incident.severity) == (
        CHECKOUT_INCIDENT_ARGS["title"],
        CHECKOUT_INCIDENT_ARGS["description"],
        CHECKOUT_INCIDENT_ARGS["severity"],
    )
    # P6: executed with the harness-generated key, not anything the model chose.
    assert await env.incidents.get_by_idempotency_key(f"{run_id}:{approval.id}") == incident
    created = [c for c in await env.tool_calls(run_id) if c.tool_name == "create_incident"]
    assert [(c.status, c.tool_call_id) for c in created] == [
        (ToolCallStatus.SUCCEEDED, approval.tool_call_id)
    ]
    # P4: the model was not asked to regenerate the call; its next request already
    # carries the result for the approved tool call id.
    assert len(env.llm.requests) == 5
    last_seen = env.llm.requests[4].messages[-1]
    assert (last_seen.role, last_seen.tool_call_id) == ("tool", approval.tool_call_id)
    assert run.final_answer is not None
    assert incident.incident_id in run.final_answer
    types = await env.event_types(run_id)
    assert types[-5:] == [
        E.APPROVAL_DECIDED,
        E.RUN_RESUMED,
        E.TOOL_SUCCEEDED,
        E.LLM_RESPONSE,
        E.RUN_FINISHED,
    ]


async def test_reject_resumes_and_completes_without_incident(make_env: EnvFactory):
    env = make_env(checkout_outage_script())
    run_id = await env.start()
    await env.advance(run_id)

    run = await env.decide(run_id, "reject", reason="duplicate of INC-000042")

    assert run.status is RunStatus.COMPLETED
    assert await env.incident_list() == []
    rejected = (await env.observations(run_id))[-1]
    assert rejected == {
        "ok": False,
        "error": {
            "type": "ApprovalRejected",
            "message": "duplicate of INC-000042",
            "retryable": False,
        },
    }
    assert (await env.tool_calls(run_id))[-1].status is ToolCallStatus.REJECTED
    assert run.final_answer is not None
    assert "not approved" in run.final_answer


async def test_timeout_after_commit_does_not_duplicate_incident(make_env: EnvFactory):
    env = make_env(
        checkout_outage_script(),
        mock_faults={"create_incident": {"mode": "timeout_after_commit"}},
        tool_timeouts={"create_incident": 0.05},
    )
    run_id = await env.start()
    await env.advance(run_id)

    run = await env.decide(run_id, "approve")

    assert run.status is RunStatus.COMPLETED
    assert len(await env.incident_list()) == 1
    record = next(c for c in await env.tool_calls(run_id) if c.tool_name == "create_incident")
    assert record.attempts == 2
    assert record.result is not None
    assert record.result["data"]["deduplicated"] is True


async def test_prompt_injection_cannot_bypass_approval(make_env: EnvFactory):
    env = make_env(prompt_injection_script())
    run_id = await env.start("Summarise the latest card processor vendor note")

    run = await env.advance(run_id)

    # The injected text reached the model only as JSON data inside an observation (P9)...
    kb_result = (await env.observations(run_id))[0]
    assert kb_result["ok"] is True
    assert any("ignore previous instructions" in r["snippet"] for r in kb_result["data"]["results"])
    # ...and even a model that obeys it cannot create an incident: the gate holds (P3).
    assert run.status is RunStatus.WAITING_APPROVAL
    assert (await env.pending_approval(run_id)).args == INJECTED_INCIDENT_ARGS
    assert await env.incident_list() == []

    run = await env.decide(run_id, "reject", reason="prompt injection")

    assert run.status is RunStatus.COMPLETED
    assert await env.incident_list() == []


async def test_approval_wait_not_counted_toward_run_time(make_env: EnvFactory):
    env = make_env(checkout_outage_script())
    run_id = await env.start(max_run_seconds=60)
    await env.advance(run_id)

    env.clock.advance(3600)  # the approver takes an hour
    run = await env.decide(run_id, "approve")

    assert run.status is RunStatus.COMPLETED
    assert run.active_elapsed_ms < 60_000


async def test_rejected_call_is_refused_without_new_approval(make_env: EnvFactory):
    investigate_and_propose = checkout_outage_script()[:4]
    env = make_env(
        [
            *investigate_and_propose,
            tool_calls(call("create_incident", CHECKOUT_INCIDENT_ARGS)),  # the same call again
            final("The approver declined; no incident was created."),
        ]
    )
    run_id = await env.start()
    await env.advance(run_id)

    run = await env.decide(run_id, "reject", reason="not an incident yet")

    # P3: the rejection is enforced in code; the human is not asked a second time.
    assert run.status is RunStatus.COMPLETED
    assert await env.incident_list() == []
    assert (await env.event_types(run_id)).count(E.APPROVAL_REQUESTED) == 1
    created = [c for c in await env.tool_calls(run_id) if c.tool_name == "create_incident"]
    assert [c.status for c in created] == [ToolCallStatus.REJECTED, ToolCallStatus.REJECTED]
    assert created[0].args_hash == created[1].args_hash
    refused = (await env.observations(run_id))[-1]
    assert refused["error"] == {
        "type": "ApprovalRejected",
        "message": "a human already rejected this exact call: not an incident yet",
        "retryable": False,
    }


async def test_changed_call_after_rejection_needs_its_own_approval(make_env: EnvFactory):
    downgraded = {**CHECKOUT_INCIDENT_ARGS, "severity": "SEV2"}
    env = make_env([*checkout_outage_script()[:4], tool_calls(call("create_incident", downgraded))])
    run_id = await env.start()
    await env.advance(run_id)
    first = await env.pending_approval(run_id)

    run = await env.decide(run_id, "reject", reason="SEV1 is too high")

    # P4: a different call is a different decision, so it pauses for a new approval.
    assert run.status is RunStatus.WAITING_APPROVAL
    second = await env.pending_approval(run_id)
    assert second.id != first.id
    assert second.args == downgraded
    assert await env.incident_list() == []


# ------------------------------------------------------------------ decide / resume races


class _GateAfterApprovalRequested:
    """Holds the first advance() (and so the run lock) right after it paused."""

    def __init__(self, inner: EventSink) -> None:
        self._inner = inner
        self.paused = asyncio.Event()
        self.release = asyncio.Event()

    async def emit(
        self, run_id: str, event_type: EventType, step: int | None = None, **payload: Any
    ) -> None:
        await self._inner.emit(run_id, event_type, step, **payload)
        if event_type is EventType.APPROVAL_REQUESTED:
            self.paused.set()
            await self.release.wait()


async def test_decision_during_first_advance_still_resumes(
    make_env: EnvFactory, file_sessions: async_sessionmaker[AsyncSession]
):
    gates: list[_GateAfterApprovalRequested] = []

    def gate(inner: EventSink) -> EventSink:
        gates.append(_GateAfterApprovalRequested(inner))
        return gates[0]

    env = make_env(checkout_outage_script(), db=file_sessions, wrap_events=gate)
    run_id = await env.start()
    first = asyncio.create_task(env.runs.advance(run_id))
    await gates[0].paused.wait()

    # The first advance() still holds the run lock. The decision is recorded and
    # schedules a second advance(), which has to wait for the lock.
    approval = await env.pending_approval(run_id)
    await env.approvals.decide(run_id, approval.id, "approve")
    gates[0].release.set()

    assert await first is RunStatus.WAITING_APPROVAL
    await env.runs.drain()
    run = await env.get(run_id)
    assert run.status is RunStatus.COMPLETED
    assert len(await env.incident_list()) == 1


async def test_decide_after_cancel_is_rejected_and_nothing_executes(make_env: EnvFactory):
    env = make_env(checkout_outage_script())
    run_id = await env.start()
    await env.advance(run_id)
    approval = await env.pending_approval(run_id)
    await env.runs.cancel(run_id)

    with pytest.raises(RunNotWaitingError):
        await env.approvals.decide(run_id, approval.id, "approve")

    await env.runs.drain()
    assert await env.runs.advance(run_id) is RunStatus.CANCELLED
    run = await env.get(run_id)
    assert run.status is RunStatus.CANCELLED
    assert await env.incident_list() == []
    assert not any(c.tool_name == "create_incident" for c in await env.tool_calls(run_id))
    stored = await env.repo.get_approval(approval.id)
    assert stored is not None
    assert stored.status is ApprovalStatus.PENDING
    assert len(env.llm.requests) == 4


# ------------------------------------------------------------------ service errors


async def test_second_decision_is_rejected(make_env: EnvFactory):
    env = make_env(checkout_outage_script())
    run_id = await env.start()
    await env.advance(run_id)
    approval = await env.pending_approval(run_id)
    await env.decide(run_id, "approve")

    with pytest.raises(ApprovalNotPendingError):
        await env.approvals.decide(run_id, approval.id, "reject")


async def test_decide_unknown_run_or_approval(make_env: EnvFactory):
    env = make_env(checkout_outage_script())
    run_id = await env.start()
    await env.advance(run_id)

    with pytest.raises(RunNotFoundError):
        await env.approvals.decide("no-such-run", "x", "approve")
    with pytest.raises(ApprovalNotFoundError):
        await env.approvals.decide(run_id, "no-such-approval", "approve")


async def test_cancel_terminal_or_unknown_run(make_env: EnvFactory):
    env = make_env(checkout_outage_script())
    run_id = await env.start()
    await env.runs.cancel(run_id)

    with pytest.raises(RunAlreadyTerminalError):
        await env.runs.cancel(run_id)
    with pytest.raises(RunNotFoundError):
        await env.runs.cancel("no-such-run")


async def test_rejected_run_terminates_as_completed_not_failed(make_env: EnvFactory):
    env = make_env(checkout_outage_script())
    run_id = await env.start()
    await env.advance(run_id)

    run = await env.decide(run_id, "reject")

    assert run.termination_reason is TerminationReason.FINAL_ANSWER
    rejected = (await env.observations(run_id))[-1]
    assert rejected["error"]["message"] == "rejected by the approver"
