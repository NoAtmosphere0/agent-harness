import asyncio
from datetime import UTC

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from harness.clock import FakeClock
from harness.domain.errors import RunAlreadyTerminalError
from harness.domain.models import (
    ApprovalRecord,
    ApprovalStatus,
    EventType,
    PendingToolCall,
    RunRecord,
    RunStatus,
    TerminationReason,
    ToolCallStatus,
)
from harness.store.repository import Repository
from harness.store.tables import ApprovalRow

# ------------------------------------------------------------------ runs


async def test_repository_create_and_get_run_roundtrip(repo: Repository, clock: FakeClock):
    created = await repo.create_run("check auth-service", {"max_steps": 5})

    loaded = await repo.get_run(created.id)

    assert loaded == created
    assert loaded.status is RunStatus.PENDING
    assert loaded.step_count == 0
    assert loaded.created_at == clock.now_utc()
    assert loaded.created_at.tzinfo is UTC


async def test_repository_get_unknown_run_returns_none(repo: Repository):
    assert await repo.get_run("does-not-exist") is None


async def test_repository_save_run_persists_mutable_fields(repo: Repository, clock: FakeClock):
    run = await repo.create_run("objective", {})
    clock.advance(5)
    pending = PendingToolCall(
        tool_call_id="call_1", tool_name="create_incident", args={"title": "x"}, approval_id="a1"
    )

    saved = await repo.save_run(
        run.model_copy(
            update={
                "status": RunStatus.WAITING_APPROVAL,
                "step_count": 3,
                "active_elapsed_ms": 1500,
                "pending_tool_call": pending,
            }
        )
    )
    loaded = await repo.get_run(run.id)

    assert loaded == saved
    assert loaded.pending_tool_call == pending
    assert loaded.updated_at == clock.now_utc()
    assert loaded.created_at == run.created_at


async def test_repository_save_run_clears_pending_call_to_null(repo: Repository):
    run = await repo.create_run("objective", {})
    pending = PendingToolCall(tool_call_id="c", tool_name="t", args={}, approval_id="a")
    await repo.save_run(run.model_copy(update={"pending_tool_call": pending}))

    await repo.save_run(
        run.model_copy(
            update={
                "status": RunStatus.COMPLETED,
                "termination_reason": TerminationReason.FINAL_ANSWER,
                "pending_tool_call": None,
            }
        )
    )
    loaded = await repo.get_run(run.id)

    assert loaded is not None
    assert loaded.pending_tool_call is None
    assert loaded.termination_reason is TerminationReason.FINAL_ANSWER


async def test_repository_save_unknown_run_raises(repo: Repository):
    run = await repo.create_run("objective", {})

    with pytest.raises(LookupError):
        await repo.save_run(run.model_copy(update={"id": "missing"}))


async def test_repository_list_runs_newest_first(repo: Repository, clock: FakeClock):
    first = await repo.create_run("first", {})
    clock.advance(1)
    second = await repo.create_run("second", {})

    runs = await repo.list_runs()

    assert [r.id for r in runs] == [second.id, first.id]


# ------------------------------------------------------------------ messages


async def test_repository_message_seq_is_per_run_and_ordered(repo: Repository):
    run_a = await repo.create_run("a", {})
    run_b = await repo.create_run("b", {})

    await repo.append_message(run_a.id, "user", "hello")
    await repo.append_message(run_b.id, "user", "other run")
    await repo.append_message(
        run_a.id, "assistant", None, tool_calls=[{"id": "c1", "name": "t", "arguments": "{}"}]
    )
    await repo.append_message(run_a.id, "tool", '{"ok": true}', tool_call_id="c1")

    messages = await repo.list_messages(run_a.id)

    assert [(m.seq, m.role) for m in messages] == [(1, "user"), (2, "assistant"), (3, "tool")]
    assert messages[1].tool_calls == [{"id": "c1", "name": "t", "arguments": "{}"}]
    assert messages[1].content is None
    assert messages[2].tool_call_id == "c1"
    assert [m.seq for m in await repo.list_messages(run_b.id)] == [1]


async def test_repository_concurrent_appends_get_unique_seqs(
    file_sessions: async_sessionmaker[AsyncSession], clock: FakeClock
):
    repo = Repository(file_sessions, clock)
    run = await repo.create_run("objective", {})

    await asyncio.gather(*(repo.append_message(run.id, "user", str(i)) for i in range(10)))

    assert [m.seq for m in await repo.list_messages(run.id)] == list(range(1, 11))


async def test_repository_rejects_message_for_unknown_run(repo: Repository):
    with pytest.raises(IntegrityError):
        await repo.append_message("no-such-run", "user", "hello")


# ------------------------------------------------------------------ tool calls


async def test_repository_count_tool_calls_by_args_hash(repo: Repository):
    run = await repo.create_run("objective", {})
    other = await repo.create_run("other", {})
    for run_id, args_hash in [(run.id, "h1"), (run.id, "h1"), (run.id, "h2"), (other.id, "h1")]:
        await repo.record_tool_call(
            run_id=run_id,
            step=1,
            tool_call_id="c",
            tool_name="get_service_status",
            args={"service_name": "auth-service"},
            args_hash=args_hash,
            status=ToolCallStatus.SUCCEEDED,
        )

    assert await repo.count_tool_calls(run.id, "h1") == 2
    assert await repo.count_tool_calls(run.id, "h2") == 1
    assert await repo.count_tool_calls(run.id, "h3") == 0


async def test_repository_records_invalid_tool_call_without_args(repo: Repository):
    run = await repo.create_run("objective", {})

    await repo.record_tool_call(
        run_id=run.id,
        step=2,
        tool_call_id="c9",
        tool_name="get_service_status",
        args=None,
        args_hash=None,
        status=ToolCallStatus.INVALID,
        error_type="ToolArgumentsError",
    )
    (call,) = await repo.list_tool_calls(run.id)

    assert call.status is ToolCallStatus.INVALID
    assert call.args is None
    assert call.error_type == "ToolArgumentsError"


# ------------------------------------------------------------------ pause / cancel / approvals

_ARGS = {"title": "payment-gateway: outage", "description": "x" * 20, "severity": "SEV1"}


async def _paused_run(repo: Repository) -> tuple[RunRecord, ApprovalRecord]:
    run = await repo.create_run("objective", {})
    running = await repo.save_run(run.model_copy(update={"status": RunStatus.RUNNING}))
    return await repo.pause_for_approval(
        running.model_copy(update={"step_count": 4}),
        tool_call_id="call_7",
        tool_name="create_incident",
        args=_ARGS,
    )


async def test_repository_pause_for_approval_links_run_and_approval(repo: Repository):
    paused, approval = await _paused_run(repo)

    stored_run = await repo.get_run(paused.id)
    stored_approval = await repo.get_approval(approval.id)

    assert stored_run == paused
    assert stored_run.status is RunStatus.WAITING_APPROVAL
    assert stored_run.step_count == 4
    assert stored_run.pending_tool_call == PendingToolCall(
        tool_call_id="call_7", tool_name="create_incident", args=_ARGS, approval_id=approval.id
    )
    assert stored_approval == approval
    assert approval.status is ApprovalStatus.PENDING
    assert approval.args == _ARGS


async def test_repository_pause_on_cancelled_run_creates_no_approval(
    repo: Repository, sessions: async_sessionmaker[AsyncSession]
):
    run = await repo.create_run("objective", {})
    await repo.cancel_run(run.id)

    with pytest.raises(RunAlreadyTerminalError):
        await repo.pause_for_approval(run, tool_call_id="c", tool_name="create_incident", args={})

    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(ApprovalRow)) == 0
    stored = await repo.get_run(run.id)
    assert stored is not None
    assert stored.status is RunStatus.CANCELLED


async def test_repository_save_run_refuses_to_overwrite_terminal_run(repo: Repository):
    run = await repo.create_run("objective", {})
    await repo.cancel_run(run.id)

    with pytest.raises(RunAlreadyTerminalError):
        await repo.save_run(run.model_copy(update={"status": RunStatus.RUNNING}))


async def test_repository_cancel_run_only_once(repo: Repository, clock: FakeClock):
    run = await repo.create_run("objective", {})

    first = await repo.cancel_run(run.id)
    second = await repo.cancel_run(run.id)

    assert first is not None
    assert first.status is RunStatus.CANCELLED
    assert first.termination_reason is TerminationReason.CANCELLED_BY_USER
    assert first.completed_at == clock.now_utc()
    assert second is None


async def test_repository_decide_approval_only_once(repo: Repository, clock: FakeClock):
    run, approval = await _paused_run(repo)
    clock.advance(30)

    first = await repo.decide_approval(
        run.id, approval.id, status=ApprovalStatus.APPROVED, decided_by="alice"
    )
    second = await repo.decide_approval(
        run.id, approval.id, status=ApprovalStatus.REJECTED, reason="changed my mind"
    )

    assert first is not None
    assert first.status is ApprovalStatus.APPROVED
    assert first.decided_by == "alice"
    assert first.decided_at == clock.now_utc()
    assert second is None
    stored = await repo.get_approval(approval.id)
    assert stored is not None
    assert stored.status is ApprovalStatus.APPROVED


async def test_repository_decide_approval_requires_waiting_run(repo: Repository):
    run, approval = await _paused_run(repo)
    await repo.cancel_run(run.id)

    result = await repo.decide_approval(run.id, approval.id, status=ApprovalStatus.APPROVED)

    assert result is None
    stored = await repo.get_approval(approval.id)
    assert stored is not None
    assert stored.status is ApprovalStatus.PENDING


async def test_repository_concurrent_decisions_have_one_winner(
    file_sessions: async_sessionmaker[AsyncSession], clock: FakeClock
):
    repo = Repository(file_sessions, clock)
    run, approval = await _paused_run(repo)

    results = await asyncio.gather(
        repo.decide_approval(run.id, approval.id, status=ApprovalStatus.APPROVED),
        repo.decide_approval(run.id, approval.id, status=ApprovalStatus.REJECTED),
    )

    assert sum(r is not None for r in results) == 1


async def test_repository_decide_approval_of_other_run_is_noop(repo: Repository):
    _, approval = await _paused_run(repo)
    other, _ = await _paused_run(repo)

    result = await repo.decide_approval(other.id, approval.id, status=ApprovalStatus.APPROVED)

    assert result is None


async def test_repository_decide_approval_rejects_pending_as_decision(repo: Repository):
    with pytest.raises(ValueError, match="approved or rejected"):
        await repo.decide_approval("r", "a", status=ApprovalStatus.PENDING)


# ------------------------------------------------------------------ events


async def test_repository_events_are_sequenced_and_jsonable(repo: Repository, clock: FakeClock):
    run = await repo.create_run("objective", {})

    await repo.append_event(run.id, EventType.RUN_STARTED, None, {})
    await repo.append_event(
        run.id,
        EventType.TOOL_SUCCEEDED,
        1,
        {"tool": "get_service_status", "status": RunStatus.RUNNING, "at": clock.now_utc()},
    )
    events = await repo.list_events(run.id)

    assert [(e.seq, e.type, e.step) for e in events] == [
        (1, EventType.RUN_STARTED, None),
        (2, EventType.TOOL_SUCCEEDED, 1),
    ]
    assert events[1].payload == {
        "tool": "get_service_status",
        "status": "running",
        "at": "2026-01-01T00:00:00Z",
    }
