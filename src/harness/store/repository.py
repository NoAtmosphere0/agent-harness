"""Data access for runs, history, approvals, events and the mock incident system.

Every method opens its own short transaction and returns pydantic records, so
callers never hold a session across an ``await`` on the LLM or a tool. This is
what lets a run be advanced step by step from different requests (P2).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, cast

from pydantic_core import to_jsonable_python
from sqlalchemy import CursorResult, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from harness.clock import Clock
from harness.domain.errors import RunAlreadyTerminalError
from harness.domain.models import (
    TERMINAL_STATUSES,
    ApprovalRecord,
    ApprovalStatus,
    EventRecord,
    EventType,
    MessageRecord,
    MessageRole,
    PendingToolCall,
    RunRecord,
    RunStatus,
    Severity,
    TerminationReason,
    ToolCallRecord,
    ToolCallStatus,
)
from harness.store.tables import (
    ApprovalRow,
    EventRow,
    IncidentRow,
    MessageRow,
    RunRow,
    ToolCallRow,
)
from harness.tools.incidents import IdempotencyKeyConflictError, IncidentRecord, format_incident_id

# Fields the loop may change after a run is created.
_RUN_MUTABLE_FIELDS = {
    "status",
    "termination_reason",
    "final_answer",
    "step_count",
    "active_elapsed_ms",
    "consecutive_tool_errors",
    "consecutive_malformed",
    "pending_tool_call",
    "completed_at",
}


class Repository:
    """Harness state: runs, messages, tool calls, approvals and trace events."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession], clock: Clock) -> None:
        self._sessions = sessions
        self._clock = clock
        # Sequence numbers are allocated as max(seq) + 1. The lock makes "allocate and
        # insert" atomic within this process; the (run_id, seq) unique constraint is the
        # backstop. Running a single process is a documented limitation.
        self._append_lock = asyncio.Lock()

    # ------------------------------------------------------------------ runs

    async def create_run(self, objective: str, config: dict[str, Any]) -> RunRecord:
        now = self._clock.now_utc()
        run = RunRecord(
            id=str(uuid.uuid4()),
            objective=objective,
            status=RunStatus.PENDING,
            config=config,
            created_at=now,
            updated_at=now,
        )
        async with self._sessions.begin() as session:
            session.add(RunRow(**run.model_dump()))
        return run

    async def get_run(self, run_id: str) -> RunRecord | None:
        async with self._sessions() as session:
            row = await session.get(RunRow, run_id)
            return RunRecord.model_validate(row) if row else None

    async def list_runs(self, limit: int = 50) -> list[RunRecord]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(RunRow).order_by(RunRow.created_at.desc(), RunRow.id).limit(limit)
            )
            return [RunRecord.model_validate(r) for r in rows]

    async def save_run(self, run: RunRecord) -> RunRecord:
        """Persist the run's mutable fields; returns the run with a fresh ``updated_at``.

        Refuses to overwrite a run that is already terminal in the database (raises
        ``RunAlreadyTerminalError``). That is how a cancel made by another request
        while the loop is mid-tick wins over the loop's next save.
        """
        saved = run.model_copy(update={"updated_at": self._clock.now_utc()})
        async with self._sessions.begin() as session:
            await self._update_active_run(session, saved)
        return saved

    async def pause_for_approval(
        self, run: RunRecord, *, tool_call_id: str, tool_name: str, args: dict[str, Any]
    ) -> tuple[RunRecord, ApprovalRecord]:
        """Create the approval and park the run on it, in one transaction.

        P2/P4: the approval, ``pending_tool_call`` and ``WAITING_APPROVAL`` become
        visible together. A decision can therefore never find a pending approval on
        a run that is not yet waiting for it, and a crash cannot leave one without
        the other.
        """
        now = self._clock.now_utc()
        approval = ApprovalRow(
            id=str(uuid.uuid4()),
            run_id=run.id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            args=args,
            status=ApprovalStatus.PENDING,
            requested_at=now,
        )
        paused = run.model_copy(
            update={
                "status": RunStatus.WAITING_APPROVAL,
                "pending_tool_call": PendingToolCall(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    args=args,
                    approval_id=approval.id,
                ),
                "updated_at": now,
            }
        )
        async with self._sessions.begin() as session:
            await self._update_active_run(session, paused)
            session.add(approval)
        return paused, ApprovalRecord.model_validate(approval)

    async def cancel_run(self, run_id: str) -> RunRecord | None:
        """Cancel a run unless it already finished. Returns ``None`` if it had.

        A single conditional update, so it needs no run lock and cannot race with
        the loop: whichever write lands first decides, and the loop's conditional
        ``save_run`` then refuses to overwrite the cancellation.
        """
        now = self._clock.now_utc()
        async with self._sessions.begin() as session:
            result = await session.execute(
                update(RunRow)
                .where(RunRow.id == run_id, _run_is_active())
                .values(
                    status=RunStatus.CANCELLED,
                    termination_reason=TerminationReason.CANCELLED_BY_USER,
                    completed_at=now,
                    updated_at=now,
                )
            )
            if cast(CursorResult[Any], result).rowcount != 1:
                return None
        return await self.get_run(run_id)

    async def _update_active_run(self, session: AsyncSession, run: RunRecord) -> None:
        values = run.model_dump(include=_RUN_MUTABLE_FIELDS | {"updated_at"})
        result = await session.execute(
            update(RunRow).where(RunRow.id == run.id, _run_is_active()).values(**values)
        )
        if cast(CursorResult[Any], result).rowcount == 1:
            return
        if await session.get(RunRow, run.id) is None:
            raise LookupError(f"run {run.id} does not exist")
        raise RunAlreadyTerminalError(f"run {run.id} already finished")

    # ------------------------------------------------------------------ messages

    async def append_message(
        self,
        run_id: str,
        role: MessageRole,
        content: str | None = None,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
    ) -> MessageRecord:
        async with self._append_lock, self._sessions.begin() as session:
            row = MessageRow(
                run_id=run_id,
                seq=await _next_seq(session, MessageRow, run_id),
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
                created_at=self._clock.now_utc(),
            )
            session.add(row)
        return MessageRecord.model_validate(row)

    async def list_messages(self, run_id: str) -> list[MessageRecord]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(MessageRow).where(MessageRow.run_id == run_id).order_by(MessageRow.seq)
            )
            return [MessageRecord.model_validate(r) for r in rows]

    # ------------------------------------------------------------------ tool calls

    async def record_tool_call(
        self,
        *,
        run_id: str,
        step: int,
        tool_call_id: str,
        tool_name: str,
        args: dict[str, Any] | None,
        args_hash: str | None,
        status: ToolCallStatus,
        result: dict[str, Any] | None = None,
        error_type: str | None = None,
        attempts: int = 0,
        duration_ms: int = 0,
    ) -> ToolCallRecord:
        row = ToolCallRow(
            run_id=run_id,
            step=step,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            args=args,
            args_hash=args_hash,
            status=status,
            result=result,
            error_type=error_type,
            attempts=attempts,
            duration_ms=duration_ms,
            created_at=self._clock.now_utc(),
        )
        async with self._sessions.begin() as session:
            session.add(row)
        return ToolCallRecord.model_validate(row)

    async def count_tool_calls(self, run_id: str, args_hash: str) -> int:
        """How often this exact call (tool + canonical args) was already made in the run."""
        async with self._sessions() as session:
            count = await session.scalar(
                select(func.count())
                .select_from(ToolCallRow)
                .where(ToolCallRow.run_id == run_id, ToolCallRow.args_hash == args_hash)
            )
            return count or 0

    async def list_tool_calls(self, run_id: str) -> list[ToolCallRecord]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(ToolCallRow).where(ToolCallRow.run_id == run_id).order_by(ToolCallRow.id)
            )
            return [ToolCallRecord.model_validate(r) for r in rows]

    # ------------------------------------------------------------------ approvals

    async def get_approval(self, approval_id: str) -> ApprovalRecord | None:
        async with self._sessions() as session:
            row = await session.get(ApprovalRow, approval_id)
            return ApprovalRecord.model_validate(row) if row else None

    async def decide_approval(
        self,
        run_id: str,
        approval_id: str,
        *,
        status: ApprovalStatus,
        reason: str | None = None,
        decided_by: str | None = None,
    ) -> ApprovalRecord | None:
        """Record a decision only if the approval is pending and its run is waiting.

        Returns ``None`` otherwise: already decided, not this run's approval, or the
        run is no longer waiting (e.g. cancelled). The conditional update makes a
        second, concurrent decision a no-op instead of overwriting the first.
        """
        if status is ApprovalStatus.PENDING:
            raise ValueError("a decision must be approved or rejected")
        async with self._sessions.begin() as session:
            result = await session.execute(
                update(ApprovalRow)
                .where(
                    ApprovalRow.id == approval_id,
                    ApprovalRow.run_id == run_id,
                    ApprovalRow.status == ApprovalStatus.PENDING,
                    select(RunRow.id)
                    .where(RunRow.id == run_id, RunRow.status == RunStatus.WAITING_APPROVAL)
                    .exists(),
                )
                .values(
                    status=status,
                    reason=reason,
                    decided_by=decided_by,
                    decided_at=self._clock.now_utc(),
                )
            )
            if cast(CursorResult[Any], result).rowcount != 1:
                return None
        return await self.get_approval(approval_id)

    # ------------------------------------------------------------------ events

    async def append_event(
        self, run_id: str, event_type: EventType, step: int | None, payload: dict[str, Any]
    ) -> EventRecord:
        async with self._append_lock, self._sessions.begin() as session:
            row = EventRow(
                run_id=run_id,
                seq=await _next_seq(session, EventRow, run_id),
                ts=self._clock.now_utc(),
                type=event_type,
                step=step,
                payload=to_jsonable_python(payload),
            )
            session.add(row)
        return EventRecord.model_validate(row)

    async def list_events(self, run_id: str) -> list[EventRecord]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(EventRow).where(EventRow.run_id == run_id).order_by(EventRow.seq)
            )
            return [EventRecord.model_validate(r) for r in rows]


def _run_is_active() -> ColumnElement[bool]:
    """SQL condition: the run has not reached a terminal status."""
    return RunRow.status.not_in([s.value for s in TERMINAL_STATUSES])


async def _next_seq(
    session: AsyncSession, table: type[MessageRow] | type[EventRow], run_id: str
) -> int:
    current = await session.scalar(
        select(func.coalesce(func.max(table.seq), 0)).where(table.run_id == run_id)
    )
    return int(current or 0) + 1


class SqlIncidentStore:
    """``IncidentStore`` backed by the ``incidents`` table (the mock external system)."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession], clock: Clock) -> None:
        self._sessions = sessions
        self._clock = clock

    async def get_by_idempotency_key(self, idempotency_key: str) -> IncidentRecord | None:
        async with self._sessions() as session:
            row = await session.scalar(
                select(IncidentRow).where(IncidentRow.idempotency_key == idempotency_key)
            )
            return _incident_record(row) if row else None

    async def create(
        self, *, idempotency_key: str, title: str, description: str, severity: Severity
    ) -> IncidentRecord:
        row = IncidentRow(
            idempotency_key=idempotency_key,
            title=title,
            description=description,
            severity=severity,
            status="open",
            created_at=self._clock.now_utc(),
        )
        try:
            async with self._sessions.begin() as session:
                session.add(row)
        except IntegrityError as exc:
            # P6: the unique key is the real guarantee; translate it for the tool.
            raise IdempotencyKeyConflictError(
                f"an incident with idempotency key {idempotency_key!r} already exists"
            ) from exc
        return _incident_record(row)

    async def list_incidents(self) -> list[IncidentRecord]:
        async with self._sessions() as session:
            rows = await session.scalars(select(IncidentRow).order_by(IncidentRow.id))
            return [_incident_record(r) for r in rows]


def _incident_record(row: IncidentRow) -> IncidentRecord:
    return IncidentRecord(
        incident_id=format_incident_id(row.id),
        title=row.title,
        description=row.description,
        severity=cast(Severity, row.severity),
        created_at=row.created_at,
    )
