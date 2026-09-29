"""Human decisions on paused tool calls (PLAN §9.2).

Used by both the API and the CLI, so the rules are the same everywhere: a
decision is recorded at most once, only while the run is waiting for it, and it
resumes the run.
"""

from __future__ import annotations

from typing import Literal

from harness.domain.errors import (
    ApprovalNotFoundError,
    ApprovalNotPendingError,
    RunNotFoundError,
    RunNotWaitingError,
)
from harness.domain.models import ApprovalRecord, ApprovalStatus, EventType
from harness.observability.tracer import EventSink
from harness.services.run_service import RunService
from harness.store.repository import Repository

Decision = Literal["approve", "reject"]


class ApprovalService:
    def __init__(self, *, repo: Repository, runs: RunService, events: EventSink) -> None:
        self._repo = repo
        self._runs = runs
        self._events = events

    async def decide(
        self,
        run_id: str,
        approval_id: str,
        decision: Decision,
        *,
        reason: str | None = None,
        decided_by: str | None = None,
    ) -> ApprovalRecord:
        """Record the decision and schedule the run to resume.

        Raises ``RunNotFoundError`` / ``ApprovalNotFoundError`` (404),
        ``ApprovalNotPendingError`` if it was already decided, and
        ``RunNotWaitingError`` if the run stopped waiting, e.g. it was cancelled (409).
        """
        run = await self._repo.get_run(run_id)
        if run is None:
            raise RunNotFoundError(f"run {run_id} not found")
        approval = await self._repo.get_approval(approval_id)
        if approval is None or approval.run_id != run_id:
            raise ApprovalNotFoundError(f"approval {approval_id} not found for run {run_id}")

        status = ApprovalStatus.APPROVED if decision == "approve" else ApprovalStatus.REJECTED
        # Conditional update: pending approval AND waiting run, checked atomically.
        decided = await self._repo.decide_approval(
            run_id, approval_id, status=status, reason=reason, decided_by=decided_by
        )
        if decided is None:
            current = await self._repo.get_approval(approval_id)
            if current is not None and current.status is not ApprovalStatus.PENDING:
                raise ApprovalNotPendingError(f"approval {approval_id} is already {current.status}")
            raise RunNotWaitingError(f"run {run_id} is no longer waiting for approval")

        await self._events.emit(
            run_id,
            EventType.APPROVAL_DECIDED,
            run.step_count,
            approval_id=approval_id,
            decision=status,
            reason=reason,
            decided_by=decided_by,
        )
        self._runs.start(run_id)
        return decided
