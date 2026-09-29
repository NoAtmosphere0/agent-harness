"""Creating, advancing and cancelling runs (PLAN §8.1, §12)."""

from __future__ import annotations

import asyncio

from harness.config import ConfigOverrides, Settings, build_run_config
from harness.core.loop import AgentLoop
from harness.domain.errors import RunAlreadyTerminalError, RunNotFoundError
from harness.domain.models import EventType, RunRecord, RunStatus, TerminationReason
from harness.llm.prompts import SYSTEM_PROMPT
from harness.observability.logging import get_logger
from harness.observability.tracer import EventSink
from harness.store.repository import Repository
from harness.tools.faults import FaultInjector


class RunService:
    """Owns the per-run locks and the background tasks that advance runs."""

    def __init__(
        self,
        *,
        repo: Repository,
        loop: AgentLoop,
        events: EventSink,
        faults: FaultInjector,
        settings: Settings,
        model_name: str,
    ) -> None:
        self._repo = repo
        self._loop = loop
        self._events = events
        self._faults = faults
        self._settings = settings
        self._model_name = model_name
        self._locks: dict[str, asyncio.Lock] = {}
        # Background advances and the run each one drives.
        self._tasks: dict[asyncio.Task[RunStatus], str] = {}
        self._log = get_logger(component="run_service")

    async def create_run(
        self, objective: str, overrides: ConfigOverrides | None = None
    ) -> RunRecord:
        """Persist a new run with its config snapshot and opening messages."""
        config = build_run_config(self._settings, self._model_name, overrides)
        run = await self._repo.create_run(objective, config.model_dump(mode="json"))
        # The system prompt is stored as message 1, so the history records exactly
        # what the model was told.
        await self._repo.append_message(run.id, "system", SYSTEM_PROMPT)
        await self._repo.append_message(run.id, "user", objective)
        await self._events.emit(
            run.id, EventType.RUN_CREATED, 0, objective=objective, config=run.config
        )
        return run

    async def advance(self, run_id: str) -> RunStatus:
        """Advance a run, one ``advance()`` per run at a time (§8.1).

        A second caller waits for the lock, then ``advance()`` re-reads the run and
        decides afresh whether there is anything to do.
        """
        lock = self._locks.setdefault(run_id, asyncio.Lock())
        async with lock:
            return await self._loop.advance(run_id)

    def start(self, run_id: str) -> asyncio.Task[RunStatus]:
        """Advance in the background, so an API request can return immediately."""
        task = asyncio.create_task(self.advance(run_id), name=f"advance:{run_id}")
        # asyncio only keeps weak references to tasks; hold one until it is done.
        self._tasks[task] = run_id
        task.add_done_callback(self._on_task_done)
        return task

    async def drain(self) -> None:
        """Wait until no background advance is running (tests, shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def cancel(self, run_id: str) -> RunRecord:
        """Cancel a run that has not finished.

        Takes no lock: it is one conditional update, so it works even while an
        ``advance()`` is mid-tick. The loop sees the cancellation at its next tick,
        or when its next save is refused, and stops.
        """
        cancelled = await self._repo.cancel_run(run_id)
        if cancelled is None:
            if await self._repo.get_run(run_id) is None:
                raise RunNotFoundError(f"run {run_id} not found")
            raise RunAlreadyTerminalError(f"run {run_id} already finished")
        self._faults.clear_run(run_id)
        await self._events.emit(
            run_id,
            EventType.RUN_FINISHED,
            cancelled.step_count,
            status=RunStatus.CANCELLED,
            reason=TerminationReason.CANCELLED_BY_USER,
        )
        return cancelled

    async def shutdown(self) -> None:
        """Stop background advances and fail the runs they were driving (§12).

        A run interrupted mid-tick cannot be resumed safely (resume-after-crash is
        out of scope), so it is marked FAILED/internal_error with message "shutdown"
        rather than left in RUNNING. Runs waiting for approval are not in flight
        and stay resumable.
        """
        in_flight = {task: run_id for task, run_id in self._tasks.items() if not task.done()}
        for task in in_flight:
            task.cancel()
        await asyncio.gather(*in_flight, return_exceptions=True)
        # Only runs whose advance() was actually cut short; a task that finished on
        # its own (e.g. it just paused for approval) left its run in a valid state.
        interrupted = {run_id for task, run_id in in_flight.items() if task.cancelled()}
        for run_id in sorted(interrupted):
            status = await self._loop.fail_run(run_id, error_type="Shutdown", message="shutdown")
            self._log.warning("run_interrupted_by_shutdown", run_id=run_id, status=status)

    def _on_task_done(self, task: asyncio.Task[RunStatus]) -> None:
        self._tasks.pop(task, None)
        if not task.cancelled() and (error := task.exception()) is not None:
            # advance() already turns run failures into FAILED runs; reaching this
            # means even that failed (e.g. the database is gone).
            self._log.error("advance_task_failed", task=task.get_name(), exc_info=error)
