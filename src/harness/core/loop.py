"""The agent loop (PLAN §8).

``AgentLoop.advance(run_id)`` moves a persisted run forward one tick at a time
until it finishes or pauses for human approval, then returns. It keeps nothing
in memory between calls (P2): a later ``advance()``, possibly from another
request, picks the run up from the database.

One tick:

1. stop if the run was cancelled;
2. if the run paused on an approval that has since been decided, execute or
   reject the stored call;
3. check the step and active-time limits;
4. ask the model for the next action (with the only LLM retry layer);
5. act on it: record a final answer, repair an empty reply, or dispatch one
   tool call (validating it, and pausing if it needs approval);
6. check the consecutive-error limits and save.

Tool failures become observations for the model (P7); only harness-level
problems (limits, LLM unavailable, internal errors) end a run.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from harness.clock import Clock
from harness.config import RunConfig
from harness.core import guardrails, observations
from harness.core.parser import EmptyResponse, FinalAnswer, ToolCallAction, parse
from harness.domain.errors import (
    HarnessError,
    LLMError,
    LLMTimeoutError,
    RunAlreadyTerminalError,
    RunNotFoundError,
    ToolArgumentsError,
    ToolError,
    describe_validation_error,
)
from harness.domain.models import (
    STATUS_FOR_REASON,
    TERMINAL_STATUSES,
    ApprovalStatus,
    EventType,
    RunRecord,
    RunStatus,
    TerminationReason,
    ToolCallRecord,
    ToolCallStatus,
    ToolErrorInfo,
)
from harness.llm.base import ChatMessage, LLMClient, LLMResponse, LLMToolCall
from harness.llm.prompts import (
    DEFAULT_REJECTION_REASON,
    EMPTY_RESPONSE_REPAIR,
    NOT_EXECUTED_MESSAGE,
)
from harness.observability.logging import get_logger
from harness.observability.tracer import EventSink
from harness.store.repository import Repository
from harness.tools.executor import ToolExecutor
from harness.tools.faults import FaultInjector
from harness.tools.registry import ToolRegistry
from harness.tools.spec import RetryPolicy, ToolContext, ToolSpec


@dataclass
class _ActiveRun:
    """What ``advance()`` holds in memory while it drives one run."""

    run: RunRecord
    config: RunConfig
    # Monotonic time up to which active time has been added to the run (P8).
    charged_until: float


@dataclass(frozen=True)
class _LLMFailure:
    reason: TerminationReason
    details: dict[str, Any]


class AgentLoop:
    def __init__(
        self,
        *,
        repo: Repository,
        llm: LLMClient,
        registry: ToolRegistry,
        executor: ToolExecutor,
        faults: FaultInjector,
        events: EventSink,
        clock: Clock,
        rng: random.Random | None = None,
    ) -> None:
        self._repo = repo
        self._llm = llm
        self._registry = registry
        self._executor = executor
        self._faults = faults
        self._events = events
        self._clock = clock
        self._rng = rng or random.Random()
        self._log = get_logger(component="loop")

    # ------------------------------------------------------------------ entry point

    async def advance(self, run_id: str) -> RunStatus:
        """Drive the run until it finishes or pauses for approval; return its status.

        The caller holds the run's lock (``RunService``), so at most one
        ``advance()`` per run is active. Calling it on a run that has nothing to
        do is harmless: it returns the current status.
        """
        run = await self._repo.get_run(run_id)
        if run is None:
            raise RunNotFoundError(f"run {run_id} not found")
        if not await self._ready_to_advance(run):
            return run.status
        try:
            return await self._drive(run)
        except RunAlreadyTerminalError:
            # Cancelled by another request while this tick was in flight: the
            # conditional save refused to overwrite the cancellation, which already
            # recorded run_finished. Stop here without touching the run.
            self._faults.clear_run(run_id)
            stored = await self._repo.get_run(run_id)
            return stored.status if stored else RunStatus.CANCELLED
        except Exception as error:
            # §8.2: whatever goes wrong, a run never stays stuck in RUNNING.
            return await self._fail_internal(run_id, error)

    async def _ready_to_advance(self, run: RunRecord) -> bool:
        if run.status in (RunStatus.PENDING, RunStatus.RUNNING):
            return True
        # A paused run resumes once its approval is decided. Checking the approval
        # itself (not only the run status) means a decision that arrived while an
        # earlier advance() still held the lock is picked up by the next advance().
        if run.status is RunStatus.WAITING_APPROVAL and run.pending_tool_call:
            approval = await self._repo.get_approval(run.pending_tool_call.approval_id)
            return approval is not None and approval.status is not ApprovalStatus.PENDING
        return False

    async def _drive(self, run: RunRecord) -> RunStatus:
        starting = run.status is RunStatus.PENDING
        config = RunConfig.model_validate(run.config)
        self._faults.set_run_faults(run.id, config.faults)
        # P8: the clock starts now. Time before this call, such as waiting for a
        # human to approve, is never charged to the run.
        active = _ActiveRun(
            run=run.model_copy(update={"status": RunStatus.RUNNING}),
            config=config,
            charged_until=self._clock.monotonic(),
        )
        await self._save(active)
        await self._emit(active, EventType.RUN_STARTED if starting else EventType.RUN_RESUMED)
        while True:
            status = await self._tick(active)
            if status is not None:
                return status

    # ------------------------------------------------------------------ one tick

    async def _tick(self, active: _ActiveRun) -> RunStatus | None:
        """One step of the run. Returns the status if the run stopped or paused."""
        run = active.run

        stored = await self._repo.get_run(run.id)
        if stored is None or stored.status is RunStatus.CANCELLED:
            self._faults.clear_run(run.id)
            return RunStatus.CANCELLED

        if run.pending_tool_call:
            await self._resolve_pending(active)

        if reason := guardrails.check(run, active.config):
            return await self._finish(active, reason)

        response = await self._call_llm(active)
        if isinstance(response, _LLMFailure):
            return await self._finish(active, response.reason, **response.details)
        run.step_count += 1

        match parse(response):
            case EmptyResponse():
                run.consecutive_malformed += 1
                await self._emit(
                    active,
                    EventType.LLM_MALFORMED_OUTPUT,
                    problem="empty_response",
                    consecutive=run.consecutive_malformed,
                )
                await self._repo.append_message(run.id, "user", EMPTY_RESPONSE_REPAIR)
            case FinalAnswer(text=text):
                await self._repo.append_message(run.id, "assistant", text)
                run.final_answer = text
                return await self._finish(active, TerminationReason.FINAL_ANSWER)
            case ToolCallAction(call=call, extra_calls=extra_calls):
                await self._repo.append_message(
                    run.id,
                    "assistant",
                    response.text,
                    tool_calls=[c.model_dump() for c in response.tool_calls],
                )
                for extra in extra_calls:
                    await self._refuse_extra_call(active, extra)
                if (status := await self._dispatch(active, call)) is not None:
                    return status

        if run.consecutive_malformed > active.config.max_parse_retries:
            return await self._finish(active, TerminationReason.MALFORMED_LLM_OUTPUT)
        if run.consecutive_tool_errors >= active.config.max_consecutive_tool_errors:
            return await self._finish(active, TerminationReason.TOO_MANY_TOOL_ERRORS)
        await self._save(active)
        return None

    # ------------------------------------------------------------------ LLM call

    async def _call_llm(self, active: _ActiveRun) -> LLMResponse | _LLMFailure:
        """Ask the model for the next action.

        P5: this is the only place LLM calls are retried (the SDK client is created
        with ``max_retries=0``). Timeouts and retryable provider errors back off and
        try again; anything else ends the run.
        """
        run, config = active.run, active.config
        step = run.step_count + 1
        messages = [ChatMessage.from_record(m) for m in await self._repo.list_messages(run.id)]
        tools = self._registry.llm_tools()
        policy = RetryPolicy(max_retries=config.llm_max_retries)
        attempt = 0
        while True:
            attempt += 1
            remaining_s = self._remaining_s(active)
            if remaining_s <= 0:
                # Backoff sleeps used up the run's time budget.
                return _LLMFailure(TerminationReason.MAX_RUN_TIME, {})
            timeout_s = min(config.llm_timeout_seconds, remaining_s)
            started = self._clock.monotonic()
            try:
                response = await self._complete(messages, tools, timeout_s)
            except LLMError as error:
                if error.retryable and attempt <= policy.max_retries:
                    delay_s = policy.backoff_delay(attempt, self._rng)
                    await self._emit(
                        active,
                        EventType.LLM_RETRY,
                        step=step,
                        attempt=attempt,
                        error_type=type(error).__name__,
                        message=error.message,
                        delay_s=round(delay_s, 3),
                    )
                    await self._clock.sleep(delay_s)
                    continue
                reason = (
                    TerminationReason.LLM_UNAVAILABLE
                    if error.retryable
                    else TerminationReason.LLM_REQUEST_REJECTED
                )
                details = {
                    "error_type": type(error).__name__,
                    "message": error.message,
                    "attempts": attempt,
                }
                return _LLMFailure(reason, details)

            await self._emit(
                active,
                EventType.LLM_RESPONSE,
                step=step,
                attempt=attempt,
                latency_ms=round((self._clock.monotonic() - started) * 1000),
                usage=response.usage,
                tool_calls=[c.name for c in response.tool_calls],
                has_text=bool(response.text and response.text.strip()),
            )
            return response

    async def _complete(
        self, messages: list[ChatMessage], tools: list[dict[str, Any]], timeout_s: float
    ) -> LLMResponse:
        # The client is also given the timeout; this is the backstop if it ignores it.
        try:
            async with asyncio.timeout(timeout_s):
                return await self._llm.complete(messages, tools, timeout_s)
        except TimeoutError:
            raise LLMTimeoutError(f"no response within {timeout_s:g}s") from None

    # ------------------------------------------------------------------ dispatch

    async def _dispatch(self, active: _ActiveRun, call: LLMToolCall) -> RunStatus | None:
        """Validate and run one tool call (PLAN §8.4).

        Returns ``None`` to continue, ``WAITING_APPROVAL`` if the call needs a
        human decision, or a terminal status if a limit was hit.
        """
        run, config = active.run, active.config
        raw_args: dict[str, Any] | None = None
        try:
            raw_args = _parse_arguments(call)
            spec = self._registry.get(call.name)
            args = _validate_arguments(spec, raw_args)
        except ToolError as error:
            # P7: an unusable call is answered like a failed tool, so the model can fix it.
            run.consecutive_malformed += 1
            info = ToolErrorInfo.from_error(error)
            await self._answer_with_error(
                active, call.id, call.name, raw_args, None, info, ToolCallStatus.INVALID
            )
            await self._emit(
                active,
                EventType.TOOL_INVALID_CALL,
                tool=call.name,
                error_type=info.type,
                message=info.message,
            )
            return None
        run.consecutive_malformed = 0

        canonical_args = args.model_dump(mode="json")
        args_hash = _args_hash(spec.name, canonical_args)
        # P8: the same call with the same arguments again and again is a loop, not progress.
        if await self._repo.count_tool_calls(run.id, args_hash) >= config.repeat_call_limit:
            return await self._finish(
                active, TerminationReason.REPEATED_TOOL_CALL, tool=spec.name, args=canonical_args
            )

        if spec.requires_approval:
            # P3: a human's rejection is enforced in code, not only by the prompt. The
            # exact call they rejected is refused again without asking them twice.
            rejected = await self._repo.find_rejected_call(run.id, args_hash)
            if rejected is not None:
                await self._refuse_rejected_call(active, call, spec, canonical_args, rejected)
                return None
            # P3: the approval gate is enforced here, in code. Whatever the model
            # outputs, an approval-gated tool is only reached via _resolve_pending,
            # after a stored approval. P4: the approval is bound to this call id and
            # these validated arguments.
            self._charge_active_time(active)
            active.run, approval = await self._repo.pause_for_approval(
                run, tool_call_id=call.id, tool_name=spec.name, args=canonical_args
            )
            await self._emit(
                active,
                EventType.APPROVAL_REQUESTED,
                approval_id=approval.id,
                tool=spec.name,
                args=canonical_args,
            )
            return RunStatus.WAITING_APPROVAL

        await self._execute(active, spec, args, call.id, args_hash)
        return None

    async def _refuse_extra_call(self, active: _ActiveRun, call: LLMToolCall) -> None:
        """One tool call per step (§8.4): answer the others so every call id has a reply."""
        info = ToolErrorInfo(type="NotExecuted", message=NOT_EXECUTED_MESSAGE, retryable=True)
        await self._answer_with_error(
            active, call.id, call.name, None, None, info, ToolCallStatus.NOT_EXECUTED
        )
        await self._emit(active, EventType.TOOL_INVALID_CALL, tool=call.name, error_type=info.type)

    async def _refuse_rejected_call(
        self,
        active: _ActiveRun,
        call: LLMToolCall,
        spec: ToolSpec,
        canonical_args: dict[str, Any],
        rejected: ToolCallRecord,
    ) -> None:
        earlier = ToolErrorInfo.model_validate((rejected.result or {}).get("error", {}))
        info = ToolErrorInfo(
            type="ApprovalRejected",
            message=f"a human already rejected this exact call: {earlier.message}",
            retryable=False,
        )
        await self._answer_with_error(
            active,
            call.id,
            spec.name,
            canonical_args,
            rejected.args_hash,
            info,
            ToolCallStatus.REJECTED,
        )
        await self._emit(
            active,
            EventType.TOOL_INVALID_CALL,
            tool=spec.name,
            error_type=info.type,
            previously_rejected=True,
        )

    async def _resolve_pending(self, active: _ActiveRun) -> None:
        """Carry out the human's decision on the paused call (PLAN §9.3)."""
        run = active.run
        pending = run.pending_tool_call
        assert pending is not None
        approval = await self._repo.get_approval(pending.approval_id)
        if approval is None or approval.status is ApprovalStatus.PENDING:
            raise HarnessError(f"approval {pending.approval_id} has not been decided")
        spec = self._registry.get(pending.tool_name)
        args_hash = _args_hash(spec.name, pending.args)

        if approval.status is ApprovalStatus.APPROVED:
            # P4: execute exactly the stored arguments the human approved; the model
            # is not asked to produce them again.
            args = spec.input_model.model_validate(pending.args)
            # P6: a key the model never sees. If the call times out after the
            # incident was committed, the retry gets the same incident back.
            await self._execute(
                active,
                spec,
                args,
                pending.tool_call_id,
                args_hash,
                idempotency_key=f"{run.id}:{approval.id}",
            )
        else:
            info = ToolErrorInfo(
                type="ApprovalRejected",
                message=approval.reason or DEFAULT_REJECTION_REASON,
                retryable=False,
            )
            await self._answer_with_error(
                active,
                pending.tool_call_id,
                spec.name,
                pending.args,
                args_hash,
                info,
                ToolCallStatus.REJECTED,
            )
        run.pending_tool_call = None
        await self._save(active)

    async def _execute(
        self,
        active: _ActiveRun,
        spec: ToolSpec,
        args: BaseModel,
        tool_call_id: str,
        args_hash: str,
        *,
        idempotency_key: str | None = None,
    ) -> None:
        run = active.run
        context = ToolContext(
            run_id=run.id,
            step=run.step_count,
            idempotency_key=idempotency_key,
            deadline_remaining_s=self._remaining_s(active),
        )
        result = await self._executor.run(spec, args, context)
        payload = observations.result_payload(result)
        await self._repo.append_message(
            run.id,
            "tool",
            observations.render(payload, active.config.observation_max_chars),
            tool_call_id=tool_call_id,
        )
        await self._repo.record_tool_call(
            run_id=run.id,
            step=run.step_count,
            tool_call_id=tool_call_id,
            tool_name=spec.name,
            args=args.model_dump(mode="json"),
            args_hash=args_hash,
            status=ToolCallStatus.SUCCEEDED if result.ok else ToolCallStatus.FAILED,
            result=payload,
            error_type=result.error.type if result.error else None,
            attempts=result.attempts,
            duration_ms=result.duration_ms,
        )
        # P7: a failed tool is reported to the model; the run only ends if failures repeat.
        run.consecutive_tool_errors = 0 if result.ok else run.consecutive_tool_errors + 1

    async def _answer_with_error(
        self,
        active: _ActiveRun,
        tool_call_id: str,
        tool_name: str,
        args: dict[str, Any] | None,
        args_hash: str | None,
        info: ToolErrorInfo,
        status: ToolCallStatus,
    ) -> None:
        """Reply to a tool call that was not executed, and record why."""
        run = active.run
        payload = observations.error_payload(info)
        await self._repo.append_message(
            run.id,
            "tool",
            observations.render(payload, active.config.observation_max_chars),
            tool_call_id=tool_call_id,
        )
        await self._repo.record_tool_call(
            run_id=run.id,
            step=run.step_count,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            args=args,
            args_hash=args_hash,
            status=status,
            result=payload,
            error_type=info.type,
        )

    # ------------------------------------------------------------------ ending a run

    async def _finish(
        self, active: _ActiveRun, reason: TerminationReason, **details: Any
    ) -> RunStatus:
        run = active.run
        run.status = STATUS_FOR_REASON[reason]
        run.termination_reason = reason
        run.completed_at = self._clock.now_utc()
        await self._save(active)
        if run.status is RunStatus.LIMIT_EXCEEDED:
            await self._emit(
                active,
                EventType.LIMIT_EXCEEDED,
                reason=reason,
                step_count=run.step_count,
                active_elapsed_ms=run.active_elapsed_ms,
                **details,
            )
        await self._emit(
            active, EventType.RUN_FINISHED, status=run.status, reason=reason, **details
        )
        self._faults.clear_run(run.id)
        return run.status

    async def _fail_internal(self, run_id: str, error: Exception) -> RunStatus:
        self._log.error("run_internal_error", run_id=run_id, exc_info=error)
        return await self.fail_run(run_id, error_type=type(error).__name__, message=str(error))

    async def fail_run(self, run_id: str, *, error_type: str, message: str) -> RunStatus:
        """Mark an unfinished run FAILED with reason ``internal_error``.

        Used for unexpected exceptions and, by ``RunService.shutdown``, for runs whose
        advance() was interrupted by a shutdown (§12). A run that already finished
        is left alone.
        """
        self._faults.clear_run(run_id)
        run = await self._repo.get_run(run_id)
        if run is None or run.status in TERMINAL_STATUSES:
            return run.status if run else RunStatus.FAILED
        failed = run.model_copy(
            update={
                "status": RunStatus.FAILED,
                "termination_reason": TerminationReason.INTERNAL_ERROR,
                "completed_at": self._clock.now_utc(),
            }
        )
        try:
            await self._repo.save_run(failed)
        except RunAlreadyTerminalError:
            stored = await self._repo.get_run(run_id)
            return stored.status if stored else RunStatus.FAILED
        await self._events.emit(
            run_id,
            EventType.RUN_FINISHED,
            run.step_count,
            status=RunStatus.FAILED,
            reason=TerminationReason.INTERNAL_ERROR,
            error_type=error_type,
            message=message,
        )
        return RunStatus.FAILED

    # ------------------------------------------------------------------ helpers

    async def _save(self, active: _ActiveRun) -> None:
        self._charge_active_time(active)
        saved = await self._repo.save_run(active.run)
        active.run.updated_at = saved.updated_at

    def _charge_active_time(self, active: _ActiveRun) -> None:
        now = self._clock.monotonic()
        active.run.active_elapsed_ms += round((now - active.charged_until) * 1000)
        active.charged_until = now

    def _remaining_s(self, active: _ActiveRun) -> float:
        uncharged_s = self._clock.monotonic() - active.charged_until
        spent_s = active.run.active_elapsed_ms / 1000 + uncharged_s
        return active.config.max_run_seconds - spent_s

    async def _emit(
        self, active: _ActiveRun, event_type: EventType, step: int | None = None, **payload: Any
    ) -> None:
        run_step = active.run.step_count if step is None else step
        await self._events.emit(active.run.id, event_type, run_step, **payload)


def _parse_arguments(call: LLMToolCall) -> dict[str, Any]:
    try:
        parsed = json.loads(call.arguments_raw or "{}")
    except json.JSONDecodeError as exc:
        raise ToolArgumentsError(call.name, f"arguments are not valid JSON: {exc.msg}") from None
    if not isinstance(parsed, dict):
        raise ToolArgumentsError(call.name, "arguments must be a JSON object")
    return parsed


def _validate_arguments(spec: ToolSpec, raw_args: dict[str, Any]) -> BaseModel:
    try:
        return spec.input_model.model_validate(raw_args)
    except ValidationError as exc:
        raise ToolArgumentsError(
            spec.name, f"invalid arguments: {describe_validation_error(exc)}"
        ) from None


def _args_hash(tool_name: str, canonical_args: dict[str, Any]) -> str:
    """Identity of a call for the repeated-call limit: tool name + canonical JSON args."""
    canonical = json.dumps(canonical_args, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{tool_name}:{canonical}".encode()).hexdigest()
