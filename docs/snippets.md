# Core code excerpts

Three excerpts from the harness, copied from the source with small elisions (`# …`). Each is under 60 lines.

## 1. The loop tick — `src/harness/core/loop.py`, `AgentLoop._tick`

One step of a run: honour a cancel, carry out a decided approval, check limits, ask the model, act on its reply.
`advance()` calls this until it returns a status; nothing survives between calls except what is saved to the database.

```python
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
```

## 2. Dispatch and the approval gate — `AgentLoop._dispatch` and `_resolve_pending`

A validated call to an approval-gated tool never reaches its handler from dispatch: the run is parked on an approval bound to this call and its arguments.
When a human approves, `_resolve_pending` executes exactly those stored arguments, under an idempotency key the model never sees.

```python
# _dispatch, after the call's JSON, tool name and arguments have been validated:
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
        # P3: the approval gate is enforced here, in code. [...] P4: the approval
        # is bound to this call id and these validated arguments.
        self._charge_active_time(active)
        active.run, approval = await self._repo.pause_for_approval(
            run, tool_call_id=call.id, tool_name=spec.name, args=canonical_args
        )
        await self._emit(active, EventType.APPROVAL_REQUESTED, approval_id=approval.id,
                         tool=spec.name, args=canonical_args)
        return RunStatus.WAITING_APPROVAL

    await self._execute(active, spec, args, call.id, args_hash)
    return None

# _resolve_pending, on the next advance() after the decision:
    if approval.status is ApprovalStatus.APPROVED:
        # P4: execute exactly the stored arguments the human approved; the model
        # is not asked to produce them again.
        args = spec.input_model.model_validate(pending.args)
        # P6: a key the model never sees. If the call times out after the
        # incident was committed, the retry gets the same incident back.
        await self._execute(
            active, spec, args, pending.tool_call_id, args_hash,
            idempotency_key=f"{run.id}:{approval.id}",
        )
    else:
        # … answer the call with {"ok": false, "error": {"type": "ApprovalRejected", …}}
    run.pending_tool_call = None
    await self._save(active)
```

## 3. Executor retry — `src/harness/tools/executor.py`, `ToolExecutor.run`

The only place tool calls are retried: each attempt has a timeout capped by the run's remaining active time, and only retryable errors back off and try again.
Output that breaks the tool's schema, and permanent errors, fail at once; every outcome returns as a `ToolResult`, never as an exception.

```python
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
                    ctx.run_id, EventType.TOOL_ATTEMPT_FAILED, ctx.step,
                    # … tool, attempt, error_type, message, will_retry, delay_s
                )
                await self._clock.sleep(delay_s)
                continue
            result = ToolResult.failed(
                error, attempts=attempt, duration_ms=self._elapsed_ms(started)
            )
            await self._events.emit(ctx.run_id, EventType.TOOL_FAILED, ctx.step, …)
            return result

        result = ToolResult.succeeded(
            spec.name, data, attempts=attempt, duration_ms=self._elapsed_ms(started)
        )
        await self._events.emit(ctx.run_id, EventType.TOOL_SUCCEEDED, ctx.step, …)
        return result

# tools/spec.py — the backoff itself [R6]:
def backoff_delay(self, attempt: int, rng: random.Random) -> float:
    cap = min(self.max_delay_s, self.base_delay_s * 2 ** (attempt - 1))
    return rng.uniform(0, cap)
```
