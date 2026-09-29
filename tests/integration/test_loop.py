"""Loop + SQLite + ScriptedLLM: success, tool failure, malformed LLM output, limits,
control (PLAN §16). Approval flows are in test_approval.py."""

import asyncio

import pytest

from harness.domain.errors import (
    InvalidConfigOverridesError,
    LLMProviderError,
    LLMTimeoutError,
    RunNotFoundError,
)
from harness.domain.models import (
    EventType,
    RunStatus,
    TerminationReason,
    ToolCallStatus,
)
from harness.llm.base import ChatMessage, LLMResponse
from harness.llm.prompts import EMPTY_RESPONSE_REPAIR
from harness.llm.scripted import call, empty, final, tool_calls
from harness.tools.spec import ToolContext
from tests.fixtures.scripts import kb_and_status_script
from tests.integration.conftest import EnvFactory

E = EventType

# ------------------------------------------------------------------ success


async def test_run_completes_with_kb_and_status_calls(make_env: EnvFactory):
    env = make_env(kb_and_status_script())
    run_id = await env.start("Is auth-service healthy?")

    run = await env.advance(run_id)

    assert run.status is RunStatus.COMPLETED
    assert run.termination_reason is TerminationReason.FINAL_ANSWER
    assert run.final_answer is not None
    assert run.final_answer.startswith("auth-service is operational")
    assert run.step_count == 3
    assert run.completed_at is not None
    assert [m.role for m in await env.messages(run_id)] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    calls = await env.tool_calls(run_id)
    assert [(c.tool_name, c.status, c.step) for c in calls] == [
        ("search_knowledge_base", ToolCallStatus.SUCCEEDED, 1),
        ("get_service_status", ToolCallStatus.SUCCEEDED, 2),
    ]
    events = await env.events(run_id)
    assert [e.type for e in events] == [
        E.RUN_CREATED,
        E.RUN_STARTED,
        E.LLM_RESPONSE,
        E.TOOL_SUCCEEDED,
        E.LLM_RESPONSE,
        E.TOOL_SUCCEEDED,
        E.LLM_RESPONSE,
        E.RUN_FINISHED,
    ]
    assert [e.seq for e in events] == list(range(1, 9))
    assert events[-1].payload == {"status": "completed", "reason": "final_answer"}
    # The model saw the tool schemas and, on its second call, the first observation.
    assert len(env.llm.requests[0].tools) == 3
    assert env.llm.requests[1].messages[-1].role == "tool"


async def test_final_answer_without_tools(make_env: EnvFactory):
    env = make_env([final("Nothing to investigate.")])
    run_id = await env.start("Say hello")

    run = await env.advance(run_id)

    assert run.status is RunStatus.COMPLETED
    assert run.final_answer == "Nothing to investigate."
    assert run.step_count == 1
    assert await env.tool_calls(run_id) == []


async def test_advance_on_finished_run_is_noop(make_env: EnvFactory):
    env = make_env([final("done")])
    run_id = await env.start()
    await env.advance(run_id)
    before = await env.event_types(run_id)

    assert await env.runs.advance(run_id) is RunStatus.COMPLETED
    assert await env.event_types(run_id) == before


# ------------------------------------------------------------------ tool failure


async def test_flaky_tool_recovers_after_retries(make_env: EnvFactory):
    env = make_env(
        [
            tool_calls(call("get_service_status", {"service_name": "auth-service"})),
            final("auth-service is operational."),
        ],
        mock_faults={"get_service_status": {"mode": "flaky", "fail_times": 2}},
    )
    run_id = await env.start()

    run = await env.advance(run_id)

    assert run.status is RunStatus.COMPLETED
    (record,) = await env.tool_calls(run_id)
    assert record.status is ToolCallStatus.SUCCEEDED
    assert record.attempts == 3
    assert len(env.clock.sleeps) == 2
    assert (await env.event_types(run_id)).count(E.TOOL_ATTEMPT_FAILED) == 2
    assert (await env.observations(run_id))[0]["ok"] is True


async def test_permanent_error_returned_to_llm_and_run_completes(make_env: EnvFactory):
    env = make_env(
        [
            tool_calls(call("get_service_status", {"service_name": "billing-service"})),
            final("billing-service is not a known service."),
        ]
    )
    run_id = await env.start("Check billing-service")

    run = await env.advance(run_id)

    assert run.status is RunStatus.COMPLETED
    (record,) = await env.tool_calls(run_id)
    assert (record.status, record.attempts) == (ToolCallStatus.FAILED, 1)
    seen = env.llm.requests[1].messages[-1]
    assert seen.role == "tool"
    assert '"type": "ToolPermanentError"' in (seen.content or "")
    assert "known services: auth-service" in (seen.content or "")


async def test_tool_timeout_reported_to_llm(make_env: EnvFactory):
    env = make_env(
        [
            tool_calls(call("get_service_status", {"service_name": "auth-service"})),
            final("Status service is not responding."),
        ],
        mock_faults={"get_service_status": {"mode": "slow", "delay_s": 0.5}},
        tool_timeouts={"get_service_status": 0.02},
    )
    run_id = await env.start()

    run = await env.advance(run_id)

    assert run.status is RunStatus.COMPLETED
    (observation,) = await env.observations(run_id)
    assert observation["ok"] is False
    assert observation["error"]["type"] == "ToolTimeoutError"
    assert observation["error"]["retryable"] is True
    (record,) = await env.tool_calls(run_id)
    assert record.attempts == 3


async def test_consecutive_tool_errors_fail_run(make_env: EnvFactory):
    services = ["auth-service", "checkout-api", "search-api"]
    env = make_env(
        [tool_calls(call("get_service_status", {"service_name": s})) for s in services],
        mock_faults={"get_service_status": {"mode": "error"}},
    )
    run_id = await env.start()

    run = await env.advance(run_id)

    assert run.status is RunStatus.FAILED
    assert run.termination_reason is TerminationReason.TOO_MANY_TOOL_ERRORS
    assert run.consecutive_tool_errors == 3
    assert len(env.llm.requests) == 3


# ------------------------------------------------------------------ malformed LLM output


async def test_invalid_json_arguments_then_recovery(make_env: EnvFactory):
    env = make_env(
        [
            tool_calls(call("get_service_status", "{not json")),
            tool_calls(call("get_service_status", {"service_name": "auth-service"})),
            final("auth-service is operational."),
        ]
    )
    run_id = await env.start()

    run = await env.advance(run_id)

    assert run.status is RunStatus.COMPLETED
    assert run.consecutive_malformed == 0
    first, second = await env.observations(run_id)
    assert first["error"]["type"] == "ToolArgumentsError"
    assert "not valid JSON" in first["error"]["message"]
    assert second["ok"] is True
    assert [c.status for c in await env.tool_calls(run_id)] == [
        ToolCallStatus.INVALID,
        ToolCallStatus.SUCCEEDED,
    ]
    assert E.TOOL_INVALID_CALL in await env.event_types(run_id)


async def test_schema_violation_reports_field_errors(make_env: EnvFactory):
    env = make_env(
        [
            tool_calls(call("get_service_status", {"service_name": "Checkout API"})),
            final("Could not check."),
        ]
    )
    run_id = await env.start()

    await env.advance(run_id)

    (observation,) = await env.observations(run_id)
    assert observation["error"]["type"] == "ToolArgumentsError"
    assert "service_name: String should match pattern" in observation["error"]["message"]


async def test_unknown_tool_then_recovery(make_env: EnvFactory):
    env = make_env(
        [
            tool_calls(call("restart_service", {"service_name": "checkout-api"})),
            final("I cannot restart services; please do it manually."),
        ]
    )
    run_id = await env.start()

    run = await env.advance(run_id)

    assert run.status is RunStatus.COMPLETED
    (observation,) = await env.observations(run_id)
    assert observation["error"]["type"] == "ToolNotFoundError"
    assert "available tools: create_incident" in observation["error"]["message"]


async def test_empty_response_then_recovery(make_env: EnvFactory):
    env = make_env([empty(), final("done")])
    run_id = await env.start()

    run = await env.advance(run_id)

    assert run.status is RunStatus.COMPLETED
    assert env.llm.requests[1].messages[-1] == ChatMessage(
        role="user", content=EMPTY_RESPONSE_REPAIR
    )


async def test_repeated_empty_responses_fail_run(make_env: EnvFactory):
    env = make_env([empty(), LLMResponse(text="   "), empty()])
    run_id = await env.start()

    run = await env.advance(run_id)

    assert run.status is RunStatus.FAILED
    assert run.termination_reason is TerminationReason.MALFORMED_LLM_OUTPUT
    assert (await env.event_types(run_id)).count(E.LLM_MALFORMED_OUTPUT) == 3


async def test_llm_provider_errors_then_llm_unavailable(make_env: EnvFactory):
    env = make_env([LLMProviderError("HTTP 503", retryable=True) for _ in range(3)])
    run_id = await env.start()

    run = await env.advance(run_id)

    assert run.status is RunStatus.FAILED
    assert run.termination_reason is TerminationReason.LLM_UNAVAILABLE
    assert len(env.llm.requests) == 3
    assert len(env.clock.sleeps) == 2
    events = await env.events(run_id)
    assert [e.type for e in events].count(E.LLM_RETRY) == 2
    assert events[-1].payload["message"] == "HTTP 503"


async def test_llm_non_retryable_error_rejects_run_without_retry(make_env: EnvFactory):
    env = make_env([LLMProviderError("HTTP 401: invalid API key", retryable=False)])
    run_id = await env.start()

    run = await env.advance(run_id)

    assert run.termination_reason is TerminationReason.LLM_REQUEST_REJECTED
    assert env.clock.sleeps == []


async def test_llm_timeout_is_retried(make_env: EnvFactory):
    async def hangs(messages: list[ChatMessage]) -> LLMResponse:
        await asyncio.sleep(1)
        return final("too late")

    env = make_env([LLMTimeoutError("slow"), hangs, final("done")], llm_timeout_seconds=0.02)
    run_id = await env.start()

    run = await env.advance(run_id)

    assert run.status is RunStatus.COMPLETED
    retries = [e for e in await env.events(run_id) if e.type is E.LLM_RETRY]
    assert [r.payload["error_type"] for r in retries] == ["LLMTimeoutError", "LLMTimeoutError"]
    assert "no response within 0.02s" in retries[1].payload["message"]


async def test_unexpected_exception_fails_run_with_internal_error(make_env: EnvFactory):
    env = make_env([RuntimeError("bug")])
    run_id = await env.start()

    run = await env.advance(run_id)

    assert run.status is RunStatus.FAILED
    assert run.termination_reason is TerminationReason.INTERNAL_ERROR
    finished = (await env.events(run_id))[-1]
    assert finished.type is E.RUN_FINISHED
    assert finished.payload["error_type"] == "RuntimeError"


# ------------------------------------------------------------------ limits


async def test_max_steps_exceeded(make_env: EnvFactory):
    env = make_env(
        [
            tool_calls(call("get_service_status", {"service_name": s}))
            for s in ["auth-service", "checkout-api", "search-api"]
        ]
    )
    run_id = await env.start(max_steps=2)

    run = await env.advance(run_id)

    assert run.status is RunStatus.LIMIT_EXCEEDED
    assert run.termination_reason is TerminationReason.MAX_STEPS
    assert len(env.llm.requests) == 2
    assert (await env.event_types(run_id))[-2:] == [E.LIMIT_EXCEEDED, E.RUN_FINISHED]


async def test_max_run_time_exceeded(make_env: EnvFactory):
    def slow_thinking(messages: list[ChatMessage]) -> None:
        env.clock.advance(130)

    env = make_env(
        [slow_thinking, tool_calls(call("get_service_status", {"service_name": "auth-service"}))]
    )
    run_id = await env.start()

    run = await env.advance(run_id)

    assert run.status is RunStatus.LIMIT_EXCEEDED
    assert run.termination_reason is TerminationReason.MAX_RUN_TIME
    assert run.active_elapsed_ms >= 120_000
    assert len(env.llm.requests) == 1


async def test_repeated_identical_call_terminates(make_env: EnvFactory):
    same = {"service_name": "auth-service"}
    env = make_env([tool_calls(call("get_service_status", same)) for _ in range(4)])
    run_id = await env.start()

    run = await env.advance(run_id)

    assert run.status is RunStatus.LIMIT_EXCEEDED
    assert run.termination_reason is TerminationReason.REPEATED_TOOL_CALL
    assert len(await env.tool_calls(run_id)) == 3
    limit_event = next(e for e in await env.events(run_id) if e.type is E.LIMIT_EXCEEDED)
    assert limit_event.payload["args"] == same


async def test_per_run_override_cannot_raise_operator_limit(make_env: EnvFactory):
    env = make_env([final("ok")])

    with pytest.raises(InvalidConfigOverridesError):
        await env.start(max_steps=13)
    assert await env.repo.list_runs() == []


# ------------------------------------------------------------------ control


async def test_cancel_stops_run_at_next_tick(make_env: EnvFactory):
    async def cancel_mid_tick(messages: list[ChatMessage]) -> None:
        await env.runs.cancel(run_id)

    env = make_env(
        [
            cancel_mid_tick,
            tool_calls(call("get_service_status", {"service_name": "auth-service"})),
            final("never reached"),
        ]
    )
    run_id = await env.start()

    status = await env.runs.advance(run_id)

    assert status is RunStatus.CANCELLED
    run = await env.get(run_id)
    assert run.termination_reason is TerminationReason.CANCELLED_BY_USER
    assert len(env.llm.requests) == 1
    assert env.llm.remaining_steps == 1
    finished = [e for e in await env.events(run_id) if e.type is E.RUN_FINISHED]
    assert [e.payload["reason"] for e in finished] == ["cancelled_by_user"]


async def test_cancel_pending_run_before_it_starts(make_env: EnvFactory):
    env = make_env([final("never reached")])
    run_id = await env.start()
    await env.runs.cancel(run_id)

    assert await env.runs.advance(run_id) is RunStatus.CANCELLED
    assert env.llm.requests == []


async def test_extra_tool_calls_not_executed(make_env: EnvFactory):
    env = make_env(
        [
            tool_calls(
                call("get_service_status", {"service_name": "auth-service"}),
                call("get_service_status", {"service_name": "checkout-api"}),
            ),
            final("auth-service is operational."),
        ]
    )
    run_id = await env.start()

    run = await env.advance(run_id)

    assert run.status is RunStatus.COMPLETED
    calls = await env.tool_calls(run_id)
    assert [(c.status, c.error_type) for c in calls] == [
        (ToolCallStatus.NOT_EXECUTED, "NotExecuted"),
        (ToolCallStatus.SUCCEEDED, None),
    ]
    # Every tool call id in the assistant message got exactly one tool reply.
    messages = await env.messages(run_id)
    assistant = next(m for m in messages if m.tool_calls)
    replied = [m.tool_call_id or "" for m in messages if m.role == "tool"]
    assert sorted(replied) == sorted(c["id"] for c in assistant.tool_calls or [])
    assert (await env.event_types(run_id)).count(E.TOOL_SUCCEEDED) == 1


async def test_faults_cleared_when_run_finishes(make_env: EnvFactory):
    env = make_env([final("done")], mock_faults={"get_service_status": {"mode": "error"}})
    run_id = await env.start()
    await env.advance(run_id)
    handler = env.faults.wrap(env.registry.get("get_service_status"))
    args = env.registry.get("get_service_status").input_model(service_name="auth-service")

    out = await handler(args, ToolContext(run_id=run_id, step=1, deadline_remaining_s=5))

    assert out["status"] == "operational"


async def test_llm_backoff_that_exhausts_run_time_stops_run(make_env: EnvFactory):
    env = make_env([LLMProviderError("HTTP 503", retryable=True) for _ in range(3)])
    run_id = await env.start(max_run_seconds=0.1)

    run = await env.advance(run_id)

    assert run.termination_reason is TerminationReason.MAX_RUN_TIME
    assert len(env.llm.requests) == 1  # the backoff sleep alone used up the budget


async def test_non_object_json_arguments_are_rejected(make_env: EnvFactory):
    env = make_env([tool_calls(call("get_service_status", "[1, 2]")), final("ok")])
    run_id = await env.start()

    await env.advance(run_id)

    (observation,) = await env.observations(run_id)
    assert observation["error"]["message"] == "arguments must be a JSON object"


async def test_advance_unknown_run_raises(make_env: EnvFactory):
    env = make_env([])

    with pytest.raises(RunNotFoundError):
        await env.runs.advance("no-such-run")
