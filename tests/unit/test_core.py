"""Parser, guardrails and observations: the pure pieces of the loop."""

import json
from datetime import UTC, datetime

import pytest

from harness.config import RunConfig
from harness.core import guardrails, observations
from harness.core.parser import EmptyResponse, FinalAnswer, ToolCallAction, parse
from harness.domain.errors import ToolTransientError
from harness.domain.models import RunRecord, RunStatus, TerminationReason, ToolResult
from harness.llm.base import LLMResponse
from harness.llm.scripted import call

# ------------------------------------------------------------------ parser


def test_parser_text_without_calls_is_final_answer():
    assert parse(LLMResponse(text="  All good.  ")) == FinalAnswer(text="All good.")


def test_parser_single_call():
    c = call("get_service_status", {"service_name": "auth-service"})

    assert parse(LLMResponse(tool_calls=[c])) == ToolCallAction(call=c)


def test_parser_multiple_calls_keeps_first_and_extras():
    a, b, c = (call("t", {}) for _ in range(3))

    action = parse(LLMResponse(text="thinking", tool_calls=[a, b, c]))

    assert action == ToolCallAction(call=a, extra_calls=(b, c))


@pytest.mark.parametrize("text", [None, "", "   \n"])
def test_parser_empty_response(text: str | None):
    assert parse(LLMResponse(text=text)) == EmptyResponse()


# ------------------------------------------------------------------ guardrails

_CONFIG = RunConfig(
    provider="scripted",
    model="scripted",
    max_steps=5,
    max_run_seconds=10,
    max_parse_retries=2,
    max_consecutive_tool_errors=3,
    repeat_call_limit=3,
    llm_timeout_seconds=60,
    llm_max_retries=2,
    observation_max_chars=4000,
    faults={},
)


def _run(step_count: int = 0, active_elapsed_ms: int = 0) -> RunRecord:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return RunRecord(
        id="r",
        objective="o",
        status=RunStatus.RUNNING,
        config={},
        created_at=now,
        updated_at=now,
        step_count=step_count,
        active_elapsed_ms=active_elapsed_ms,
    )


def test_guardrails_allow_run_within_limits():
    assert guardrails.check(_run(step_count=4, active_elapsed_ms=9_999), _CONFIG) is None


def test_guardrails_stop_at_max_steps():
    assert guardrails.check(_run(step_count=5), _CONFIG) is TerminationReason.MAX_STEPS


def test_guardrails_stop_at_max_run_time():
    assert (
        guardrails.check(_run(active_elapsed_ms=10_000), _CONFIG) is TerminationReason.MAX_RUN_TIME
    )


# ------------------------------------------------------------------ observations


def test_observation_success_envelope():
    result = ToolResult.succeeded("t", {"status": "operational"}, attempts=1, duration_ms=3)

    text = observations.render(observations.result_payload(result), 4000)

    assert json.loads(text) == {"ok": True, "data": {"status": "operational"}}


def test_observation_error_envelope():
    result = ToolResult.failed(ToolTransientError("t", "HTTP 503"), attempts=3, duration_ms=9)

    payload = observations.result_payload(result)

    assert payload == {
        "ok": False,
        "error": {"type": "ToolTransientError", "message": "HTTP 503", "retryable": True},
    }


def test_observation_truncated_with_marker():
    payload = {"ok": True, "data": {"text": "x" * 200}}
    full = json.dumps(payload)

    text = observations.render(payload, 100)

    assert text == f"{full[:100]}[truncated {len(full) - 100} chars]"


def test_observation_at_limit_not_truncated():
    payload = {"ok": True, "data": {}}
    full = json.dumps(payload)

    assert observations.render(payload, len(full)) == full
