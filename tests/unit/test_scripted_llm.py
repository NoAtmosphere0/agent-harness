import pytest

from harness.domain.errors import LLMProviderError
from harness.llm.base import ChatMessage, LLMResponse
from harness.llm.scripted import (
    ReplayLLM,
    ScriptedLLM,
    ScriptExhaustedError,
    call,
    checkout_outage_script,
    final,
    last_tool_result,
)

_MESSAGES = [ChatMessage(role="user", content="hi")]


async def test_scripted_llm_replays_responses_and_records_requests():
    llm = ScriptedLLM([final("one"), final("two")])

    first = await llm.complete(_MESSAGES, [{"tool": 1}], timeout_s=5)
    second = await llm.complete(_MESSAGES, [], timeout_s=5)

    assert (first.text, second.text) == ("one", "two")
    assert [r.tools for r in llm.requests] == [[{"tool": 1}], []]
    assert llm.requests[0].messages == _MESSAGES
    assert llm.remaining_steps == 0


async def test_scripted_llm_raises_scripted_exception():
    llm = ScriptedLLM([LLMProviderError("503", retryable=True)])

    with pytest.raises(LLMProviderError, match="503"):
        await llm.complete(_MESSAGES, [], timeout_s=5)


async def test_scripted_llm_side_effect_step_then_next_step():
    seen: list[int] = []
    llm = ScriptedLLM([lambda messages: seen.append(len(messages)), final("after")])

    response = await llm.complete(_MESSAGES, [], timeout_s=5)

    assert response.text == "after"
    assert seen == [1]


async def test_scripted_llm_async_callable_step():
    async def dynamic(messages: list[ChatMessage]) -> LLMResponse:
        return final(f"{len(messages)} messages")

    llm = ScriptedLLM([dynamic])

    assert (await llm.complete(_MESSAGES, [], timeout_s=5)).text == "1 messages"


async def test_scripted_llm_fails_loudly_when_exhausted():
    llm = ScriptedLLM([])

    with pytest.raises(ScriptExhaustedError, match="request #1"):
        await llm.complete(_MESSAGES, [], timeout_s=5)


def test_call_builder_keeps_raw_string_arguments():
    assert call("t", "{broken").arguments_raw == "{broken"
    assert call("t", {"a": 1}, call_id="c1").model_dump() == {
        "id": "c1",
        "name": "t",
        "arguments_raw": '{"a": 1}',
    }


def test_last_tool_result_requires_a_tool_message():
    with pytest.raises(ValueError, match="no tool message"):
        last_tool_result(_MESSAGES)


async def test_checkout_script_final_answer_reports_incident_id():
    final_step = checkout_outage_script()[-1]
    assert callable(final_step)
    tool = ChatMessage(
        role="tool",
        content='{"ok": true, "data": {"incident_id": "INC-000007"}}',
        tool_call_id="c",
    )

    response = final_step([*_MESSAGES, tool])

    assert isinstance(response, LLMResponse)
    assert "opened INC-000007" in (response.text or "")


# ------------------------------------------------------------------ ReplayLLM


def _assistant(n: int) -> list[ChatMessage]:
    return [ChatMessage(role="assistant", content=f"a{i}") for i in range(n)]


async def test_replay_llm_picks_step_by_assistant_turns():
    async def third(messages: list[ChatMessage]) -> LLMResponse:
        return final("three")

    llm = ReplayLLM([final("one"), lambda messages: final("two"), third])

    answers = [(await llm.complete(_assistant(n), [], timeout_s=5)).text for n in (0, 1, 2)]

    assert answers == ["one", "two", "three"]
    # Stateless: asking again for turn 0 replays turn 0.
    assert (await llm.complete([], [], timeout_s=5)).text == "one"


async def test_replay_llm_finishes_after_script_ends():
    llm = ReplayLLM([final("only")])

    response = await llm.complete(_assistant(3), [], timeout_s=5)

    assert response.text == "The scripted demo has no further steps."


async def test_replay_llm_raises_scripted_exception_and_rejects_empty_step():
    llm = ReplayLLM([LLMProviderError("503", retryable=True), lambda messages: None])

    with pytest.raises(LLMProviderError):
        await llm.complete([], [], timeout_s=5)
    with pytest.raises(ScriptExhaustedError, match="no response"):
        await llm.complete(_assistant(1), [], timeout_s=5)
