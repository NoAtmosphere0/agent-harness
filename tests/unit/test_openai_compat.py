"""OpenAI-compatible client against a mock HTTP transport (no network).

``httpx2`` is the HTTP library the openai SDK itself depends on.
"""

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from harness.domain.errors import LLMProviderError, LLMTimeoutError
from harness.llm.base import ChatMessage, LLMToolCall
from harness.llm.openai_compat import OpenAICompatLLM

TOOLS = [{"type": "function", "function": {"name": "get_service_status", "parameters": {}}}]


def _completion(message: dict[str, Any], usage: bool = True) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "qwen2.5:7b",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }
    if usage:
        body["usage"] = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    return body


class _Server:
    """Records requests and answers with a fixed handler."""

    def __init__(self, respond: Callable[[httpx2.Request], httpx2.Response]) -> None:
        self.requests: list[dict[str, Any]] = []
        self._respond = respond

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        return self._respond(request)


def _llm(server: _Server, **kwargs: Any) -> OpenAICompatLLM:
    return OpenAICompatLLM(
        base_url="http://llm.test/v1",
        api_key="test-key",
        model="qwen2.5:7b",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(server)),
        **kwargs,
    )


def _ok(message: dict[str, Any]) -> _Server:
    return _Server(lambda request: httpx2.Response(200, json=_completion(message)))


MESSAGES = [
    ChatMessage(role="system", content="sys"),
    ChatMessage(role="user", content="check auth-service"),
    ChatMessage(
        role="assistant",
        tool_calls=[
            LLMToolCall(id="c1", name="get_service_status", arguments_raw='{"service_name": "a"}')
        ],
    ),
    ChatMessage(role="tool", content='{"ok": true}', tool_call_id="c1"),
]


async def test_openai_compat_sends_messages_tools_and_single_call_flag():
    server = _ok({"role": "assistant", "content": "done"})

    await _llm(server).complete(MESSAGES, TOOLS, timeout_s=5)

    (body,) = server.requests
    assert body["model"] == "qwen2.5:7b"
    assert body["tools"] == TOOLS
    assert body["parallel_tool_calls"] is False
    assert body["messages"][2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "get_service_status", "arguments": '{"service_name": "a"}'},
            }
        ],
    }
    assert body["messages"][3] == {"role": "tool", "content": '{"ok": true}', "tool_call_id": "c1"}


async def test_openai_compat_omits_parallel_flag_when_disabled_or_no_tools():
    server = _ok({"role": "assistant", "content": "done"})

    await _llm(server, send_parallel_tool_calls_flag=False).complete(MESSAGES, TOOLS, 5)
    await _llm(server).complete(MESSAGES, [], 5)

    assert "parallel_tool_calls" not in server.requests[0]
    assert "tools" not in server.requests[1]
    assert "parallel_tool_calls" not in server.requests[1]


async def test_openai_compat_parses_text_tool_calls_and_usage():
    server = _ok(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_9",
                    "type": "function",
                    "function": {
                        "name": "get_service_status",
                        "arguments": '{"service_name": "x"}',
                    },
                },
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "search_knowledge_base", "arguments": "{}"},
                },
            ],
        }
    )

    response = await _llm(server).complete(MESSAGES, TOOLS, 5)

    assert response.text is None
    assert [(c.id, c.name, c.arguments_raw) for c in response.tool_calls] == [
        ("call_9", "get_service_status", '{"service_name": "x"}'),
        ("call_1", "search_knowledge_base", "{}"),  # missing id is generated
    ]
    assert response.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


async def test_openai_compat_empty_choices_become_empty_response():
    body = _completion({"role": "assistant", "content": "x"}, usage=False) | {"choices": []}
    server = _Server(lambda request: httpx2.Response(200, json=body))

    response = await _llm(server).complete(MESSAGES, TOOLS, 5)

    assert (response.text, response.tool_calls, response.usage) == (None, [], None)


@pytest.mark.parametrize(
    ("status", "retryable"),
    [
        (429, True),
        (500, True),
        (503, True),
        (408, True),
        (400, False),
        (401, False),
        (403, False),
        (404, False),
    ],
)
async def test_openai_compat_maps_http_status_to_retryability(status: int, retryable: bool):
    server = _Server(lambda request: httpx2.Response(status, json={"error": {"message": "boom"}}))

    with pytest.raises(LLMProviderError) as exc_info:
        await _llm(server).complete(MESSAGES, TOOLS, 5)

    assert exc_info.value.retryable is retryable
    assert exc_info.value.message.startswith(f"HTTP {status}")
    assert len(server.requests) == 1  # P5: the SDK itself never retries


async def test_openai_compat_connection_error_is_retryable():
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    with pytest.raises(LLMProviderError) as exc_info:
        await _llm(_Server(refuse)).complete(MESSAGES, TOOLS, 5)

    assert exc_info.value.retryable is True


async def test_openai_compat_timeout_maps_to_llm_timeout():
    def slow(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timed out", request=request)

    with pytest.raises(LLMTimeoutError, match="no response within 5s"):
        await _llm(_Server(slow)).complete(MESSAGES, TOOLS, 5)
