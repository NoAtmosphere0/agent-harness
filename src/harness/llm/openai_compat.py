"""``LLMClient`` for any OpenAI-compatible chat completions endpoint.

Works with OpenAI, Ollama (``http://localhost:11434/v1``), vLLM and similar
servers, selected with ``LLM_BASE_URL``. Its only jobs are translating messages
and mapping provider failures onto the harness error taxonomy (PLAN §6.4):

* timeouts, connection errors, 408, 429 and 5xx → retryable;
* other 4xx (400 bad request, 401/403 auth, 404 unknown model, ...) → not retryable.

The loop decides what to do with them. P5: the SDK client is created with
``max_retries=0`` so the loop stays the only LLM retry layer.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import openai

from harness.domain.errors import HarnessError, LLMProviderError, LLMTimeoutError
from harness.llm.base import ChatMessage, LLMResponse, LLMToolCall

if TYPE_CHECKING:
    import httpx2
    from openai.types.chat import ChatCompletion

    from harness.config import Settings

_MAX_ERROR_CHARS = 300


class OpenAICompatLLM:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        send_parallel_tool_calls_flag: bool = True,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self.model_name = model
        self._send_parallel_flag = send_parallel_tool_calls_flag
        self._client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,  # P5: retries happen in the loop, and only there
            http_client=http_client,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAICompatLLM:
        api_key = settings.llm_api_key.get_secret_value()
        if not api_key:
            raise HarnessError("LLM_API_KEY is required for openai_compat (any value for Ollama)")
        return cls(
            base_url=settings.llm_base_url,
            api_key=api_key,
            model=settings.llm_model,
            send_parallel_tool_calls_flag=settings.llm_send_parallel_tool_calls_flag,
        )

    async def complete(
        self, messages: list[ChatMessage], tools: list[dict[str, Any]], timeout_s: float
    ) -> LLMResponse:
        options: dict[str, Any] = {}
        if tools:
            options["tools"] = tools
            if self._send_parallel_flag:
                # One tool call per step (§8.4); providers that ignore this are
                # handled by answering the extra calls "not executed".
                options["parallel_tool_calls"] = False
        try:
            completion = await self._client.chat.completions.create(
                model=self.model_name,
                messages=cast(Any, [_to_openai(m) for m in messages]),
                timeout=timeout_s,
                **options,
            )
        except openai.APITimeoutError as exc:  # subclass of APIConnectionError: check first
            raise LLMTimeoutError(f"no response within {timeout_s:g}s") from exc
        except openai.APIConnectionError as exc:
            raise LLMProviderError(f"connection error: {exc}", retryable=True) from exc
        except openai.APIStatusError as exc:
            detail = str(exc.message)[:_MAX_ERROR_CHARS]
            raise LLMProviderError(
                f"HTTP {exc.status_code}: {detail}",
                retryable=_is_retryable_status(exc.status_code),
            ) from exc
        return _from_completion(completion)


def _is_retryable_status(status_code: int) -> bool:
    return status_code in (408, 429) or status_code >= 500


def _to_openai(message: ChatMessage) -> dict[str, Any]:
    out: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.role == "tool":
        out["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        out["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_raw},
            }
            for call in message.tool_calls
        ]
    return out


def _from_completion(completion: ChatCompletion) -> LLMResponse:
    usage = None
    if completion.usage is not None:
        usage = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens,
        }
    if not completion.choices:
        # Nothing usable: the parser treats this as an empty reply and repairs it.
        return LLMResponse(usage=usage)
    message = completion.choices[0].message
    calls = []
    for index, tool_call in enumerate(message.tool_calls or []):
        function = getattr(tool_call, "function", None)
        if function is None:  # non-function tool types are not used by the harness
            continue
        arguments = function.arguments
        calls.append(
            LLMToolCall(
                # Some compatible servers omit ids; every call needs one for its reply.
                id=tool_call.id or f"call_{index}",
                name=function.name,
                arguments_raw=arguments if isinstance(arguments, str) else json.dumps(arguments),
            )
        )
    return LLMResponse(text=message.content, tool_calls=calls, usage=usage)
