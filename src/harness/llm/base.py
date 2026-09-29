"""Provider-neutral LLM types (PLAN §6.2).

The loop talks only to ``LLMClient``; OpenAI-compatible APIs, Ollama and the
scripted test client all sit behind it. Tool-call arguments are kept as the raw
provider string and parsed during dispatch, where a malformed string becomes an
observation for the model rather than an exception here.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel

from harness.domain.models import MessageRecord, MessageRole


class LLMToolCall(BaseModel):
    id: str
    name: str
    arguments_raw: str  # raw provider string, parsed during dispatch


class LLMResponse(BaseModel):
    text: str | None = None
    tool_calls: list[LLMToolCall] = []
    usage: dict[str, int] | None = None


class ChatMessage(BaseModel):
    """One message of the conversation sent to the model."""

    role: MessageRole
    content: str | None = None
    tool_calls: list[LLMToolCall] | None = None
    tool_call_id: str | None = None

    @classmethod
    def from_record(cls, record: MessageRecord) -> ChatMessage:
        return cls(
            role=record.role,
            content=record.content,
            tool_calls=[LLMToolCall.model_validate(c) for c in record.tool_calls]
            if record.tool_calls
            else None,
            tool_call_id=record.tool_call_id,
        )


class LLMClient(Protocol):
    """A chat-completion endpoint with tool calling.

    Implementations raise ``LLMTimeoutError`` / ``LLMProviderError`` and never retry
    internally: the loop is the only LLM retry layer (P5).
    """

    model_name: str

    async def complete(
        self, messages: list[ChatMessage], tools: list[dict[str, Any]], timeout_s: float
    ) -> LLMResponse: ...
