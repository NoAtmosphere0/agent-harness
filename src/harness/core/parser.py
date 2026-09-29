"""Turn an LLM response into one of three actions (PLAN §6.3).

Only the response *shape* is judged here. Whether a tool call is usable (valid
JSON, known tool, valid arguments) is decided in dispatch, so every tool call id
still gets a tool message back, as the chat APIs require.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from harness.llm.base import LLMResponse, LLMToolCall


class FinalAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str


class ToolCallAction(BaseModel):
    """The first tool call is executed; the rest are answered "not executed" (§8.4)."""

    model_config = ConfigDict(frozen=True)

    call: LLMToolCall
    extra_calls: tuple[LLMToolCall, ...] = ()


class EmptyResponse(BaseModel):
    model_config = ConfigDict(frozen=True)


Action = FinalAnswer | ToolCallAction | EmptyResponse


def parse(response: LLMResponse) -> Action:
    if response.tool_calls:
        first, *rest = response.tool_calls
        return ToolCallAction(call=first, extra_calls=tuple(rest))
    if response.text and response.text.strip():
        # A final answer is text with no tool calls (same rule as [R2]).
        return FinalAnswer(text=response.text.strip())
    return EmptyResponse()
