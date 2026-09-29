"""A deterministic ``LLMClient`` that replays a script (PLAN §16).

Used by every test and by ``harness demo``, so the whole harness can be exercised
without an API key. A script is a list of steps:

* an ``LLMResponse`` to return;
* an ``Exception`` to raise (e.g. ``LLMProviderError`` to exercise retries);
* a callable ``(messages) -> LLMResponse | None`` (sync or async). Returning a
  response answers the request; returning ``None`` means "side effect only" (e.g.
  advance a fake clock) and the next step is consumed for the same request.

Running out of steps raises ``ScriptExhaustedError``: a test that makes more LLM
calls than it scripted fails loudly instead of hanging or passing by accident.
"""

from __future__ import annotations

import inspect
import itertools
import json
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from harness.domain.errors import HarnessError
from harness.llm.base import ChatMessage, LLMResponse, LLMToolCall

ScriptCallable = Callable[[list[ChatMessage]], Awaitable[LLMResponse | None] | LLMResponse | None]
ScriptStep = LLMResponse | Exception | ScriptCallable


class ScriptExhaustedError(HarnessError):
    """The scripted LLM was asked for more responses than the script has."""


@dataclass(frozen=True)
class ScriptedRequest:
    messages: list[ChatMessage]
    tools: list[dict[str, Any]]
    timeout_s: float


class ScriptedLLM:
    model_name = "scripted"

    def __init__(self, steps: Iterable[ScriptStep]) -> None:
        self._steps: deque[ScriptStep] = deque(steps)
        self.requests: list[ScriptedRequest] = []

    @property
    def remaining_steps(self) -> int:
        return len(self._steps)

    async def complete(
        self, messages: list[ChatMessage], tools: list[dict[str, Any]], timeout_s: float
    ) -> LLMResponse:
        self.requests.append(ScriptedRequest(list(messages), tools, timeout_s))
        while self._steps:
            step = self._steps.popleft()
            if isinstance(step, LLMResponse):
                return step
            if isinstance(step, Exception):
                raise step
            result = step(messages)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                return result
        raise ScriptExhaustedError(
            f"scripted LLM has no step left for request #{len(self.requests)}"
        )


# ------------------------------------------------------------------ step builders

_call_ids = itertools.count(1)


def call(name: str, args: dict[str, Any] | str, call_id: str | None = None) -> LLMToolCall:
    """A tool call. Pass a string for ``args`` to script malformed JSON."""
    raw = args if isinstance(args, str) else json.dumps(args)
    return LLMToolCall(id=call_id or f"call_{next(_call_ids)}", name=name, arguments_raw=raw)


def tool_calls(*calls: LLMToolCall, text: str | None = None) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=list(calls))


def final(text: str) -> LLMResponse:
    return LLMResponse(text=text)


def empty() -> LLMResponse:
    return LLMResponse()


def last_tool_result(messages: list[ChatMessage]) -> dict[str, Any]:
    """The parsed JSON observation in the most recent tool message."""
    for message in reversed(messages):
        if message.role == "tool" and message.content:
            parsed: dict[str, Any] = json.loads(message.content)
            return parsed
    raise ValueError("no tool message in the conversation")


# ------------------------------------------------------------------ demo scenario


def _checkout_outage_final_answer(messages: list[ChatMessage]) -> LLMResponse:
    result = last_tool_result(messages)
    findings = (
        "Findings: checkout-api is degraded (12.5% errors, p95 1850 ms) because its "
        "dependency payment-gateway is in a full outage (96.8% errors)."
    )
    if result.get("ok"):
        action = f"Action taken: opened {result['data']['incident_id']} (SEV1) for payment-gateway."
        next_steps = "Next steps: follow the payment-gateway outage runbook; page payments on-call."
    else:
        reason = result.get("error", {}).get("message", "rejected")
        action = f"Action taken: none. Incident creation was not approved ({reason})."
        next_steps = "Next steps: keep monitoring payment-gateway and escalate manually if needed."
    return final(f"{findings}\n{action}\n{next_steps}")


CHECKOUT_INCIDENT_ARGS: dict[str, Any] = {
    "title": "payment-gateway: full outage causing checkout failures",
    "description": (
        "Impact: customers cannot complete checkout. Evidence: payment-gateway status outage "
        "with 96.8% error rate; checkout-api degraded with 12.5% errors and p95 1850 ms. "
        "Suspected cause: payment-gateway outage. Severity: SEV1 per the severity matrix "
        "(revenue-critical service in outage)."
    ),
    "severity": "SEV1",
}


def checkout_outage_script() -> list[ScriptStep]:
    """KB → checkout-api → payment-gateway → approval-gated incident → answer.

    The final step reads the last observation, so the same script works for both
    the approve and the reject path.
    """
    return [
        tool_calls(call("search_knowledge_base", {"query": "checkout failures payment gateway"})),
        tool_calls(call("get_service_status", {"service_name": "checkout-api"})),
        tool_calls(call("get_service_status", {"service_name": "payment-gateway"})),
        tool_calls(call("create_incident", CHECKOUT_INCIDENT_ARGS)),
        _checkout_outage_final_answer,
    ]
