"""The contract every tool implements (PLAN §6.5)."""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# The first argument is an instance of the tool's own ``input_model``. It is typed
# ``Any`` because the registry holds tools with different input types; the executor
# only ever passes arguments that ``input_model`` has already validated.
ToolHandler = Callable[[Any, "ToolContext"], Awaitable[dict[str, Any]]]


class RetryPolicy(BaseModel):
    """How often and how patiently a tool call is retried (P5)."""

    model_config = ConfigDict(frozen=True)

    max_retries: int = Field(default=2, ge=0)  # attempts = max_retries + 1
    base_delay_s: float = Field(default=0.5, ge=0)
    max_delay_s: float = Field(default=4.0, ge=0)

    def backoff_delay(self, attempt: int, rng: random.Random) -> float:
        """Delay after failed attempt number ``attempt`` (1-based).

        P5: capped exponential backoff with full jitter [R6]: uniform over
        ``[0, min(max_delay, base * 2**(attempt-1))]``. Jitter spreads retries from
        many callers so they do not hit a recovering backend in lockstep.
        """
        if attempt < 1:
            raise ValueError("attempt is 1-based")
        cap = min(self.max_delay_s, self.base_delay_s * 2 ** (attempt - 1))
        return rng.uniform(0, cap)


class ToolContext(BaseModel):
    """Harness-side facts about a call that the model neither sees nor controls."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    step: int
    # P6: set only when executing an approved call; never part of the LLM schema.
    idempotency_key: str | None = None
    deadline_remaining_s: float


class ToolSpec(BaseModel):
    """A tool: typed input/output, a handler, and its execution policy."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: ToolHandler
    timeout_s: float = Field(gt=0)
    retry: RetryPolicy = RetryPolicy()
    requires_approval: bool = False

    def llm_schema(self) -> dict[str, Any]:
        """OpenAI-style function definition, derived from ``input_model`` only."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }
