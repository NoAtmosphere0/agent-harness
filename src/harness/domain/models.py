"""Statuses, reasons, results and persisted records (PLAN §6.1, §10, §11).

The ``*Record`` models are what the repository returns: plain pydantic objects, so
no SQLAlchemy session or lazy loading leaks past the store.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator

from harness.domain.errors import ToolError

Severity = Literal["SEV1", "SEV2", "SEV3", "SEV4"]


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    LIMIT_EXCEEDED = "limit_exceeded"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset(
    {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.LIMIT_EXCEEDED, RunStatus.CANCELLED}
)


class TerminationReason(StrEnum):
    FINAL_ANSWER = "final_answer"
    MAX_STEPS = "max_steps"
    MAX_RUN_TIME = "max_run_time"
    REPEATED_TOOL_CALL = "repeated_tool_call"
    TOO_MANY_TOOL_ERRORS = "too_many_tool_errors"
    MALFORMED_LLM_OUTPUT = "malformed_llm_output"
    LLM_UNAVAILABLE = "llm_unavailable"
    LLM_REQUEST_REJECTED = "llm_request_rejected"
    INTERNAL_ERROR = "internal_error"
    CANCELLED_BY_USER = "cancelled_by_user"


# Every reason maps to exactly one terminal status (PLAN §6.1).
STATUS_FOR_REASON: dict[TerminationReason, RunStatus] = {
    TerminationReason.FINAL_ANSWER: RunStatus.COMPLETED,
    TerminationReason.MAX_STEPS: RunStatus.LIMIT_EXCEEDED,
    TerminationReason.MAX_RUN_TIME: RunStatus.LIMIT_EXCEEDED,
    TerminationReason.REPEATED_TOOL_CALL: RunStatus.LIMIT_EXCEEDED,
    TerminationReason.TOO_MANY_TOOL_ERRORS: RunStatus.FAILED,
    TerminationReason.MALFORMED_LLM_OUTPUT: RunStatus.FAILED,
    TerminationReason.LLM_UNAVAILABLE: RunStatus.FAILED,
    TerminationReason.LLM_REQUEST_REJECTED: RunStatus.FAILED,
    TerminationReason.INTERNAL_ERROR: RunStatus.FAILED,
    TerminationReason.CANCELLED_BY_USER: RunStatus.CANCELLED,
}


class ToolCallStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INVALID = "invalid"
    REJECTED = "rejected"
    NOT_EXECUTED = "not_executed"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class EventType(StrEnum):
    """Trace event types (PLAN §11). Adding one is a deliberate spec change."""

    RUN_CREATED = "run_created"
    RUN_STARTED = "run_started"
    RUN_RESUMED = "run_resumed"
    LLM_RESPONSE = "llm_response"
    LLM_RETRY = "llm_retry"
    LLM_MALFORMED_OUTPUT = "llm_malformed_output"
    TOOL_INVALID_CALL = "tool_invalid_call"
    TOOL_ATTEMPT_FAILED = "tool_attempt_failed"
    TOOL_SUCCEEDED = "tool_succeeded"
    TOOL_FAILED = "tool_failed"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_DECIDED = "approval_decided"
    LIMIT_EXCEEDED = "limit_exceeded"
    RUN_FINISHED = "run_finished"


class ToolErrorInfo(BaseModel):
    """The ``error`` object the model sees in a failed observation."""

    model_config = ConfigDict(frozen=True)

    type: str
    message: str
    retryable: bool

    @classmethod
    def from_error(cls, error: ToolError) -> Self:
        return cls(type=error.error_type, message=error.message, retryable=error.retryable)


class ToolResult(BaseModel):
    """Outcome of one tool execution, after all retries. Exactly one of data/error is set."""

    model_config = ConfigDict(frozen=True)

    tool_name: str
    ok: bool
    data: dict[str, Any] | None = None
    error: ToolErrorInfo | None = None
    attempts: int
    duration_ms: int

    @model_validator(mode="after")
    def _data_xor_error(self) -> Self:
        if self.ok != (self.data is not None) or self.ok == (self.error is not None):
            raise ValueError("ok results carry data, failed results carry an error")
        return self

    @classmethod
    def succeeded(
        cls, tool_name: str, data: dict[str, Any], *, attempts: int, duration_ms: int
    ) -> Self:
        return cls(
            tool_name=tool_name, ok=True, data=data, attempts=attempts, duration_ms=duration_ms
        )

    @classmethod
    def failed(cls, error: ToolError, *, attempts: int, duration_ms: int) -> Self:
        return cls(
            tool_name=error.tool_name,
            ok=False,
            error=ToolErrorInfo.from_error(error),
            attempts=attempts,
            duration_ms=duration_ms,
        )


# ---------------------------------------------------------------- persisted records

MessageRole = Literal["system", "user", "assistant", "tool"]


class _Record(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class PendingToolCall(BaseModel):
    """The one tool call a paused run is waiting on. Its ``args`` are what gets
    executed on approval (P4): the model is not asked to produce them again."""

    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    approval_id: str


class RunRecord(_Record):
    """A run's persisted state. The loop mutates a copy and saves it back (P2)."""

    id: str
    objective: str
    status: RunStatus
    termination_reason: TerminationReason | None = None
    final_answer: str | None = None
    step_count: int = 0
    active_elapsed_ms: int = 0
    consecutive_tool_errors: int = 0
    consecutive_malformed: int = 0
    pending_tool_call: PendingToolCall | None = None
    config: dict[str, Any]
    created_at: AwareDatetime
    updated_at: AwareDatetime
    completed_at: AwareDatetime | None = None


class MessageRecord(_Record):
    run_id: str
    seq: int
    role: MessageRole
    content: str | None
    tool_calls: list[dict[str, Any]] | None
    tool_call_id: str | None
    created_at: AwareDatetime


class ToolCallRecord(_Record):
    run_id: str
    step: int
    tool_call_id: str
    tool_name: str
    args: dict[str, Any] | None
    args_hash: str | None
    status: ToolCallStatus
    result: dict[str, Any] | None
    error_type: str | None
    attempts: int
    duration_ms: int
    created_at: AwareDatetime


class ApprovalRecord(_Record):
    id: str
    run_id: str
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    status: ApprovalStatus
    reason: str | None
    decided_by: str | None
    requested_at: AwareDatetime
    decided_at: AwareDatetime | None


class EventRecord(_Record):
    run_id: str
    seq: int
    ts: AwareDatetime
    type: EventType
    step: int | None
    payload: dict[str, Any]
