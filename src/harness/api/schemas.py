"""Request and response bodies. Records from the store are returned as they are
where they already make a good API shape; runs get a public view that attaches the
pending approval and leaves loop internals out."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from harness.config import ConfigOverrides
from harness.domain.models import ApprovalRecord, RunRecord, RunStatus, TerminationReason


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    objective: str = Field(min_length=1, max_length=2000)
    config_overrides: ConfigOverrides | None = None
    wait: bool = Field(
        default=False,
        description="Block until the run finishes or waits for approval (at most 60 s).",
    )


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    decision: Literal["approve", "reject"]
    reason: str | None = Field(default=None, max_length=2000)
    decided_by: str | None = Field(default=None, max_length=128)


class RunView(BaseModel):
    """The public shape of a run. Loop internals (error counters, the pending call
    as the loop stores it) stay out; the trace records every decision instead."""

    id: str
    objective: str
    status: RunStatus
    termination_reason: TerminationReason | None
    final_answer: str | None
    step_count: int
    active_elapsed_ms: int
    pending_approval: ApprovalRecord | None
    config: dict[str, Any]
    created_at: AwareDatetime
    updated_at: AwareDatetime
    completed_at: AwareDatetime | None

    @classmethod
    def from_run(cls, run: RunRecord, pending_approval: ApprovalRecord | None) -> RunView:
        return cls(
            **run.model_dump(include=set(cls.model_fields) - {"pending_approval"}),
            pending_approval=pending_approval,
        )


class HealthView(BaseModel):
    status: Literal["ok"] = "ok"
    provider: str
    model: str
    note: str | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
