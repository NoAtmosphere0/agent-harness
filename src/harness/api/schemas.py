"""Request and response bodies. Records from the store are returned as they are
where they already make a good API shape; only runs get a view, to attach the
pending approval."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from harness.config import ConfigOverrides
from harness.domain.models import ApprovalRecord, RunRecord


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


class RunView(RunRecord):
    pending_approval: ApprovalRecord | None = None


class HealthView(BaseModel):
    status: Literal["ok"] = "ok"
    provider: str
    model: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
