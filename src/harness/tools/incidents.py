"""``create_incident``: opens an incident in the mock incident system (PLAN §7.3).

The only side-effecting tool. Two guarantees live here rather than in the loop:

* it refuses to run without a harness-generated idempotency key, which only the
  approved-execution path provides (P3, defence in depth behind the approval gate);
* a repeated key returns the original incident instead of creating another (P6),
  including when two calls race past the lookup and one loses on the unique key.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from harness.domain.errors import HarnessError, ToolPermanentError, ToolTransientError
from harness.domain.models import Severity
from harness.tools.spec import RetryPolicy, ToolContext, ToolSpec

TOOL_NAME = "create_incident"


class IdempotencyKeyConflictError(HarnessError):
    """Raised by an ``IncidentStore`` when another call already created an incident
    with this key. Keeps storage-specific errors (e.g. IntegrityError) out of the tool."""


def format_incident_id(number: int) -> str:
    return f"INC-{number:06d}"


class CreateIncidentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(
        min_length=10, max_length=120, description="Short summary: service + symptom."
    )
    description: str = Field(
        min_length=20,
        max_length=2000,
        description="Impact, evidence, suspected cause and severity justification.",
    )
    severity: Severity = Field(description="Severity from the KB severity matrix.")


class IncidentRecord(BaseModel):
    """An incident as stored by the incident system."""

    incident_id: str
    title: str
    description: str
    severity: Severity
    status: Literal["open"] = "open"
    created_at: AwareDatetime


class CreateIncidentOutput(BaseModel):
    incident_id: str = Field(pattern=r"^INC-\d{6}$")
    status: Literal["open"]
    severity: Severity
    created_at: AwareDatetime
    deduplicated: bool


class IncidentStore(Protocol):
    """Persistence for the mock incident system; ``idempotency_key`` is unique."""

    async def get_by_idempotency_key(self, idempotency_key: str) -> IncidentRecord | None: ...

    async def create(
        self, *, idempotency_key: str, title: str, description: str, severity: Severity
    ) -> IncidentRecord:
        """Insert a new incident. Raises ``IdempotencyKeyConflictError`` if the key exists."""
        ...


def make_create_incident_tool(store: IncidentStore, retry: RetryPolicy) -> ToolSpec:
    async def handler(args: CreateIncidentInput, ctx: ToolContext) -> dict[str, Any]:
        key = ctx.idempotency_key
        if key is None:
            # P3: only the approved-execution path sets a key, so this cannot be reached
            # by the model alone. Kept as a second line of defence behind dispatch.
            raise ToolPermanentError(TOOL_NAME, "refusing to create an incident without approval")

        # P6: a retried call (e.g. after a timeout that happened post-commit) gets
        # the original incident back instead of a duplicate.
        existing = await store.get_by_idempotency_key(key)
        if existing is not None:
            return _output(existing, deduplicated=True)
        try:
            record = await store.create(
                idempotency_key=key,
                title=args.title,
                description=args.description,
                severity=args.severity,
            )
        except IdempotencyKeyConflictError:
            # P6: a concurrent call with the same key committed between our lookup and
            # insert. The unique constraint stopped the duplicate; return its incident.
            winner = await store.get_by_idempotency_key(key)
            if winner is None:
                raise ToolTransientError(
                    TOOL_NAME, "idempotency conflict but no incident found; retry"
                ) from None
            return _output(winner, deduplicated=True)
        return _output(record, deduplicated=False)

    return ToolSpec(
        name=TOOL_NAME,
        description=(
            "Open an incident. Requires human approval before it runs; investigate first "
            "and pick the severity from the KB severity matrix."
        ),
        input_model=CreateIncidentInput,
        output_model=CreateIncidentOutput,
        handler=handler,
        timeout_s=5.0,
        retry=retry,
        requires_approval=True,
    )


def _output(record: IncidentRecord, *, deduplicated: bool) -> dict[str, Any]:
    return {
        "incident_id": record.incident_id,
        "status": record.status,
        "severity": record.severity,
        "created_at": record.created_at.isoformat(),
        "deduplicated": deduplicated,
    }
