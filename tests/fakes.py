"""Test doubles shared across test modules."""

from dataclasses import dataclass, field
from typing import Any

from harness.clock import Clock
from harness.domain.models import EventType, Severity
from harness.tools.incidents import (
    IdempotencyKeyConflictError,
    IncidentRecord,
    format_incident_id,
)


@dataclass(frozen=True)
class RecordedEvent:
    run_id: str
    event_type: EventType
    step: int | None
    payload: dict[str, Any]


@dataclass
class RecordingEvents:
    """EventSink that keeps events in memory."""

    events: list[RecordedEvent] = field(default_factory=list)

    async def emit(
        self, run_id: str, event_type: EventType, step: int | None = None, **payload: Any
    ) -> None:
        self.events.append(RecordedEvent(run_id, event_type, step, payload))

    def types(self) -> list[EventType]:
        return [e.event_type for e in self.events]


class InMemoryIncidentStore:
    """IncidentStore without a database; the SQL-backed one is tested separately."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self.records: dict[str, IncidentRecord] = {}

    async def get_by_idempotency_key(self, idempotency_key: str) -> IncidentRecord | None:
        return self.records.get(idempotency_key)

    async def create(
        self, *, idempotency_key: str, title: str, description: str, severity: Severity
    ) -> IncidentRecord:
        if idempotency_key in self.records:
            raise IdempotencyKeyConflictError(idempotency_key)
        record = IncidentRecord(
            incident_id=format_incident_id(len(self.records) + 1),
            title=title,
            description=description,
            severity=severity,
            created_at=self._clock.now_utc(),
        )
        self.records[idempotency_key] = record
        return record
