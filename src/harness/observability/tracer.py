"""Per-run trace events (PLAN §11).

Phase 1 only needs the ``EventSink`` interface so the tool executor can emit
events; the persisted ``Tracer`` implementation is added with the store.
"""

from __future__ import annotations

from typing import Any, Protocol

from harness.domain.models import EventType


class EventSink(Protocol):
    """Anything that can record a trace event for a run (P10)."""

    async def emit(
        self, run_id: str, event_type: EventType, step: int | None = None, **payload: Any
    ) -> None: ...
