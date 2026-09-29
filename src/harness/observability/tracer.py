"""Per-run trace events (PLAN §11).

Each event is written twice with the same fields: to the append-only ``events``
table (what ``GET /runs/{id}/trace`` returns) and as a structlog JSON line (what an
operator tails). P10: every decision point in the harness goes through here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from harness.domain.models import EventType
from harness.observability.logging import get_logger

if TYPE_CHECKING:
    # Type-only: the tool executor imports EventSink from here and should not
    # pull in the store layer with it.
    from harness.store.repository import Repository


class EventSink(Protocol):
    """Anything that can record a trace event for a run."""

    async def emit(
        self, run_id: str, event_type: EventType, step: int | None = None, **payload: Any
    ) -> None: ...


class Tracer:
    """The production ``EventSink``: persist, then log."""

    def __init__(self, repo: Repository) -> None:
        self._repo = repo
        self._log = get_logger(component="tracer")

    async def emit(
        self, run_id: str, event_type: EventType, step: int | None = None, **payload: Any
    ) -> None:
        event = await self._repo.append_event(run_id, event_type, step, payload)
        # Payload is nested so its keys can never clash with structlog's own fields.
        self._log.info(
            event_type.value, run_id=run_id, step=step, seq=event.seq, payload=event.payload
        )
