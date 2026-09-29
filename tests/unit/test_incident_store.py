"""SQL-backed incident store, including the lookup-then-create race (P6)."""

import asyncio
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from harness.clock import FakeClock
from harness.domain.errors import ToolTransientError
from harness.domain.models import Severity
from harness.store.repository import SqlIncidentStore
from harness.tools.incidents import (
    CreateIncidentInput,
    IdempotencyKeyConflictError,
    IncidentRecord,
    make_create_incident_tool,
)
from harness.tools.spec import RetryPolicy, ToolContext

_ARGS = CreateIncidentInput(
    title="payment-gateway: full outage",
    description="Payment gateway error rate 96.8%; checkout failing for all customers.",
    severity="SEV1",
)


def _ctx(key: str) -> ToolContext:
    return ToolContext(run_id="run-1", step=1, idempotency_key=key, deadline_remaining_s=60)


async def _create(store: SqlIncidentStore, key: str) -> IncidentRecord:
    return await store.create(
        idempotency_key=key, title=_ARGS.title, description=_ARGS.description, severity="SEV1"
    )


async def test_incident_store_assigns_sequential_ids(sql_incident_store: SqlIncidentStore):
    first = await _create(sql_incident_store, "k1")
    second = await _create(sql_incident_store, "k2")

    assert (first.incident_id, second.incident_id) == ("INC-000001", "INC-000002")
    assert first.status == "open"
    assert await sql_incident_store.get_by_idempotency_key("k1") == first
    assert await sql_incident_store.get_by_idempotency_key("missing") is None
    assert await sql_incident_store.list_incidents() == [first, second]


async def test_incident_store_duplicate_key_raises_conflict(sql_incident_store: SqlIncidentStore):
    await _create(sql_incident_store, "k1")

    with pytest.raises(IdempotencyKeyConflictError):
        await _create(sql_incident_store, "k1")

    assert len(await sql_incident_store.list_incidents()) == 1


class _StaleLookupStore:
    """Wraps a store so the first lookup misses, as if a concurrent call committed
    between our lookup and our insert."""

    def __init__(self, inner: SqlIncidentStore) -> None:
        self._inner = inner
        self._lookups = 0

    async def get_by_idempotency_key(self, idempotency_key: str) -> IncidentRecord | None:
        self._lookups += 1
        if self._lookups == 1:
            return None
        return await self._inner.get_by_idempotency_key(idempotency_key)

    async def create(
        self, *, idempotency_key: str, title: str, description: str, severity: Severity
    ) -> IncidentRecord:
        return await self._inner.create(
            idempotency_key=idempotency_key, title=title, description=description, severity=severity
        )


async def test_create_incident_race_on_unique_key_returns_deduplicated(
    sql_incident_store: SqlIncidentStore,
):
    winner = await _create(sql_incident_store, "run-1:appr-1")
    spec = make_create_incident_tool(_StaleLookupStore(sql_incident_store), RetryPolicy())

    out = await spec.handler(_ARGS, _ctx("run-1:appr-1"))

    assert out["deduplicated"] is True
    assert out["incident_id"] == winner.incident_id
    assert len(await sql_incident_store.list_incidents()) == 1


async def test_create_incident_concurrent_calls_create_exactly_one(
    file_sessions: async_sessionmaker[AsyncSession], clock: FakeClock
):
    store = SqlIncidentStore(file_sessions, clock)
    spec = make_create_incident_tool(store, RetryPolicy())

    results = await asyncio.gather(
        spec.handler(_ARGS, _ctx("run-1:appr-1")), spec.handler(_ARGS, _ctx("run-1:appr-1"))
    )

    assert sorted(r["deduplicated"] for r in results) == [False, True]
    assert results[0]["incident_id"] == results[1]["incident_id"]
    assert len(await store.list_incidents()) == 1


async def test_create_incident_conflict_without_winner_is_transient(
    sql_incident_store: SqlIncidentStore,
):
    class _ConflictingStore(_StaleLookupStore):
        async def get_by_idempotency_key(self, idempotency_key: str) -> IncidentRecord | None:
            return None

        async def create(self, **kwargs: Any) -> IncidentRecord:
            raise IdempotencyKeyConflictError("k")

    spec = make_create_incident_tool(_ConflictingStore(sql_incident_store), RetryPolicy())

    with pytest.raises(ToolTransientError, match="retry"):
        await spec.handler(_ARGS, _ctx("run-1:appr-1"))
