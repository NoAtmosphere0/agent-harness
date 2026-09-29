"""Table definitions (PLAN §10).

Timestamps come from the injected ``Clock`` rather than database defaults, so tests
control them. JSON columns use ``none_as_null`` so Python ``None`` is SQL NULL, not
the JSON literal ``null``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Dialect,
    ForeignKey,
    Index,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

_JSON = JSON(none_as_null=True)


class UTCDateTime(TypeDecorator[datetime]):
    """Stores UTC, always returns timezone-aware values.

    SQLite has no timezone support and hands back naive datetimes; this makes
    every layer above the store see aware UTC datetimes regardless of backend.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime; pass an aware UTC datetime from the Clock")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class Base(DeclarativeBase):
    pass


class RunRow(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    objective: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), index=True)
    termination_reason: Mapped[str | None] = mapped_column(String(32))
    final_answer: Mapped[str | None] = mapped_column(Text)
    step_count: Mapped[int] = mapped_column(default=0)
    active_elapsed_ms: Mapped[int] = mapped_column(default=0)
    consecutive_tool_errors: Mapped[int] = mapped_column(default=0)
    consecutive_malformed: Mapped[int] = mapped_column(default=0)
    pending_tool_call: Mapped[dict[str, Any] | None] = mapped_column(_JSON)
    config: Mapped[dict[str, Any]] = mapped_column(_JSON)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class MessageRow(Base):
    __tablename__ = "messages"
    __table_args__ = (UniqueConstraint("run_id", "seq"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    seq: Mapped[int]
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str | None] = mapped_column(Text)
    tool_calls: Mapped[list[dict[str, Any]] | None] = mapped_column(_JSON)
    tool_call_id: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)


class ToolCallRow(Base):
    __tablename__ = "tool_calls"
    # Serves the repeated-identical-call check (PLAN §8.4).
    __table_args__ = (Index("ix_tool_calls_run_args_hash", "run_id", "args_hash"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    step: Mapped[int]
    tool_call_id: Mapped[str] = mapped_column(String(128))
    tool_name: Mapped[str] = mapped_column(String(128))
    # Null for calls whose arguments were not a JSON object.
    args: Mapped[dict[str, Any] | None] = mapped_column(_JSON)
    args_hash: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    result: Mapped[dict[str, Any] | None] = mapped_column(_JSON)
    error_type: Mapped[str | None] = mapped_column(String(64))
    attempts: Mapped[int] = mapped_column(default=0)
    duration_ms: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)


class ApprovalRow(Base):
    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    tool_call_id: Mapped[str] = mapped_column(String(128))
    tool_name: Mapped[str] = mapped_column(String(128))
    args: Mapped[dict[str, Any]] = mapped_column(_JSON)
    status: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(Text)
    decided_by: Mapped[str | None] = mapped_column(String(128))
    requested_at: Mapped[datetime] = mapped_column(UTCDateTime)
    decided_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class EventRow(Base):
    """Append-only trace. Nothing updates or deletes these rows."""

    __tablename__ = "events"
    __table_args__ = (UniqueConstraint("run_id", "seq"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    seq: Mapped[int]
    ts: Mapped[datetime] = mapped_column(UTCDateTime)
    type: Mapped[str] = mapped_column(String(64))
    step: Mapped[int | None]
    payload: Mapped[dict[str, Any]] = mapped_column(_JSON)


class IncidentRow(Base):
    """The mock external incident system. Deliberately not linked to runs by a
    foreign key: in reality it would be a separate service."""

    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True)  # P6
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)
