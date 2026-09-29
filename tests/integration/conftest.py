"""Wires the real loop, services, SQLite store and tools around a ScriptedLLM."""

import json
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from harness.clock import FakeClock
from harness.config import ConfigOverrides, Settings
from harness.core.loop import AgentLoop
from harness.domain.models import (
    ApprovalRecord,
    EventRecord,
    EventType,
    MessageRecord,
    RunRecord,
    ToolCallRecord,
)
from harness.llm.scripted import ScriptedLLM, ScriptStep
from harness.observability.tracer import EventSink, Tracer
from harness.services.approval_service import ApprovalService, Decision
from harness.services.run_service import RunService
from harness.store.repository import Repository, SqlIncidentStore
from harness.tools.executor import ToolExecutor
from harness.tools.faults import FaultInjector
from harness.tools.incidents import IncidentRecord
from harness.tools.registry import ToolRegistry, build_registry


@dataclass
class Env:
    clock: FakeClock
    repo: Repository
    incidents: SqlIncidentStore
    registry: ToolRegistry
    faults: FaultInjector
    llm: ScriptedLLM
    runs: RunService
    approvals: ApprovalService

    async def start(
        self, objective: str = "Investigate checkout failures", **overrides: Any
    ) -> str:
        run = await self.runs.create_run(objective, ConfigOverrides(**overrides))
        return run.id

    async def advance(self, run_id: str) -> RunRecord:
        await self.runs.advance(run_id)
        return await self.get(run_id)

    async def get(self, run_id: str) -> RunRecord:
        run = await self.repo.get_run(run_id)
        assert run is not None
        return run

    async def pending_approval(self, run_id: str) -> ApprovalRecord:
        run = await self.get(run_id)
        assert run.pending_tool_call is not None, f"run is {run.status}, not paused"
        approval = await self.repo.get_approval(run.pending_tool_call.approval_id)
        assert approval is not None
        return approval

    async def decide(self, run_id: str, decision: Decision, **kwargs: Any) -> RunRecord:
        """Decide the pending approval and wait for the resumed run to settle."""
        approval = await self.pending_approval(run_id)
        await self.approvals.decide(run_id, approval.id, decision, **kwargs)
        await self.runs.drain()
        return await self.get(run_id)

    async def events(self, run_id: str) -> list[EventRecord]:
        return await self.repo.list_events(run_id)

    async def event_types(self, run_id: str) -> list[EventType]:
        return [e.type for e in await self.events(run_id)]

    async def messages(self, run_id: str) -> list[MessageRecord]:
        return await self.repo.list_messages(run_id)

    async def observations(self, run_id: str) -> list[dict[str, Any]]:
        return [
            json.loads(m.content or "") for m in await self.messages(run_id) if m.role == "tool"
        ]

    async def tool_calls(self, run_id: str) -> list[ToolCallRecord]:
        return await self.repo.list_tool_calls(run_id)

    async def incident_list(self) -> list[IncidentRecord]:
        return await self.incidents.list_incidents()


EnvFactory = Callable[..., Env]


@pytest.fixture
def make_env(
    sessions: async_sessionmaker[AsyncSession], clock: FakeClock, data_dir: Path
) -> EnvFactory:
    def build(
        steps: Iterable[ScriptStep],
        *,
        db: async_sessionmaker[AsyncSession] | None = None,
        wrap_events: Callable[[EventSink], EventSink] | None = None,
        tool_timeouts: dict[str, float] | None = None,
        **settings: Any,
    ) -> Env:
        config = Settings(_env_file=None, data_dir=data_dir, **settings)
        repo = Repository(db or sessions, clock)
        incidents = SqlIncidentStore(db or sessions, clock)
        registry = build_registry(
            data_dir=data_dir,
            incident_store=incidents,
            tool_max_retries=config.tool_max_retries,
        )
        if tool_timeouts:
            registry = ToolRegistry(
                registry.get(name).model_copy(
                    update={"timeout_s": tool_timeouts.get(name, registry.get(name).timeout_s)}
                )
                for name in registry.names()
            )
        tracer: EventSink = Tracer(repo)
        events = wrap_events(tracer) if wrap_events else tracer
        faults = FaultInjector()
        llm = ScriptedLLM(steps)
        loop = AgentLoop(
            repo=repo,
            llm=llm,
            registry=registry,
            executor=ToolExecutor(clock, events, faults, random.Random(0)),
            faults=faults,
            events=events,
            clock=clock,
            rng=random.Random(0),
        )
        runs = RunService(
            repo=repo,
            loop=loop,
            events=events,
            faults=faults,
            settings=config,
            model_name=llm.model_name,
        )
        approvals = ApprovalService(repo=repo, runs=runs, events=events)
        return Env(clock, repo, incidents, registry, faults, llm, runs, approvals)

    return build
