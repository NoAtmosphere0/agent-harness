"""Composition root: builds the object graph once for the API, the CLI and tests.

Nothing else in the package constructs services from settings, so this is the one
place to read how the pieces fit together.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncEngine

from harness.clock import Clock, SystemClock
from harness.config import Settings
from harness.core.loop import AgentLoop
from harness.domain.errors import HarnessError
from harness.llm.base import LLMClient
from harness.llm.scripted import ReplayLLM, checkout_outage_script
from harness.observability.tracer import Tracer
from harness.services.approval_service import ApprovalService
from harness.services.run_service import RunService
from harness.store.db import create_engine, create_session_factory, init_db
from harness.store.repository import Repository, SqlIncidentStore
from harness.tools.executor import ToolExecutor
from harness.tools.faults import FaultInjector
from harness.tools.registry import build_registry


@dataclass
class Container:
    settings: Settings
    engine: AsyncEngine
    clock: Clock
    llm: LLMClient
    repo: Repository
    incidents: SqlIncidentStore
    runs: RunService
    approvals: ApprovalService

    async def close(self) -> None:
        """Fail in-flight runs (§12), then release the database."""
        await self.runs.shutdown()
        await self.engine.dispose()


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "scripted":
        # No API key needed: every run replays the checkout_outage scenario.
        return ReplayLLM(checkout_outage_script())
    raise HarnessError(f"LLM provider {settings.llm_provider!r} is not available in this build")


async def build_container(
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    clock: Clock | None = None,
    engine: AsyncEngine | None = None,
) -> Container:
    engine = engine or create_engine(settings.database_url)
    await init_db(engine)
    sessions = create_session_factory(engine)
    clock = clock or SystemClock()
    llm = llm or build_llm(settings)

    repo = Repository(sessions, clock)
    incidents = SqlIncidentStore(sessions, clock)
    tracer = Tracer(repo)
    faults = FaultInjector()
    registry = build_registry(
        data_dir=settings.data_dir,
        incident_store=incidents,
        tool_max_retries=settings.tool_max_retries,
    )
    # Seeded jitter makes a run's retry timing reproducible for a given MOCK_SEED.
    executor = ToolExecutor(clock, tracer, faults, random.Random(settings.mock_seed))
    loop = AgentLoop(
        repo=repo,
        llm=llm,
        registry=registry,
        executor=executor,
        faults=faults,
        events=tracer,
        clock=clock,
        rng=random.Random(settings.mock_seed),
    )
    runs = RunService(
        repo=repo,
        loop=loop,
        events=tracer,
        faults=faults,
        settings=settings,
        model_name=llm.model_name,
    )
    approvals = ApprovalService(repo=repo, runs=runs, events=tracer)
    return Container(settings, engine, clock, llm, repo, incidents, runs, approvals)
