from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from harness.api.app import create_app
from harness.clock import FakeClock
from harness.config import Settings
from harness.llm.scripted import ReplayLLM, checkout_outage_script
from harness.wiring import Container, build_container


@pytest.fixture
async def container(
    engine: AsyncEngine, clock: FakeClock, data_dir: Path
) -> AsyncIterator[Container]:
    settings = Settings(_env_file=None, data_dir=data_dir)
    c = await build_container(
        settings, llm=ReplayLLM(checkout_outage_script()), clock=clock, engine=engine
    )
    yield c
    await c.runs.shutdown()


@pytest.fixture
async def client(container: Container) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(container)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
