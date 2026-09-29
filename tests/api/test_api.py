"""REST API over the real services, SQLite and the scripted provider (PLAN §16)."""

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from harness.api.app import create_app
from harness.clock import FakeClock
from harness.config import Settings
from harness.domain.models import RunStatus, TerminationReason
from harness.llm.base import ChatMessage, LLMResponse
from harness.llm.scripted import ScriptedLLM
from harness.wiring import Container, build_container

OBJECTIVE = "Customers are reporting failed checkouts. Investigate checkout-api."


async def _create(client: httpx.AsyncClient, **body: Any) -> httpx.Response:
    return await client.post("/runs", json={"objective": OBJECTIVE, **body})


async def _waiting_run(client: httpx.AsyncClient) -> dict[str, Any]:
    response = await _create(client, wait=True)
    assert response.status_code == 200
    run: dict[str, Any] = response.json()
    assert run["status"] == "waiting_approval"
    return run


def _decide_url(run: dict[str, Any]) -> str:
    return f"/runs/{run['id']}/approvals/{run['pending_approval']['id']}"


def _error_code(response: httpx.Response) -> str:
    code: str = response.json()["error"]["code"]
    return code


async def test_api_health(client: httpx.AsyncClient):
    response = await client.get("/health")

    assert response.json() == {"status": "ok", "provider": "scripted", "model": "scripted"}


async def test_api_create_returns_202_then_run_reaches_approval(
    client: httpx.AsyncClient, container: Container
):
    response = await _create(client)

    assert response.status_code == 202
    assert response.json()["status"] == "pending"
    await container.runs.drain()
    run = (await client.get(f"/runs/{response.json()['id']}")).json()
    assert run["status"] == "waiting_approval"
    assert run["pending_approval"]["tool_name"] == "create_incident"


async def test_api_wait_returns_at_waiting_approval(client: httpx.AsyncClient):
    run = await _waiting_run(client)

    assert run["step_count"] == 4
    assert run["pending_approval"]["status"] == "pending"
    assert run["pending_approval"]["args"]["severity"] == "SEV1"


async def test_api_approve_completes_run_with_one_incident(
    client: httpx.AsyncClient, container: Container
):
    run = await _waiting_run(client)

    response = await client.post(
        _decide_url(run), json={"decision": "approve", "decided_by": "bob"}
    )
    await container.runs.drain()

    assert response.status_code == 200
    assert response.json()["status"] == "approved"
    final = (await client.get(f"/runs/{run['id']}")).json()
    assert final["status"] == "completed"
    assert final["pending_approval"] is None
    incidents = (await client.get("/incidents")).json()
    assert [i["incident_id"] for i in incidents] == ["INC-000001"]
    assert "INC-000001" in final["final_answer"]


async def test_api_reject_completes_without_incident(
    client: httpx.AsyncClient, container: Container
):
    run = await _waiting_run(client)

    await client.post(_decide_url(run), json={"decision": "reject", "reason": "not yet"})
    await container.runs.drain()

    final = (await client.get(f"/runs/{run['id']}")).json()
    assert final["status"] == "completed"
    assert (await client.get("/incidents")).json() == []


async def test_api_second_decision_conflicts(client: httpx.AsyncClient, container: Container):
    run = await _waiting_run(client)
    await client.post(_decide_url(run), json={"decision": "approve"})
    await container.runs.drain()

    response = await client.post(_decide_url(run), json={"decision": "reject"})

    assert response.status_code == 409
    assert _error_code(response) == "approval_not_pending"


async def test_api_decision_after_cancel_conflicts(client: httpx.AsyncClient):
    run = await _waiting_run(client)
    await client.post(f"/runs/{run['id']}/cancel")

    response = await client.post(_decide_url(run), json={"decision": "approve"})

    assert response.status_code == 409
    assert _error_code(response) == "run_not_waiting"
    assert (await client.get("/incidents")).json() == []


async def test_api_cancel_then_cancel_again_conflicts(client: httpx.AsyncClient):
    run = await _waiting_run(client)

    first = await client.post(f"/runs/{run['id']}/cancel")
    second = await client.post(f"/runs/{run['id']}/cancel")

    assert first.status_code == 200
    assert first.json()["status"] == "cancelled"
    assert second.status_code == 409
    assert _error_code(second) == "run_already_terminal"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/runs/nope"),
        ("GET", "/runs/nope/messages"),
        ("GET", "/runs/nope/trace"),
        ("POST", "/runs/nope/cancel"),
    ],
)
async def test_api_unknown_run_is_404(client: httpx.AsyncClient, method: str, path: str):
    response = await client.request(method, path)

    assert response.status_code == 404
    assert _error_code(response) == "run_not_found"


async def test_api_unknown_approval_is_404(client: httpx.AsyncClient):
    run = await _waiting_run(client)

    response = await client.post(f"/runs/{run['id']}/approvals/nope", json={"decision": "approve"})

    assert response.status_code == 404
    assert _error_code(response) == "approval_not_found"


@pytest.mark.parametrize(
    "body",
    [{}, {"objective": ""}, {"objective": "   "}, {"objective": "x" * 2001}, {"objective": 42}],
)
async def test_api_invalid_objective_is_422(client: httpx.AsyncClient, body: dict[str, Any]):
    response = await client.post("/runs", json=body)

    assert response.status_code == 422
    assert _error_code(response) == "validation_error"
    assert "objective" in response.json()["error"]["message"]


async def test_api_override_above_operator_limit_is_422(client: httpx.AsyncClient):
    response = await _create(client, config_overrides={"max_steps": 13})

    assert response.status_code == 422
    assert _error_code(response) == "invalid_config_overrides"
    assert (await client.get("/runs")).json() == []


async def test_api_invalid_decision_is_422(client: httpx.AsyncClient):
    run = await _waiting_run(client)

    response = await client.post(_decide_url(run), json={"decision": "maybe"})

    assert response.status_code == 422


async def test_api_trace_is_ordered(client: httpx.AsyncClient, container: Container):
    run = await _waiting_run(client)
    await client.post(_decide_url(run), json={"decision": "approve"})
    await container.runs.drain()

    events = (await client.get(f"/runs/{run['id']}/trace")).json()

    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert events[0]["type"] == "run_created"
    assert events[-1]["type"] == "run_finished"
    assert "approval_decided" in [e["type"] for e in events]


async def test_api_messages_and_run_list(client: httpx.AsyncClient):
    run = await _waiting_run(client)

    messages = (await client.get(f"/runs/{run['id']}/messages")).json()
    runs = (await client.get("/runs")).json()

    assert [m["role"] for m in messages[:2]] == ["system", "user"]
    assert messages[1]["content"] == OBJECTIVE
    assert [r["id"] for r in runs] == [run["id"]]
    assert runs[0]["pending_approval"]["id"] == run["pending_approval"]["id"]


async def test_api_fault_override_shows_retries_in_trace(client: httpx.AsyncClient):
    response = await _create(
        client,
        wait=True,
        config_overrides={"faults": {"get_service_status": {"mode": "flaky", "fail_times": 1}}},
    )

    events = (await client.get(f"/runs/{response.json()['id']}/trace")).json()

    # The flaky counter is per (run, tool): only the first status call fails, then recovers.
    types = [e["type"] for e in events]
    assert types.count("tool_attempt_failed") == 1
    assert types[-1] == "approval_requested"


async def test_api_wait_timeout_returns_202(container: Container):
    app = create_app(container)
    app.state.wait_timeout_s = 0.0
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await _create(client, wait=True)

    assert response.status_code == 202
    await container.runs.drain()


async def test_api_shutdown_fails_in_flight_runs(
    engine: AsyncEngine, clock: FakeClock, data_dir: Path
):
    entered = asyncio.Event()

    async def never_answers(messages: list[ChatMessage]) -> LLMResponse:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    container = await build_container(
        Settings(_env_file=None, data_dir=data_dir),
        llm=ScriptedLLM([never_answers]),
        clock=clock,
        engine=engine,
    )
    app = create_app(container)
    async with app.router.lifespan_context(app):
        run = await container.runs.create_run(OBJECTIVE)
        container.runs.start(run.id)
        await entered.wait()

    stored = await container.repo.get_run(run.id)
    assert stored is not None
    assert stored.status is RunStatus.FAILED
    assert stored.termination_reason is TerminationReason.INTERNAL_ERROR
    finished = (await container.repo.list_events(run.id))[-1]
    assert finished.payload["message"] == "shutdown"
