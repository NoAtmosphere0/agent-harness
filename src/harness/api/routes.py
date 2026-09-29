"""HTTP endpoints. Each one delegates to a service; no harness logic lives here."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, Query, Request, Response, status

from harness.api.schemas import (
    CreateRunRequest,
    DecisionRequest,
    ErrorResponse,
    HealthView,
    RunView,
)
from harness.domain.errors import RunNotFoundError
from harness.domain.models import (
    ApprovalRecord,
    EventRecord,
    MessageRecord,
    RunRecord,
    RunStatus,
)
from harness.tools.incidents import IncidentRecord
from harness.wiring import Container

router = APIRouter()

_ERRORS: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
}


def _container(request: Request) -> Container:
    return cast(Container, request.app.state.container)


ContainerDep = Annotated[Container, Depends(_container)]


@router.get("/health", response_model=HealthView, tags=["Health"])
async def health(c: ContainerDep) -> HealthView:
    return HealthView(provider=c.settings.llm_provider, model=c.llm.model_name)


@router.post(
    "/runs",
    response_model=RunView,
    status_code=status.HTTP_202_ACCEPTED,
    responses={200: {"model": RunView}, **_ERRORS},
    tags=["Runs"],
)
async def create_run(
    body: CreateRunRequest, request: Request, response: Response, c: ContainerDep
) -> RunView:
    """Start a run in the background (202), or with ``wait`` block until it
    finishes or waits for approval (200). A wait that times out returns 202."""
    run = await c.runs.create_run(body.objective, body.config_overrides)
    task = c.runs.start(run.id)
    if body.wait:
        # asyncio.wait never cancels the task: on timeout the run keeps going.
        done, _ = await asyncio.wait({task}, timeout=request.app.state.wait_timeout_s)
        if done:
            response.status_code = status.HTTP_200_OK
    return await _run_view(c, run.id)


@router.get("/runs", response_model=list[RunView], tags=["Runs"])
async def list_runs(
    c: ContainerDep, limit: Annotated[int, Query(ge=1, le=200)] = 50
) -> list[RunView]:
    return [await _view(c, run) for run in await c.repo.list_runs(limit)]


@router.get("/runs/{run_id}", response_model=RunView, responses=_ERRORS, tags=["Runs"])
async def get_run(run_id: str, c: ContainerDep) -> RunView:
    return await _run_view(c, run_id)


@router.get(
    "/runs/{run_id}/messages",
    response_model=list[MessageRecord],
    responses=_ERRORS,
    tags=["Runs"],
)
async def get_messages(run_id: str, c: ContainerDep) -> list[MessageRecord]:
    await _require_run(c, run_id)
    return await c.repo.list_messages(run_id)


@router.get(
    "/runs/{run_id}/trace", response_model=list[EventRecord], responses=_ERRORS, tags=["Runs"]
)
async def get_trace(run_id: str, c: ContainerDep) -> list[EventRecord]:
    await _require_run(c, run_id)
    return await c.repo.list_events(run_id)


@router.post(
    "/runs/{run_id}/approvals/{approval_id}",
    response_model=ApprovalRecord,
    responses=_ERRORS,
    tags=["Approvals"],
)
async def decide(
    run_id: str, approval_id: str, body: DecisionRequest, c: ContainerDep
) -> ApprovalRecord:
    """Approve or reject the pending tool call; the run resumes in the background."""
    return await c.approvals.decide(
        run_id, approval_id, body.decision, reason=body.reason, decided_by=body.decided_by
    )


@router.post("/runs/{run_id}/cancel", response_model=RunView, responses=_ERRORS, tags=["Runs"])
async def cancel(run_id: str, c: ContainerDep) -> RunView:
    run = await c.runs.cancel(run_id)
    return await _view(c, run)


@router.get("/incidents", response_model=list[IncidentRecord], tags=["Incidents"])
async def list_incidents(c: ContainerDep) -> list[IncidentRecord]:
    """Contents of the mock incident system."""
    return await c.incidents.list_incidents()


async def _require_run(c: Container, run_id: str) -> RunRecord:
    run = await c.repo.get_run(run_id)
    if run is None:
        raise RunNotFoundError(f"run {run_id} not found")
    return run


async def _run_view(c: Container, run_id: str) -> RunView:
    return await _view(c, await _require_run(c, run_id))


async def _view(c: Container, run: RunRecord) -> RunView:
    pending = None
    if run.status is RunStatus.WAITING_APPROVAL and run.pending_tool_call:
        pending = await c.repo.get_approval(run.pending_tool_call.approval_id)
    return RunView(**run.model_dump(), pending_approval=pending)
