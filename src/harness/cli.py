"""Command-line interface (PLAN §13): run, trace, demo, serve.

The CLI uses the same services as the API, so approvals behave identically: a
paused run shows the pending call, asks for a decision, and resumes.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from enum import StrEnum
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from harness.config import ConfigOverrides, Settings, get_settings
from harness.domain.errors import ServiceError
from harness.domain.models import ApprovalRecord, EventRecord, RunRecord, RunStatus
from harness.llm.scripted import ReplayLLM, checkout_outage_script
from harness.observability.logging import configure_logging
from harness.services.approval_service import Decision
from harness.wiring import Container, build_container

app = typer.Typer(help="Agent harness for an operations assistant.", no_args_is_help=True)
console = Console()


class ApprovalMode(StrEnum):
    ASK = "ask"
    APPROVE = "approve"
    REJECT = "reject"


AutoApprove = Annotated[bool, typer.Option("--auto-approve", help="Approve every request.")]
AutoReject = Annotated[bool, typer.Option("--auto-reject", help="Reject every request.")]


@app.command()
def run(
    objective: Annotated[str, typer.Argument(help="What the assistant should do.")],
    auto_approve: AutoApprove = False,
    auto_reject: AutoReject = False,
    max_steps: Annotated[int | None, typer.Option(help="Lower the step limit.")] = None,
) -> None:
    """Run an objective with the configured LLM provider."""
    mode = _approval_mode(auto_approve, auto_reject)
    settings = _settings()
    overrides = ConfigOverrides(max_steps=max_steps)
    asyncio.run(_run_and_report(settings, objective, overrides, mode, show_trace=False))


@app.command()
def demo(
    scenario: Annotated[str, typer.Option(help="Scenario id from data/scenarios.json.")] = (
        "checkout_outage"
    ),
    auto_approve: AutoApprove = False,
    auto_reject: AutoReject = False,
) -> None:
    """Replay a scripted scenario end to end; no API key needed."""
    if scenario != "checkout_outage":
        raise typer.BadParameter(
            "only 'checkout_outage' has a scripted demo", param_hint="--scenario"
        )
    mode = _approval_mode(auto_approve, auto_reject)
    settings = _settings()
    scenarios = json.loads((settings.data_dir / "scenarios.json").read_text(encoding="utf-8"))
    objective = next(s["objective"] for s in scenarios if s["id"] == scenario)
    asyncio.run(
        _run_and_report(
            settings,
            objective,
            None,
            mode,
            show_trace=True,
            llm=ReplayLLM(checkout_outage_script()),
        )
    )


@app.command()
def trace(run_id: Annotated[str, typer.Argument(help="Run id.")]) -> None:
    """Print a run's trace as a timeline."""
    asyncio.run(_trace(_settings(), run_id))


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port.")] = 8000,
) -> None:
    """Start the REST API."""
    import uvicorn

    uvicorn.run("harness.api.app:app", host=host, port=port)


# ------------------------------------------------------------------ implementation


def _settings() -> Settings:
    settings = get_settings()
    # Trace events are printed as a timeline; keep the JSON event log off the terminal.
    configure_logging("WARNING", settings.log_format)
    return settings


def _approval_mode(auto_approve: bool, auto_reject: bool) -> ApprovalMode:
    if auto_approve and auto_reject:
        raise typer.BadParameter("use at most one of --auto-approve and --auto-reject")
    if auto_approve:
        return ApprovalMode.APPROVE
    return ApprovalMode.REJECT if auto_reject else ApprovalMode.ASK


async def _run_and_report(
    settings: Settings,
    objective: str,
    overrides: ConfigOverrides | None,
    mode: ApprovalMode,
    *,
    show_trace: bool,
    llm: ReplayLLM | None = None,
) -> None:
    container = await build_container(settings, llm=llm)
    try:
        try:
            run = await container.runs.create_run(objective, overrides)
        except ServiceError as exc:
            console.print(f"[red]{exc.code}[/red]: {exc}")
            raise typer.Exit(2) from exc
        console.print(f"[bold]run[/bold] {run.id}\n[dim]objective:[/dim] {objective}")
        final = await _drive(container, run.id, mode)
        if show_trace:
            console.print(_timeline(await container.repo.list_events(run.id)))
        _print_result(final)
    finally:
        # §12: an interrupted run (e.g. Ctrl+C) is failed, not left RUNNING.
        await container.close()


async def _drive(container: Container, run_id: str, mode: ApprovalMode) -> RunRecord:
    await container.runs.start(run_id)
    while True:
        run = await container.repo.get_run(run_id)
        assert run is not None
        if run.status is not RunStatus.WAITING_APPROVAL or run.pending_tool_call is None:
            return run
        approval = await container.repo.get_approval(run.pending_tool_call.approval_id)
        assert approval is not None
        decision, reason = _ask(approval, mode)
        await container.approvals.decide(
            run_id, approval.id, decision, reason=reason, decided_by="cli"
        )
        await container.runs.drain()


def _ask(approval: ApprovalRecord, mode: ApprovalMode) -> tuple[Decision, str | None]:
    args = json.dumps(approval.args, indent=2, ensure_ascii=False)
    console.print(
        Panel(args, title=f"approval required: {approval.tool_name}", border_style="yellow")
    )
    if mode is ApprovalMode.APPROVE:
        console.print("[green]auto-approved[/green]")
        return "approve", None
    if mode is ApprovalMode.REJECT:
        console.print("[red]auto-rejected[/red]")
        return "reject", "auto-rejected from the CLI"
    approved = typer.confirm("Approve this call?", default=False)
    reason = typer.prompt("Reason (optional)", default="", show_default=False).strip()
    return ("approve" if approved else "reject"), (reason or None)


def _print_result(run: RunRecord) -> None:
    colour = "green" if run.status is RunStatus.COMPLETED else "red"
    console.print(
        f"\n[bold {colour}]{run.status}[/bold {colour}] ({run.termination_reason}) "
        f"after {run.step_count} steps, {run.active_elapsed_ms} ms active"
    )
    if run.final_answer:
        console.print(Panel(run.final_answer, title="final answer", border_style=colour))
    console.print(f"[dim]trace: harness trace {run.id}[/dim]")


async def _trace(settings: Settings, run_id: str) -> None:
    container = await build_container(settings)
    try:
        if await container.repo.get_run(run_id) is None:
            console.print(f"[red]run_not_found[/red]: {run_id}")
            raise typer.Exit(1)
        console.print(_timeline(await container.repo.list_events(run_id)))
    finally:
        await container.close()


def _timeline(events: list[EventRecord]) -> Table:
    table = Table(title="trace", show_lines=False)
    for column in ("seq", "+ms", "step", "event", "details"):
        table.add_column(column, justify="right" if column in ("seq", "+ms", "step") else "left")
    start: datetime | None = events[0].ts if events else None
    for event in events:
        offset = (event.ts - start).total_seconds() * 1000 if start else 0.0
        table.add_row(
            str(event.seq),
            f"{offset:.0f}",
            "" if event.step is None else str(event.step),
            event.type.value,
            _details(event),
        )
    return table


def _details(event: EventRecord) -> str:
    payload = {k: v for k, v in event.payload.items() if k not in ("config", "usage")}
    text = json.dumps(payload, ensure_ascii=False, default=str)
    return text if len(text) <= 120 else text[:119] + "…"
