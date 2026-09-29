"""Optional live run of data/scenarios.json against a real model (PLAN §16).

Informational, not a CI gate: it prints each run's tool sequence against the
expected one and the trace timeline, and only fails if the harness itself breaks
(an internal error or a run that never finishes).

    LLM_PROVIDER=openai_compat LLM_BASE_URL=http://localhost:11434/v1 \\
    LLM_API_KEY=ollama LLM_MODEL=qwen2.5:7b make test-live
"""

import json
from pathlib import Path
from typing import Any

import pytest

from harness.config import Settings
from harness.domain.models import (
    TERMINAL_STATUSES,
    EventRecord,
    RunStatus,
    TerminationReason,
)
from harness.wiring import build_container

REPO_ROOT = Path(__file__).resolve().parents[2]
# Read at import time: the autouse clean_env fixture removes LLM_* env vars later.
LIVE = Settings()
SCENARIOS: list[dict[str, Any]] = json.loads(
    (REPO_ROOT / "data" / "scenarios.json").read_text(encoding="utf-8")
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not LIVE.llm_api_key.get_secret_value(), reason="LLM_API_KEY not set"),
]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
async def test_live_scenario(
    scenario: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    settings = LIVE.model_copy(
        update={
            "llm_provider": "openai_compat",
            "database_url": f"sqlite+aiosqlite:///{(tmp_path / 'live.db').as_posix()}",
            "data_dir": REPO_ROOT / "data",
        }
    )
    container = await build_container(settings)
    try:
        run = await container.runs.create_run(scenario["objective"])
        await container.runs.start(run.id)
        while (current := await container.repo.get_run(run.id)) is not None and (
            current.status is RunStatus.WAITING_APPROVAL and current.pending_tool_call
        ):
            await container.approvals.decide(
                run.id, current.pending_tool_call.approval_id, "approve", decided_by="live-test"
            )
            await container.runs.drain()
        assert current is not None
        calls = await container.repo.list_tool_calls(run.id)
        events = await container.repo.list_events(run.id)
        incidents = await container.incidents.list_incidents()
    finally:
        await container.close()

    used = [c.tool_name for c in calls]
    with capsys.disabled():
        print(f"\n=== {scenario['id']} ({settings.llm_model}) ===")
        print(f"objective : {scenario['objective']}")
        outcome = f"{current.status} ({current.termination_reason})"
        print(f"result    : {outcome}, {current.step_count} steps")
        print(f"tools     : {used}")
        print(
            f"expected  : {scenario['expected_tools']}  match={used == scenario['expected_tools']}"
        )
        print(
            f"incident  : {[(i.incident_id, i.severity) for i in incidents] or 'none'}"
            f" (expected: {scenario['expects_incident']}, {scenario['expected_severity']})"
        )
        print(_timeline(events))
        print(f"answer    : {current.final_answer}")

    assert current.status in TERMINAL_STATUSES
    assert current.termination_reason is not TerminationReason.INTERNAL_ERROR


def _timeline(events: list[EventRecord]) -> str:
    start = events[0].ts
    lines = []
    for e in events:
        payload = {k: v for k, v in e.payload.items() if k not in ("config", "objective", "usage")}
        detail = json.dumps(payload, ensure_ascii=False, default=str)
        detail = detail if len(detail) <= 110 else detail[:109] + "…"
        offset = (e.ts - start).total_seconds() * 1000
        step = "" if e.step is None else e.step
        lines.append(f"  {e.seq:>3} {offset:>8.0f}ms {step!s:>3}  {e.type.value:<20} {detail}")
    return "\n".join(lines)
