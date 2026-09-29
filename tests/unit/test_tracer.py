import json

import pytest

from harness.domain.models import EventType
from harness.observability.logging import configure_logging
from harness.observability.tracer import Tracer
from harness.store.repository import Repository


async def test_tracer_persists_event_and_logs_same_fields(
    repo: Repository, tracer: Tracer, capsys: pytest.CaptureFixture[str]
):
    configure_logging("INFO", "json")
    run = await repo.create_run("objective", {})

    await tracer.emit(run.id, EventType.TOOL_SUCCEEDED, 2, tool="get_service_status", attempts=1)

    (event,) = await repo.list_events(run.id)
    assert event.type is EventType.TOOL_SUCCEEDED
    assert event.step == 2
    assert event.payload == {"tool": "get_service_status", "attempts": 1}

    line = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert line["event"] == "tool_succeeded"
    assert line["run_id"] == run.id
    assert line["step"] == 2
    assert line["seq"] == event.seq
    assert line["payload"] == event.payload


async def test_tracer_payload_keys_cannot_clash_with_log_fields(repo: Repository, tracer: Tracer):
    run = await repo.create_run("objective", {})

    await tracer.emit(
        run.id, EventType.RUN_FINISHED, None, event="x", level="y", status="completed"
    )

    (event,) = await repo.list_events(run.id)
    assert event.payload == {"event": "x", "level": "y", "status": "completed"}
