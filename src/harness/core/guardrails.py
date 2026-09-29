"""Run-level limits checked before every LLM call (PLAN §8.7, P8).

Two more limits live where their data is: the repeated-identical-call limit in
dispatch, and the consecutive-error limits at the end of a tick.
"""

from __future__ import annotations

from harness.config import RunConfig
from harness.domain.models import RunRecord, TerminationReason


def check(run: RunRecord, config: RunConfig) -> TerminationReason | None:
    if run.step_count >= config.max_steps:
        return TerminationReason.MAX_STEPS
    # P8: only active time counts. While waiting for approval, advance() has
    # returned, so nothing is added to active_elapsed_ms.
    if run.active_elapsed_ms >= config.max_run_seconds * 1000:
        return TerminationReason.MAX_RUN_TIME
    return None
