"""structlog configuration.

Logs are JSON lines on stderr (stdout is left to the CLI). Two rules from PLAN §11
are enforced here rather than at each call site:

* long strings are truncated to ``LOG_VALUE_MAX_CHARS``; full message content
  lives in the database, not the logs;
* secrets never appear: the API key is a ``SecretStr`` in ``Settings``, so even
  logging the settings object prints a mask.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, cast

import structlog
from structlog.typing import EventDict, FilteringBoundLogger, WrappedLogger

from harness.config import LogFormat, LogLevel

LOG_VALUE_MAX_CHARS = 500


def _truncate(value: Any) -> Any:
    if isinstance(value, str) and len(value) > LOG_VALUE_MAX_CHARS:
        dropped = len(value) - LOG_VALUE_MAX_CHARS
        return f"{value[:LOG_VALUE_MAX_CHARS]}[truncated {dropped} chars]"
    if isinstance(value, dict):
        return {k: _truncate(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_truncate(v) for v in value]
    return value


def truncate_long_values(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """structlog processor: cap every string value, including nested ones."""
    return {key: _truncate(value) for key, value in event_dict.items()}


def configure_logging(level: LogLevel = "INFO", fmt: LogFormat = "json") -> None:
    """Configure structlog once at process start (API, CLI, tests)."""
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            truncate_long_values,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelNamesMapping()[level]),
        logger_factory=_stderr_logger,
        # Not cached, so reconfiguring (e.g. in tests) affects existing loggers.
        cache_logger_on_first_use=False,
    )


def _stderr_logger(*_args: Any) -> structlog.PrintLogger:
    # Resolve sys.stderr per logger rather than once at configure time, so a stream
    # that was swapped out and closed (e.g. by pytest's capture) is never written to.
    return structlog.PrintLogger(sys.stderr)


def get_logger(**initial_values: Any) -> FilteringBoundLogger:
    """Return a bound logger, e.g. ``get_logger(component="executor")``."""
    return cast(FilteringBoundLogger, structlog.get_logger(**initial_values))
