"""Error taxonomy (PLAN §6.4).

The split matters more than the names:

* ``ToolError`` subclasses describe a failed tool call. They are turned into an
  observation the model can react to (P7) and never end a run on their own.
  ``retryable`` decides whether the executor tries again (P5).
* ``LLMError`` subclasses describe a failed model call. After the LLM retry layer
  gives up, they terminate the run with a recorded reason.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import ValidationError


class HarnessError(Exception):
    """Base class for every error the harness raises on purpose."""


class ToolError(HarnessError):
    """A tool call failed. ``retryable`` is fixed per subclass."""

    retryable: ClassVar[bool] = False

    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(f"{tool_name}: {message}")
        self.tool_name = tool_name
        self.message = message

    @property
    def error_type(self) -> str:
        """Name shown to the model in ``{"error": {"type": ...}}``."""
        return type(self).__name__


class ToolNotFoundError(ToolError):
    """The model asked for a tool that is not registered."""


class ToolArgumentsError(ToolError):
    """Arguments were not a JSON object or failed the tool's input schema."""


class ToolTimeoutError(ToolError):
    """No response within the tool's timeout. Transient, so worth retrying."""

    retryable = True


class ToolTransientError(ToolError):
    """A temporary upstream failure (e.g. HTTP 503)."""

    retryable = True


class ToolPermanentError(ToolError):
    """A failure that will not go away on retry (e.g. unknown service)."""


class ToolOutputValidationError(ToolError):
    """The tool answered, but not in its declared output schema. A bug, not a blip."""


class LLMError(HarnessError):
    """A model call failed."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable


class LLMProviderError(LLMError):
    """The provider returned an error. 429/5xx are retryable; 4xx request errors are not."""


class LLMTimeoutError(LLMError):
    """The provider did not answer in time."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


def describe_validation_error(exc: ValidationError, max_errors: int = 5) -> str:
    """Compact, model-readable summary of a pydantic error: ``field: problem; ...``."""
    parts = []
    for err in exc.errors()[:max_errors]:
        location = ".".join(str(p) for p in err["loc"]) or "<root>"
        parts.append(f"{location}: {err['msg']}")
    hidden = exc.error_count() - max_errors
    if hidden > 0:
        parts.append(f"(+{hidden} more)")
    return "; ".join(parts)


class ServiceError(HarnessError):
    """A request the service layer refuses. ``code`` is the API error code (PLAN §12)."""

    code: ClassVar[str] = "service_error"


class RunNotFoundError(ServiceError):
    code = "run_not_found"


class ApprovalNotFoundError(ServiceError):
    code = "approval_not_found"


class ApprovalNotPendingError(ServiceError):
    """The approval was already decided; the first decision stands."""

    code = "approval_not_pending"


class RunNotWaitingError(ServiceError):
    """The run is no longer waiting for this approval (e.g. it was cancelled)."""

    code = "run_not_waiting"


class RunAlreadyTerminalError(ServiceError):
    """The run already finished; it can be neither cancelled nor updated."""

    code = "run_already_terminal"


class InvalidConfigOverridesError(ServiceError):
    """Per-run overrides that are not allowed, e.g. raising an operator limit (422)."""

    code = "invalid_config_overrides"
