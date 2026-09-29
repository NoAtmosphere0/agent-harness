import pytest
from pydantic import BaseModel, Field, ValidationError

from harness.domain.errors import (
    LLMProviderError,
    LLMTimeoutError,
    ToolArgumentsError,
    ToolError,
    ToolNotFoundError,
    ToolOutputValidationError,
    ToolPermanentError,
    ToolTimeoutError,
    ToolTransientError,
    describe_validation_error,
)
from harness.domain.models import (
    STATUS_FOR_REASON,
    TERMINAL_STATUSES,
    TerminationReason,
    ToolResult,
)


@pytest.mark.parametrize(
    ("error_cls", "retryable"),
    [
        (ToolNotFoundError, False),
        (ToolArgumentsError, False),
        (ToolTimeoutError, True),
        (ToolTransientError, True),
        (ToolPermanentError, False),
        (ToolOutputValidationError, False),
    ],
)
def test_tool_error_retryable_flag_matches_taxonomy(error_cls: type[ToolError], retryable: bool):
    error = error_cls("some_tool", "boom")

    assert error.retryable is retryable
    assert error.error_type == error_cls.__name__
    assert error.tool_name == "some_tool"


def test_llm_errors_carry_retryable():
    assert LLMTimeoutError("slow").retryable is True
    assert LLMProviderError("bad request", retryable=False).retryable is False


def test_every_termination_reason_maps_to_a_terminal_status():
    assert set(STATUS_FOR_REASON) == set(TerminationReason)
    assert set(STATUS_FOR_REASON.values()) <= TERMINAL_STATUSES


def test_tool_result_failed_exposes_error_info():
    result = ToolResult.failed(ToolTransientError("t", "503"), attempts=3, duration_ms=12)

    assert result.ok is False
    assert result.data is None
    assert result.error is not None
    assert result.error.model_dump() == {
        "type": "ToolTransientError",
        "message": "503",
        "retryable": True,
    }


def test_tool_result_rejects_ok_without_data():
    with pytest.raises(ValidationError, match="ok results carry data"):
        ToolResult(tool_name="t", ok=True, attempts=1, duration_ms=0)


class _Many(BaseModel):
    a: int
    b: int
    c: str = Field(min_length=3)


def test_describe_validation_error_lists_fields_and_caps_count():
    with pytest.raises(ValidationError) as exc_info:
        _Many.model_validate({"a": "x", "b": "y", "c": ""})

    text = describe_validation_error(exc_info.value, max_errors=2)

    assert text.startswith("a: Input should be a valid integer")
    assert "b: " in text
    assert text.endswith("(+1 more)")
