import json

import pytest

from harness.observability.logging import LOG_VALUE_MAX_CHARS, configure_logging, get_logger


def _last_json_line(err: str) -> dict[str, object]:
    return json.loads(err.strip().splitlines()[-1])  # type: ignore[no-any-return]


def test_logging_json_line_has_standard_fields(capsys: pytest.CaptureFixture[str]):
    configure_logging("INFO", "json")

    get_logger(component="test").info("hello", run_id="r1", step=2)

    line = _last_json_line(capsys.readouterr().err)
    assert line["event"] == "hello"
    assert line["level"] == "info"
    assert line["component"] == "test"
    assert line["run_id"] == "r1"
    assert line["step"] == 2
    assert "timestamp" in line


def test_logging_truncates_long_strings_including_nested(capsys: pytest.CaptureFixture[str]):
    configure_logging("INFO", "json")
    long = "x" * (LOG_VALUE_MAX_CHARS + 20)

    get_logger().info("msg", content=long, payload={"inner": [long]})

    line = _last_json_line(capsys.readouterr().err)
    expected = "x" * LOG_VALUE_MAX_CHARS + "[truncated 20 chars]"
    assert line["content"] == expected
    assert line["payload"] == {"inner": [expected]}


def test_logging_filters_below_configured_level(capsys: pytest.CaptureFixture[str]):
    configure_logging("WARNING", "json")

    get_logger().info("hidden")

    assert capsys.readouterr().err == ""


def test_logging_console_format_is_not_json(capsys: pytest.CaptureFixture[str]):
    configure_logging("INFO", "console")

    get_logger().info("readable")

    err = capsys.readouterr().err
    assert "readable" in err
    assert not err.lstrip().startswith("{")
