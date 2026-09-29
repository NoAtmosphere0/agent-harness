from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.config import Settings, get_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from the developer's shell environment."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


def test_settings_defaults_match_plan():
    settings = Settings(_env_file=None)

    assert settings.llm_provider == "scripted"
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.llm_timeout_seconds == 60
    assert settings.llm_max_retries == 2
    assert settings.max_steps == 12
    assert settings.max_run_seconds == 120
    assert settings.max_parse_retries == 2
    assert settings.max_consecutive_tool_errors == 3
    assert settings.repeat_call_limit == 3
    assert settings.tool_max_retries == 2
    assert settings.observation_max_chars == 4000
    assert settings.database_url == "sqlite+aiosqlite:///./harness.db"
    assert settings.data_dir == Path("./data")
    assert settings.mock_faults == {}
    assert settings.mock_seed == 42


def test_settings_read_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAX_STEPS", "5")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compat")
    monkeypatch.setenv("MOCK_FAULTS", '{"get_service_status": {"mode": "flaky", "fail_times": 2}}')

    settings = Settings(_env_file=None)

    assert settings.max_steps == 5
    assert settings.llm_provider == "openai_compat"
    assert settings.mock_faults == {"get_service_status": {"mode": "flaky", "fail_times": 2}}


def test_settings_reject_invalid_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAX_STEPS", "0")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_api_key_is_masked_in_repr(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-very-secret")

    settings = Settings(_env_file=None)

    assert "sk-very-secret" not in repr(settings)
    assert settings.llm_api_key.get_secret_value() == "sk-very-secret"


def test_get_settings_is_cached():
    get_settings.cache_clear()

    assert get_settings() is get_settings()

    get_settings.cache_clear()
