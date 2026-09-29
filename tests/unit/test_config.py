from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.config import (
    HARD_MAX_RUN_SECONDS,
    HARD_MAX_STEPS,
    ConfigOverrides,
    RunConfig,
    Settings,
    build_run_config,
    get_settings,
)
from harness.tools.faults import FlakyFault


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


# ------------------------------------------------------------------ per-run config


def test_run_config_defaults_come_from_settings():
    config = build_run_config(Settings(_env_file=None), "scripted")

    assert (config.max_steps, config.max_run_seconds, config.model) == (12, 120, "scripted")
    assert config.faults == {}


def test_run_config_overrides_are_clamped_to_hard_caps():
    overrides = ConfigOverrides(max_steps=99, max_run_seconds=5000)

    config = build_run_config(Settings(_env_file=None), "m", overrides)

    assert (config.max_steps, config.max_run_seconds) == (HARD_MAX_STEPS, HARD_MAX_RUN_SECONDS)


def test_run_config_fault_override_replaces_env_faults():
    settings = Settings(_env_file=None, mock_faults={"get_service_status": {"mode": "error"}})
    overrides = ConfigOverrides(faults={"search_knowledge_base": {"mode": "flaky"}})

    config = build_run_config(settings, "m", overrides)

    assert config.faults == {"search_knowledge_base": FlakyFault(mode="flaky")}


def test_run_config_rejects_fault_override_when_disabled():
    settings = Settings(_env_file=None, allow_fault_overrides=False)

    with pytest.raises(ValueError, match="disabled"):
        build_run_config(settings, "m", ConfigOverrides(faults={}))


def test_run_config_roundtrips_through_json():
    settings = Settings(
        _env_file=None, mock_faults={"create_incident": {"mode": "timeout_after_commit"}}
    )
    config = build_run_config(settings, "m")

    assert RunConfig.model_validate(config.model_dump(mode="json")) == config
