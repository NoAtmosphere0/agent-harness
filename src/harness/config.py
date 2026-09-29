"""Process configuration, loaded from environment variables (and ``.env``).

Every tunable in PLAN §14 lives here so there is one place to read the effective
defaults. Per-run overrides (``config_overrides``) are applied on top of them by
``build_run_config`` and snapshotted onto the run as a ``RunConfig``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from harness.domain.errors import InvalidConfigOverridesError
from harness.tools.faults import Fault, parse_faults

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
LogFormat = Literal["json", "console"]
LLMProvider = Literal["scripted", "openai_compat"]


class Settings(BaseSettings):
    """Harness settings. Field names map to upper-case env vars (``max_steps`` → ``MAX_STEPS``)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: LLMProvider = "scripted"
    llm_base_url: str = "https://api.openai.com/v1"
    # SecretStr keeps the key out of repr() and therefore out of logs.
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    llm_send_parallel_tool_calls_flag: bool = True

    # Execution limits (P8)
    max_steps: int = Field(default=12, ge=1)
    max_run_seconds: float = Field(default=120.0, gt=0)
    max_parse_retries: int = Field(default=2, ge=0)
    max_consecutive_tool_errors: int = Field(default=3, ge=1)
    repeat_call_limit: int = Field(default=3, ge=1)

    # Tools
    tool_max_retries: int = Field(default=2, ge=0)
    observation_max_chars: int = Field(default=4000, ge=100)

    # Storage and data
    database_url: str = "sqlite+aiosqlite:///./harness.db"
    data_dir: Path = Path("./data")

    # Logging
    log_level: LogLevel = "INFO"
    log_format: LogFormat = "json"

    # Mock tools / fault injection (PLAN §7.4); the fault schema is validated by tools/faults.py.
    mock_faults: dict[str, Any] = Field(default_factory=dict)
    allow_fault_overrides: bool = True
    mock_seed: int = 42

    @field_validator("mock_faults")
    @classmethod
    def _check_faults(cls, value: dict[str, Any]) -> dict[str, Any]:
        parse_faults(value)  # fail at startup, not on the first run
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read once. Tests construct ``Settings`` directly instead."""
    return Settings()


# ---------------------------------------------------------------- per-run config

# Absolute caps (PLAN §8.7), applied even to the env values in case of operator error.
HARD_MAX_STEPS = 50
HARD_MAX_RUN_SECONDS = 600.0


class ConfigOverrides(BaseModel):
    """What a caller may change for a single run (``POST /runs`` ``config_overrides``)."""

    model_config = ConfigDict(extra="forbid")

    max_steps: int | None = Field(default=None, ge=1)
    max_run_seconds: float | None = Field(default=None, gt=0)
    faults: dict[str, Any] | None = None


class RunConfig(BaseModel):
    """Effective settings for one run, stored on the run row (PLAN §10).

    A run can pause for approval and resume much later, possibly after a restart
    with different env vars. Reading limits from this snapshot instead of from
    ``Settings`` keeps the run's behaviour fixed from start to finish (P2).
    """

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    max_steps: int
    max_run_seconds: float
    max_parse_retries: int
    max_consecutive_tool_errors: int
    repeat_call_limit: int
    llm_timeout_seconds: float
    llm_max_retries: int
    observation_max_chars: int
    faults: dict[str, Fault]


def build_run_config(
    settings: Settings, model_name: str, overrides: ConfigOverrides | None = None
) -> RunConfig:
    """Apply per-run overrides to the operator's settings.

    P8: the env values are the ceiling. The API has no authentication, so a caller
    may lower a limit for its own run but never raise one; trying to raises
    ``InvalidConfigOverridesError`` (422), as do fault overrides when disabled.
    """
    overrides = overrides or ConfigOverrides()
    max_steps = _lowered("max_steps", min(settings.max_steps, HARD_MAX_STEPS), overrides.max_steps)
    max_run_seconds = _lowered(
        "max_run_seconds",
        min(settings.max_run_seconds, HARD_MAX_RUN_SECONDS),
        overrides.max_run_seconds,
    )
    faults = parse_faults(settings.mock_faults)
    if overrides.faults is not None:
        if not settings.allow_fault_overrides:
            raise InvalidConfigOverridesError(
                "fault overrides are disabled (ALLOW_FAULT_OVERRIDES=false)"
            )
        try:
            faults = parse_faults(overrides.faults)
        except ValueError as exc:
            raise InvalidConfigOverridesError(f"invalid faults: {exc}") from exc
    return RunConfig(
        provider=settings.llm_provider,
        model=model_name,
        max_steps=int(max_steps),
        max_run_seconds=float(max_run_seconds),
        max_parse_retries=settings.max_parse_retries,
        max_consecutive_tool_errors=settings.max_consecutive_tool_errors,
        repeat_call_limit=settings.repeat_call_limit,
        llm_timeout_seconds=settings.llm_timeout_seconds,
        llm_max_retries=settings.llm_max_retries,
        observation_max_chars=settings.observation_max_chars,
        faults=faults,
    )


def _lowered(name: str, ceiling: float, requested: float | None) -> float:
    if requested is None:
        return ceiling
    if requested > ceiling:
        raise InvalidConfigOverridesError(
            f"{name} may only be lowered: requested {requested:g}, limit is {ceiling:g}"
        )
    return requested
