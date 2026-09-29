"""Process configuration, loaded from environment variables (and ``.env``).

Every tunable in PLAN §14 lives here so there is one place to read the effective
defaults. Per-run overrides (``config_overrides``) are applied on top of these by
the run service and snapshotted onto the run, not written back here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read once. Tests construct ``Settings`` directly instead."""
    return Settings()
