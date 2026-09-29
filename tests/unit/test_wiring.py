import pytest
from pydantic import SecretStr

from harness.config import Settings
from harness.domain.errors import HarnessError
from harness.llm.openai_compat import OpenAICompatLLM
from harness.llm.scripted import ReplayLLM
from harness.wiring import build_llm


def test_wiring_scripted_provider_replays_demo_script():
    assert isinstance(build_llm(Settings(_env_file=None)), ReplayLLM)


def test_wiring_openai_compat_requires_api_key():
    with pytest.raises(HarnessError, match="LLM_API_KEY is required"):
        build_llm(Settings(_env_file=None, llm_provider="openai_compat"))


def test_wiring_openai_compat_uses_configured_model():
    settings = Settings(
        _env_file=None,
        llm_provider="openai_compat",
        llm_base_url="http://localhost:11434/v1",
        llm_api_key=SecretStr("ollama"),
        llm_model="qwen2.5:7b",
    )

    llm = build_llm(settings)

    assert isinstance(llm, OpenAICompatLLM)
    assert llm.model_name == "qwen2.5:7b"
