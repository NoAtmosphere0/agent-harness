import pytest

from harness.config import Settings
from harness.domain.errors import HarnessError
from harness.llm.scripted import ReplayLLM
from harness.wiring import build_llm


def test_wiring_scripted_provider_replays_demo_script():
    assert isinstance(build_llm(Settings(_env_file=None)), ReplayLLM)


def test_wiring_unknown_provider_fails_clearly():
    with pytest.raises(HarnessError, match="openai_compat"):
        build_llm(Settings(_env_file=None, llm_provider="openai_compat"))
