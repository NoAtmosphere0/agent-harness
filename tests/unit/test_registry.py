import json
from pathlib import Path

import pytest

from harness.domain.errors import ToolNotFoundError
from harness.tools.registry import ToolRegistry, build_registry
from tests.fakes import InMemoryIncidentStore


@pytest.fixture
def registry(data_dir: Path, incident_store: InMemoryIncidentStore) -> ToolRegistry:
    return build_registry(data_dir=data_dir, incident_store=incident_store, tool_max_retries=2)


def test_registry_holds_the_three_tools(registry: ToolRegistry):
    assert registry.names() == ["create_incident", "get_service_status", "search_knowledge_base"]
    assert registry.get("get_service_status").retry.max_retries == 2


def test_registry_unknown_tool_lists_available(registry: ToolRegistry):
    with pytest.raises(ToolNotFoundError, match="available tools: create_incident"):
        registry.get("delete_database")


def test_registry_rejects_duplicate_names(registry: ToolRegistry):
    spec = registry.get("get_service_status")

    with pytest.raises(ValueError, match="duplicate"):
        ToolRegistry([spec, spec])


def test_registry_llm_tools_never_expose_idempotency_key(registry: ToolRegistry):
    tools = registry.llm_tools()

    assert [t["function"]["name"] for t in tools] == registry.names()
    assert "idempotency" not in json.dumps(tools).lower()
    incident = next(t for t in tools if t["function"]["name"] == "create_incident")
    assert set(incident["function"]["parameters"]["properties"]) == {
        "title",
        "description",
        "severity",
    }
