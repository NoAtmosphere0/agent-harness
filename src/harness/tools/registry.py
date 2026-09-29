"""Name → ToolSpec lookup, and the factory that wires the three mock tools."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from harness.domain.errors import ToolNotFoundError
from harness.tools.incidents import IncidentStore, make_create_incident_tool
from harness.tools.knowledge_base import KnowledgeBase, make_search_knowledge_base_tool
from harness.tools.service_status import ServiceCatalog, make_get_service_status_tool
from harness.tools.spec import RetryPolicy, ToolSpec


class ToolRegistry:
    def __init__(self, specs: Iterable[ToolSpec]) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            if spec.name in self._specs:
                raise ValueError(f"duplicate tool name: {spec.name}")
            self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise ToolNotFoundError(
                name, f"unknown tool; available tools: {', '.join(self.names())}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._specs)

    def llm_tools(self) -> list[dict[str, Any]]:
        """Function definitions sent to the model. Built from input models only, so
        harness-side context such as the idempotency key is never exposed (P6)."""
        return [self._specs[name].llm_schema() for name in self.names()]


def build_registry(
    *, data_dir: Path, incident_store: IncidentStore, tool_max_retries: int
) -> ToolRegistry:
    """The production tool set, loaded from ``data_dir``."""
    retry = RetryPolicy(max_retries=tool_max_retries)
    return ToolRegistry(
        [
            make_search_knowledge_base_tool(KnowledgeBase.from_dir(data_dir / "kb"), retry),
            make_get_service_status_tool(
                ServiceCatalog.from_file(data_dir / "services.json"), retry
            ),
            make_create_incident_tool(incident_store, retry),
        ]
    )
