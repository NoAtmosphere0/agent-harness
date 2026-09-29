"""``get_service_status``: current health of one service from ``data/services.json`` (PLAN §7.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter

from harness.domain.errors import ToolPermanentError
from harness.tools.spec import RetryPolicy, ToolContext, ToolSpec

ServiceName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")]


class GetServiceStatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service_name: ServiceName = Field(description="Service id, lower-case, e.g. 'checkout-api'.")


class ServiceStatus(BaseModel):
    """One service's health; both the catalog record and the tool's output."""

    service_name: ServiceName
    status: Literal["operational", "degraded", "outage", "maintenance"]
    error_rate_pct: float = Field(ge=0, le=100)
    p95_latency_ms: float = Field(ge=0)
    last_checked: AwareDatetime
    dependencies: list[str]


_CATALOG_ADAPTER = TypeAdapter(list[ServiceStatus])


class ServiceCatalog:
    """The mock monitoring system: a fixed snapshot of service health."""

    def __init__(self, services: list[ServiceStatus]) -> None:
        self._services = {s.service_name: s for s in services}

    @classmethod
    def from_file(cls, path: Path) -> ServiceCatalog:
        return cls(_CATALOG_ADAPTER.validate_json(path.read_bytes()))

    def names(self) -> list[str]:
        return sorted(self._services)

    def get(self, service_name: str) -> ServiceStatus | None:
        return self._services.get(service_name)


def make_get_service_status_tool(catalog: ServiceCatalog, retry: RetryPolicy) -> ToolSpec:
    async def handler(args: GetServiceStatusInput, ctx: ToolContext) -> dict[str, Any]:
        status = catalog.get(args.service_name)
        if status is None:
            # Listing the known names lets the model correct itself (P7).
            raise ToolPermanentError(
                "get_service_status",
                f"unknown service '{args.service_name}'; "
                f"known services: {', '.join(catalog.names())}",
            )
        return status.model_dump(mode="json")

    return ToolSpec(
        name="get_service_status",
        description=(
            "Get the current status, error rate, p95 latency and dependencies of one service."
        ),
        input_model=GetServiceStatusInput,
        output_model=ServiceStatus,
        handler=handler,
        timeout_s=3.0,
        retry=retry,
    )
