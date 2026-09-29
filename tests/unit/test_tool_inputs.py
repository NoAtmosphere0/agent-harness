"""Every input constraint of the three tools (PLAN §7), valid boundaries included."""

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from harness.tools.incidents import CreateIncidentInput
from harness.tools.knowledge_base import SearchKnowledgeBaseInput
from harness.tools.service_status import GetServiceStatusInput

VALID_INCIDENT = {
    "title": "checkout-api: 5xx errors",
    "description": "Checkout error rate is 12.5%, caused by payment-gateway outage.",
    "severity": "SEV2",
}

INVALID_CASES: list[tuple[type[BaseModel], dict[str, Any]]] = [
    (SearchKnowledgeBaseInput, {"query": ""}),
    (SearchKnowledgeBaseInput, {"query": "   "}),
    (SearchKnowledgeBaseInput, {"query": "x" * 501}),
    (SearchKnowledgeBaseInput, {"query": "ok", "top_k": 0}),
    (SearchKnowledgeBaseInput, {"query": "ok", "top_k": 11}),
    (SearchKnowledgeBaseInput, {"query": "ok", "limit": 3}),
    (SearchKnowledgeBaseInput, {}),
    (GetServiceStatusInput, {"service_name": ""}),
    (GetServiceStatusInput, {"service_name": "Checkout-API"}),
    (GetServiceStatusInput, {"service_name": "-checkout"}),
    (GetServiceStatusInput, {"service_name": "checkout api"}),
    (GetServiceStatusInput, {"service_name": "a" * 65}),
    (GetServiceStatusInput, {"service_name": "auth-service", "region": "eu"}),
    (CreateIncidentInput, {**VALID_INCIDENT, "title": "x" * 9}),
    (CreateIncidentInput, {**VALID_INCIDENT, "title": "x" * 121}),
    (CreateIncidentInput, {**VALID_INCIDENT, "description": "x" * 19}),
    (CreateIncidentInput, {**VALID_INCIDENT, "description": "x" * 2001}),
    (CreateIncidentInput, {**VALID_INCIDENT, "severity": "SEV5"}),
    (CreateIncidentInput, {**VALID_INCIDENT, "severity": "sev1"}),
    (CreateIncidentInput, {**VALID_INCIDENT, "idempotency_key": "model-chosen"}),
]

VALID_CASES: list[tuple[type[BaseModel], dict[str, Any]]] = [
    (SearchKnowledgeBaseInput, {"query": "x"}),
    (SearchKnowledgeBaseInput, {"query": "x" * 500, "top_k": 1}),
    (SearchKnowledgeBaseInput, {"query": "severity", "top_k": 10}),
    (GetServiceStatusInput, {"service_name": "a"}),
    (GetServiceStatusInput, {"service_name": "a" * 64}),
    (GetServiceStatusInput, {"service_name": "checkout-api"}),
    (CreateIncidentInput, {**VALID_INCIDENT, "title": "x" * 10, "description": "x" * 20}),
    (CreateIncidentInput, {**VALID_INCIDENT, "title": "x" * 120, "description": "x" * 2000}),
]


@pytest.mark.parametrize(("model", "payload"), INVALID_CASES)
def test_tool_input_rejects_constraint_violation(model: type[BaseModel], payload: dict[str, Any]):
    with pytest.raises(ValidationError):
        model.model_validate(payload)


@pytest.mark.parametrize(("model", "payload"), VALID_CASES)
def test_tool_input_accepts_boundary_values(model: type[BaseModel], payload: dict[str, Any]):
    model.model_validate(payload)


def test_search_input_defaults_top_k_to_three():
    assert SearchKnowledgeBaseInput(query="severity").top_k == 3
