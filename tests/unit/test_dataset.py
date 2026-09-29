"""The mock dataset is consistent with the tools that read it (PLAN §15)."""

import json
from pathlib import Path

from harness.tools.knowledge_base import parse_document
from harness.tools.service_status import ServiceCatalog

TOOL_NAMES = {"search_knowledge_base", "get_service_status", "create_incident"}


def test_dataset_kb_ids_match_file_names(data_dir: Path):
    paths = sorted((data_dir / "kb").glob("*.md"))

    docs = [parse_document(p.read_text(encoding="utf-8"), p.name) for p in paths]

    assert len(docs) == 8
    assert [d.doc_id for d in docs] == [p.stem for p in paths]
    assert all(d.title for d in docs)


def test_dataset_vendor_note_contains_injection_payload(data_dir: Path):
    text = (data_dir / "kb" / "vendor-note-untrusted.md").read_text(encoding="utf-8")

    assert "ignore previous instructions" in text


def test_dataset_services_match_plan(data_dir: Path):
    catalog = ServiceCatalog.from_file(data_dir / "services.json")

    statuses = {n: s.status for n in catalog.names() if (s := catalog.get(n))}
    assert statuses == {
        "auth-service": "operational",
        "checkout-api": "degraded",
        "inventory-service": "operational",
        "notification-service": "maintenance",
        "payment-gateway": "outage",
        "search-api": "degraded",
    }


def test_dataset_scenarios_reference_real_tools(data_dir: Path):
    scenarios = json.loads((data_dir / "scenarios.json").read_text(encoding="utf-8"))

    assert [s["id"] for s in scenarios] == [
        "status_check",
        "checkout_outage",
        "runbook_lookup",
        "unknown_service",
        "maintenance_no_incident",
    ]
    for scenario in scenarios:
        assert set(scenario["expected_tools"]) <= TOOL_NAMES
        assert scenario["expects_incident"] == ("create_incident" in scenario["expected_tools"])
