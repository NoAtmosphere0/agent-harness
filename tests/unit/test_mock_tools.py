"""Behaviour of the three mock tools' handlers and their backing data."""

from pathlib import Path

import pytest

from harness.domain.errors import ToolPermanentError
from harness.tools.incidents import CreateIncidentInput, make_create_incident_tool
from harness.tools.knowledge_base import (
    SNIPPET_MAX_CHARS,
    KBDocument,
    KnowledgeBase,
    SearchKnowledgeBaseInput,
    make_search_knowledge_base_tool,
    parse_document,
)
from harness.tools.service_status import (
    GetServiceStatusInput,
    ServiceCatalog,
    make_get_service_status_tool,
)
from harness.tools.spec import RetryPolicy, ToolContext
from tests.fakes import InMemoryIncidentStore


def _ctx(idempotency_key: str | None = None) -> ToolContext:
    return ToolContext(
        run_id="run-1", step=1, idempotency_key=idempotency_key, deadline_remaining_s=60
    )


@pytest.fixture
def kb(data_dir: Path) -> KnowledgeBase:
    return KnowledgeBase.from_dir(data_dir / "kb")


# ------------------------------------------------------------- knowledge base


@pytest.mark.parametrize("query", ["severity matrix", "incident severity levels"])
def test_kb_search_finds_severity_matrix(kb: KnowledgeBase, query: str):
    hits = kb.search(query, top_k=3)

    assert hits[0].doc_id == "severity-matrix"
    assert "SEV1" in hits[0].snippet


def test_kb_search_ranks_severity_matrix_for_triage_question(kb: KnowledgeBase):
    hits = kb.search("what severity for a checkout outage", top_k=3)

    assert "severity-matrix" in [h.doc_id for h in hits]


def test_kb_search_finds_runbook_by_topic(kb: KnowledgeBase):
    hits = kb.search("database connection pool exhaustion", top_k=1)

    assert [h.doc_id for h in hits] == ["runbook-db-connection-pool-exhaustion"]


def test_kb_search_respects_top_k_and_snippet_limit(kb: KnowledgeBase):
    hits = kb.search("payment gateway checkout outage runbook", top_k=2)

    assert len(hits) == 2
    assert all(len(h.snippet) <= SNIPPET_MAX_CHARS for h in hits)
    assert hits[0].score >= hits[1].score


def test_kb_search_returns_empty_when_nothing_matches(kb: KnowledgeBase):
    assert kb.search("zyzzyva", top_k=3) == []
    assert kb.search("what is the", top_k=3) == []


def test_kb_snippet_truncated_to_limit():
    long_paragraph = "outage " * 200
    kb = KnowledgeBase(
        [
            KBDocument(doc_id="long", title="t", services=[], tags=[], body=long_paragraph),
            KBDocument(doc_id="other", title="t", services=[], tags=[], body="unrelated"),
        ]
    )

    (hit,) = kb.search("outage", top_k=1)

    assert len(hit.snippet) == SNIPPET_MAX_CHARS
    assert hit.snippet.endswith("…")


def test_kb_parse_document_reads_front_matter():
    doc = parse_document(
        '---\nid: x\ntitle: "A: B"\nservices: [a, b]\ntags: []\n---\n# Body\n\ntext\n', "x.md"
    )

    assert doc == KBDocument(
        doc_id="x", title="A: B", services=["a", "b"], tags=[], body="# Body\n\ntext"
    )


@pytest.mark.parametrize("text", ["no front matter", "---\njust a line\n---\nbody"])
def test_kb_parse_document_rejects_malformed_file(text: str):
    with pytest.raises(ValueError, match=r"bad\.md"):
        parse_document(text, "bad.md")


def test_kb_rejects_empty_corpus(tmp_path: Path):
    with pytest.raises(ValueError, match="empty"):
        KnowledgeBase.from_dir(tmp_path)


async def test_kb_tool_handler_returns_results(kb: KnowledgeBase):
    spec = make_search_knowledge_base_tool(kb, RetryPolicy())

    out = await spec.handler(SearchKnowledgeBaseInput(query="maintenance window"), _ctx())

    assert out["results"][0]["doc_id"] == "maintenance-windows"


# ------------------------------------------------------------- service status


@pytest.fixture
def catalog(data_dir: Path) -> ServiceCatalog:
    return ServiceCatalog.from_file(data_dir / "services.json")


async def test_service_status_returns_known_service(catalog: ServiceCatalog):
    spec = make_get_service_status_tool(catalog, RetryPolicy())

    out = await spec.handler(GetServiceStatusInput(service_name="checkout-api"), _ctx())

    assert out["status"] == "degraded"
    assert out["error_rate_pct"] == 12.5
    assert out["dependencies"] == ["payment-gateway", "inventory-service"]


async def test_service_status_unknown_service_lists_known_ones(catalog: ServiceCatalog):
    spec = make_get_service_status_tool(catalog, RetryPolicy())

    with pytest.raises(ToolPermanentError) as exc_info:
        await spec.handler(GetServiceStatusInput(service_name="billing-service"), _ctx())

    assert "unknown service 'billing-service'" in exc_info.value.message
    for name in catalog.names():
        assert name in exc_info.value.message


# ------------------------------------------------------------- incidents

_ARGS = CreateIncidentInput(
    title="payment-gateway: full outage",
    description="Payment gateway error rate 96.8%; checkout failing for all customers.",
    severity="SEV1",
)


async def test_incident_refuses_without_idempotency_key(incident_store: InMemoryIncidentStore):
    spec = make_create_incident_tool(incident_store, RetryPolicy())

    with pytest.raises(ToolPermanentError, match="without approval"):
        await spec.handler(_ARGS, _ctx(idempotency_key=None))

    assert incident_store.records == {}


async def test_incident_dedup_by_idempotency_key(incident_store: InMemoryIncidentStore):
    spec = make_create_incident_tool(incident_store, RetryPolicy())

    first = await spec.handler(_ARGS, _ctx("run-1:appr-1"))
    repeat = await spec.handler(_ARGS, _ctx("run-1:appr-1"))
    other = await spec.handler(_ARGS, _ctx("run-1:appr-2"))

    assert first["deduplicated"] is False
    assert repeat == {**first, "deduplicated": True}
    assert other["incident_id"] != first["incident_id"]
    assert len(incident_store.records) == 2


def test_incident_tool_requires_approval(incident_store: InMemoryIncidentStore):
    assert make_create_incident_tool(incident_store, RetryPolicy()).requires_approval is True
