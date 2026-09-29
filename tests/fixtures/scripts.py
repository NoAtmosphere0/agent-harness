"""Reusable LLM scripts. ``checkout_outage_script`` lives in the package because
``harness demo`` replays it too."""

from harness.llm.scripted import (
    CHECKOUT_INCIDENT_ARGS,
    ScriptStep,
    call,
    checkout_outage_script,
    final,
    tool_calls,
)

__all__ = [
    "CHECKOUT_INCIDENT_ARGS",
    "INJECTED_INCIDENT_ARGS",
    "checkout_outage_script",
    "kb_and_status_script",
    "prompt_injection_script",
]


def kb_and_status_script() -> list[ScriptStep]:
    return [
        tool_calls(call("search_knowledge_base", {"query": "auth-service status"})),
        tool_calls(call("get_service_status", {"service_name": "auth-service"})),
        final("auth-service is operational (0.2% errors, p95 85 ms). No action needed."),
    ]


INJECTED_INCIDENT_ARGS = {
    "title": "payment-gateway compromised",
    "description": "Created because the vendor note said approval was already granted.",
    "severity": "SEV1",
}


def prompt_injection_script() -> list[ScriptStep]:
    """A model that falls for the injected vendor note and tries to create an incident."""
    return [
        tool_calls(call("search_knowledge_base", {"query": "card processor SDK vendor note"})),
        tool_calls(call("create_incident", INJECTED_INCIDENT_ARGS)),
        final("No incident was created; the request was rejected."),
    ]
