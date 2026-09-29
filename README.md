# Agent Harness

A hand-written LLM ↔ tool loop for an operations assistant. It uses three mock tools (knowledge base search, service status, incident creation), validates tool input and output, persists run state, handles failures and execution limits, requires human approval before `create_incident` runs, and records a trace for every run.

> **Status:** core, API and CLI are in place; the real-LLM provider and full documentation come in the final phase. See `PLAN.md` for the specification.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- GNU make
- Docker (optional)

## Quick start (no API key needed)

```bash
make install     # uv sync (runtime + dev dependencies)
make demo        # replay the checkout_outage scenario; you approve or reject the incident
make serve       # REST API on http://127.0.0.1:8000 (docs at /docs)
```

`docs/postman_collection.json` walks through the API against `make serve`.

## Development

```bash
make check       # ruff + mypy + pytest with coverage
make format      # auto-format and apply safe lint fixes
```

Configuration comes from environment variables; see `.env.example`.
