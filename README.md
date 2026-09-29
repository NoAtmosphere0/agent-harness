# Agent Harness

A hand-written LLM ↔ tool loop for an operations assistant. It uses three mock tools (knowledge base search, service status, incident creation), validates tool input and output, persists run state, handles failures and execution limits, requires human approval before `create_incident` runs, and records a trace for every run.

> **Status:** scaffold only (Phase 0). See `PLAN.md` for the full specification.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- GNU make
- Docker (optional)

## Development

```bash
make install     # uv sync (runtime + dev dependencies)
make check       # ruff + mypy + pytest with coverage
make format      # auto-format and apply safe lint fixes
```

Configuration comes from environment variables; see `.env.example`.
