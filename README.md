# Agent Harness

Implemented with Claude Code from a design spec I wrote; architecture, safety design and trade-off decisions are my own.

A hand-written LLM ↔ tool loop for an operations assistant. You give it an objective ("customers report failed checkouts: investigate, and open an incident if warranted"); it searches a knowledge base, checks service status and, **only after a human approves the exact call**, creates an incident. Every step is persisted and traced.

- **Own loop, no framework.** `AgentLoop.advance()` in [`src/harness/core/loop.py`](src/harness/core/loop.py) is the whole agent loop.
- **Validated tools.** Each tool has a pydantic input and output model; the model sees errors as structured observations and can recover.
- **Persisted, re-entrant runs.** A run pauses for approval, returns, and resumes later from the database, from any request.
- **Approval gate in code.** `create_incident` runs only after a stored approval, with the stored arguments, under an idempotency key the model never sees.
- **Failure handling and limits.** Retries with capped exponential backoff and jitter for transient failures only; step, active-time, repeated-call and error limits.
- **Trace per run.** Every decision is an event, in the database (`GET /runs/{id}/trace`) and as a JSON log line.

The design report is in [`docs/report.pdf`](docs/report.pdf); core code excerpts are in [`docs/snippets.md`](docs/snippets.md).

## Quick start (no API key needed)

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), GNU make. Docker is optional (image, report).

```bash
make install     # uv sync
make demo        # scripted checkout_outage run; you approve or reject the incident
make demo-auto   # the same, approving automatically
make serve       # REST API on http://127.0.0.1:8000 (OpenAPI docs at /docs)
```

With the default `LLM_PROVIDER=scripted`, **every run replays the checkout_outage demo regardless of its objective**. `/health` says so, and the scripted final answer ends with a line saying it is a scripted demo. Use a real model (below) to run your own objectives.

## Using a real model

Any OpenAI-compatible endpoint works. Put the settings in `.env` (see [`.env.example`](.env.example)) or export them.

**Ollama** (local):

```bash
ollama pull qwen2.5:7b
LLM_PROVIDER=openai_compat
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama            # any non-empty value
LLM_MODEL=qwen2.5:7b
```

**OpenAI**:

```bash
LLM_PROVIDER=openai_compat
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

Then `uv run harness run "Is auth-service healthy?"` or `make serve`. `make test-live` runs every scenario in `data/scenarios.json` against the configured model and prints each run's tool sequence and trace.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status, provider, model |
| `POST /runs` | `{"objective", "config_overrides"?, "wait"?}` → `202`; with `wait: true` blocks until the run finishes or waits for approval (≤ 60 s) → `200` |
| `GET /runs`, `GET /runs/{id}` | Runs; `pending_approval` is set while waiting |
| `GET /runs/{id}/messages`, `GET /runs/{id}/trace` | History and trace events |
| `POST /runs/{id}/approvals/{approval_id}` | `{"decision": "approve" \| "reject", "reason"?, "decided_by"?}` |
| `POST /runs/{id}/cancel` | Cancel an unfinished run |
| `GET /incidents` | Contents of the mock incident system |

```bash
curl -s -X POST localhost:8000/runs -H 'content-type: application/json' \
  -d '{"objective": "Customers report failed checkouts. Investigate.", "wait": true}'
# → status "waiting_approval", pending_approval.id = <approval_id>
curl -s -X POST localhost:8000/runs/<run_id>/approvals/<approval_id> \
  -H 'content-type: application/json' -d '{"decision": "approve", "decided_by": "me"}'
curl -s localhost:8000/runs/<run_id>/trace
```

Errors are `{"error": {"code", "message"}}`: `404 run_not_found | approval_not_found`, `409 approval_not_pending | run_not_waiting | run_already_terminal`, `422 validation_error | invalid_config_overrides`, `500 internal_error` (never a traceback).

[`docs/postman_collection.json`](docs/postman_collection.json) runs the whole flow against `make serve`, including failure demos that inject faults per run.

## Behaviour worth knowing

- **Approvals are call-scoped.** An approval covers one tool call and its stored arguments; the harness executes exactly those. A changed call needs a new approval. If the model proposes a call the human already rejected in the same run, it is refused in code without asking again.
- **Per-run overrides may only lower limits.** `config_overrides.max_steps` / `max_run_seconds` can reduce the operator's env limits for one run, never raise them (the API has no authentication). Asking for more returns `422 invalid_config_overrides`.
- **Cancellation is cooperative.** A cancel takes effect at the next tick. A tool call already in flight finishes and is recorded, so **an already-approved `create_incident` still completes if the cancel lands while it is executing**. Its trace events can appear after the cancel's `run_finished`.
- **Waiting for approval is not active time.** `MAX_RUN_SECONDS` counts only time the harness spends working on the run.
- **Shutdown.** On shutdown, runs still being advanced are marked `FAILED` / `internal_error` ("shutdown"); runs waiting for approval stay resumable.

## Testing

```bash
make check       # ruff + mypy (strict) + pytest with coverage (≥ 85 %)
make test-live   # optional: scenarios against a real model
```

Tests never call a real LLM or the network: `ScriptedLLM` replays scripted responses and errors, and `FakeClock` makes backoff and time limits instant. The suites cover success, tool failures (flaky, permanent, timeout, timeout after commit), malformed LLM output, approvals (approve, reject, prompt injection, decide/cancel races) and execution limits.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `scripted` | `scripted` (no key) or `openai_compat` |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint (Ollama: `http://localhost:11434/v1`) |
| `LLM_API_KEY` | empty | Required for `openai_compat` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `LLM_TIMEOUT_SECONDS` / `LLM_MAX_RETRIES` | `60` / `2` | Per-call timeout; retries of retryable LLM errors |
| `LLM_SEND_PARALLEL_TOOL_CALLS_FLAG` | `true` | Send `parallel_tool_calls=false` |
| `MAX_STEPS` / `MAX_RUN_SECONDS` | `12` / `120` | Step and active-time limits (ceilings for per-run overrides) |
| `MAX_PARSE_RETRIES` / `MAX_CONSECUTIVE_TOOL_ERRORS` / `REPEAT_CALL_LIMIT` | `2` / `3` / `3` | Malformed-output, tool-error and repeated-call limits |
| `TOOL_MAX_RETRIES` / `OBSERVATION_MAX_CHARS` | `2` / `4000` | Tool retries; observation size cap |
| `DATABASE_URL` / `DATA_DIR` | `sqlite+aiosqlite:///./harness.db` / `./data` | Storage and mock dataset |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `json` | Logging |
| `MOCK_FAULTS` / `ALLOW_FAULT_OVERRIDES` / `MOCK_SEED` | `{}` / `true` / `42` | Fault injection for the mock tools; jitter seed |

## Project layout

```
src/harness/
  core/        loop.py (the agent loop), parser, guardrails, observations
  tools/       spec, registry, executor, faults, and the three mock tools
  llm/         base types, scripted clients, openai_compat client, prompts
  services/    RunService (locks, background runs, cancel, shutdown), ApprovalService
  store/       SQLAlchemy tables, repository, engine setup
  observability/  structlog setup, tracer
  api/         FastAPI app, routes, schemas
  cli.py       harness run | demo | trace | serve
  wiring.py    builds the object graph
  config.py    settings and per-run config
data/          knowledge base (markdown), services.json, scenarios.json
tests/         unit/, integration/, api/, live/
docs/          report (HTML source + PDF), snippets.md, postman_collection.json
```

`make report` renders `docs/report.pdf` from `docs/report/report.html` inside Docker, so it works the same on Windows and Linux.

## Limitations

Single process (in-process locks, SQLite); each repository write is its own transaction, so a tick is not atomic across a crash; no approval expiry; no approver authentication; cooperative cancellation; one tool call per step; basic prompt-injection defence (the approval gate limits the impact). See section 13 of the report.
