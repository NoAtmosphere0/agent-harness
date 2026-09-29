# syntax=docker/dockerfile:1

# ---- build: resolve the locked environment with uv ----
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first so they are cached independently of source changes.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ---- runtime: venv + mock dataset only ----
FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 1000 harness
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY data ./data
RUN chown harness:harness /app
USER harness

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data \
    DATABASE_URL=sqlite+aiosqlite:////app/harness.db

EXPOSE 8000
CMD ["uvicorn", "harness.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
