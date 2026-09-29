.PHONY: install lint format typecheck test test-live check serve demo demo-auto report

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

test:
	uv run pytest -m "not live" --cov --cov-report=term-missing

# Optional: runs data/scenarios.json against the configured real model.
test-live:
	uv run pytest -m live -q

check: lint typecheck test

serve:
	uv run harness serve

demo:
	uv run harness demo

demo-auto:
	uv run harness demo --auto-approve

# Rendered in Docker (Pango + WeasyPrint) so it works the same on every host OS.
# MSYS_NO_PATHCONV stops Git Bash on Windows from rewriting the container paths.
report: export MSYS_NO_PATHCONV := 1
report:
	docker build -q -t agent-harness-report docs/report
	docker run --rm -v "$(CURDIR)/docs:/docs" agent-harness-report \
		/docs/report/report.html /docs/report.pdf -s /docs/report/style.css
