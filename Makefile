.PHONY: install lint format typecheck test check serve demo report

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
	uv run pytest --cov --cov-report=term-missing

check: lint typecheck test

serve:
	uv run harness serve

demo:
	uv run harness demo

report:
	uv run weasyprint docs/report/report.html docs/report.pdf -s docs/report/style.css
