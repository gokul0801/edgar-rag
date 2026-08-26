.PHONY: install ingest run test lint eval

install:
	uv sync --extra dev

ingest:
	uv run python -m edgar_rag.ingest

run:
	uv run uvicorn edgar_rag.api:app --reload --port 8000

test:
	uv run pytest -q

lint:
	uv run ruff check src tests && uv run ruff format --check src tests

eval:
	uv run python -m edgar_rag.evals.run
