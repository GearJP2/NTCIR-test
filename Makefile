.PHONY: dev worker test lint format ingest eval build

# ── Development ────────────────────────────────────────────────────────────────
dev:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

worker:
	arq workers.worker.WorkerSettings

# ── Infrastructure ─────────────────────────────────────────────────────────────
up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f app worker

# ── Quality ────────────────────────────────────────────────────────────────────
lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy app services storage workers

test:
	pytest tests/ -v --cov=app --cov=services --cov=storage --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# ── NTCIR Data Pipeline ────────────────────────────────────────────────────────
ingest:
	python scripts/ingest_corpus.py --corpus-dir data/sample/

build-index:
	python scripts/build_index.py

eval:
	python scripts/export_results.py && python -m evaluation.evaluator

# ── Docker ─────────────────────────────────────────────────────────────────────
build:
	docker compose build --no-cache
