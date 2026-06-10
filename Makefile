.PHONY: dev worker test lint format ingest eval build \
	ingest-activitynet build-activitynet-manifest eval-moments check-media-index search-moments inspect-castle

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

ingest-activitynet:
	HF_HUB_DISABLE_XET=1 python scripts/ingest_corpus.py \
		--manifest-path $(or $(MANIFEST),data/manifests/activitynet_dev50.jsonl) \
		--language en \
		--workers $(or $(WORKERS),1) \
		--media-id-source filename \
		$(if $(START_AT),--start-at-media-id $(START_AT),)

build-index:
	python scripts/train_and_index.py

index-dry-run:
	python scripts/train_and_index.py --max-items 5 --dry-run

eval:
	python scripts/export_results.py && python -m evaluation.evaluator

# ── Moment Search / Evaluation ───────────────────────────────────────────────
build-activitynet-manifest:
	python scripts/build_activitynet_manifest.py \
		--video-root $(or $(VIDEO_ROOT),data/activitynet/videos) \
		--output-path $(or $(MANIFEST),data/manifests/activitynet_dev50.jsonl)

eval-moments:
	python -m evaluation.moment_evaluator \
		$(or $(MANIFEST),data/manifests/activitynet_dev50.jsonl) \
		--profile-name $(or $(PROFILE),activitynet_visual_heavy)

check-media-index:
	python scripts/check_media_index.py \
		--media-id $(MEDIA_ID) \
		--sample-limit $(or $(SAMPLE_LIMIT),100)

list-indexed-media:
	python scripts/list_indexed_media.py \
		--sample-limit $(or $(SAMPLE_LIMIT),1000)

search-moments:
	python scripts/search_moments.py \
		--media-id $(MEDIA_ID) \
		--duration-sec $(DURATION_SEC) \
		--query "$(QUERY)" \
		--profile $(or $(PROFILE),activitynet_visual_heavy) \
		--top-k $(or $(TOP_K),10)

inspect-castle:
	python -m evaluation.castle_inspection \
		--media-id $(MEDIA_ID) \
		--duration-sec $(DURATION_SEC) \
		--queries-path $(or $(QUERIES),data/curated_queries/castle_smoke.jsonl) \
		--output-path $(or $(OUTPUT),data/inspection/castle_smoke_results.jsonl) \
		--profile-name $(or $(PROFILE),castle_lifelog_balanced) \
		--top-k $(or $(TOP_K),10)

# ── Docker ─────────────────────────────────────────────────────────────────────
build:
	docker compose build --no-cache
