.PHONY: dev worker test lint format ingest eval build \
	ingest-activitynet build-activitynet-manifest eval-moments eval-activitynet-profile-sweep \
	summarize-activitynet-results compare-activitynet-results \
	summarize-activitynet-temporal-tradeoff \
	build-activitynet-paper-artifacts \
	check-activitynet-paper-artifacts \
	estimate-activitynet-ablation-costs \
	check-media-index search-moments inspect-castle

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
		$(if $(START_AT),--start-at-media-id $(START_AT)) \
		$(foreach MEDIA_ID,$(ONLY_MEDIA_ID),--only-media-id $(MEDIA_ID)) \
		$(foreach MODALITY,$(MODALITIES),--modality $(MODALITY)) \
		$(if $(KEYFRAME_INTERVAL_SEC),--keyframe-interval-sec $(KEYFRAME_INTERVAL_SEC)) \
		$(if $(SKIP_INDEXED),--skip-indexed)

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
		--profile-name $(or $(PROFILE),activitynet_visual_heavy) \
		--window-sec $(or $(WINDOW_SEC),10.0) \
		--stride-sec $(or $(STRIDE_SEC),5.0) \
		--summary-path $(or $(SUMMARY),data/evaluation/activitynet_dev50_summary.json) \
		--results-path $(or $(RESULTS),data/evaluation/activitynet_dev50_results.jsonl) \
		--query-csv-path $(or $(QUERY_CSV),data/evaluation/activitynet_dev50_queries.csv) \
		--report-path $(or $(REPORT),data/evaluation/activitynet_dev50_report.md)

eval-activitynet-profile-sweep:
	python scripts/sweep_activitynet_profiles.py \
		--manifest-path $(or $(MANIFEST),data/manifests/activitynet_dev200.jsonl) \
		--summary-path $(or $(SUMMARY),data/evaluation/activitynet_profile_sweep_summary.json) \
		--csv-path $(or $(CSV),data/evaluation/activitynet_profile_sweep_summary.csv) \
		--output-dir $(or $(OUTPUT_DIR),data/evaluation/profile_sweep) \
		$(foreach PROFILE_NAME,$(PROFILES),--profile $(PROFILE_NAME)) \
		$(if $(WRITE_DETAILS),--write-details)

summarize-activitynet-results:
	python scripts/summarize_evaluation_results.py \
		$(or $(SUMMARIES),data/evaluation/activitynet_dev200_visual_only_summary.json data/evaluation/activitynet_dev200_visual_heavy_summary.json data/evaluation/activitynet_profile_sweep_summary.json) \
		--csv-path $(or $(CSV),data/evaluation/activitynet_results_table.csv) \
		--markdown-path $(or $(MARKDOWN),data/evaluation/activitynet_results_table.md) \
		--latex-path $(or $(LATEX),data/evaluation/activitynet_results_table.tex) \
		--findings-path $(or $(FINDINGS),data/evaluation/activitynet_findings.md) \
		--baseline-profile $(or $(BASELINE_PROFILE),activitynet_visual_only)

summarize-activitynet-temporal-tradeoff:
	python scripts/summarize_activitynet_temporal_tradeoff.py \
		$(or $(SUMMARIES),data/evaluation/activitynet_dev200_visual_only_summary.json data/evaluation/activitynet_dev200_visual_only_w20_s10_summary.json) \
		--cost-path $(or $(COSTS),data/evaluation/activitynet_ablation_costs.json) \
		--csv-path $(or $(CSV),data/evaluation/activitynet_temporal_tradeoff.csv) \
		--markdown-path $(or $(MARKDOWN),data/evaluation/activitynet_temporal_tradeoff.md) \
		$(if $(LATEX),--latex-path $(LATEX))

compare-activitynet-results:
	python scripts/compare_moment_results.py \
		--baseline-results-path $(or $(BASELINE),data/evaluation/activitynet_dev200_visual_only_results.jsonl) \
		--candidate-results-path $(or $(CANDIDATE),data/evaluation/activitynet_dev200_visual_heavy_results.jsonl) \
		--csv-path $(or $(CSV),data/evaluation/activitynet_visual_only_vs_heavy_regressions.csv) \
		--markdown-path $(or $(MARKDOWN),data/evaluation/activitynet_visual_only_vs_heavy_regressions.md) \
		$(if $(JSON),--json-path $(JSON)) \
		--limit $(or $(LIMIT),50)

estimate-activitynet-ablation-costs:
	python scripts/estimate_activitynet_ablation_cost.py \
		--manifest-path $(or $(MANIFEST),data/manifests/activitynet_dev200.jsonl) \
		--output-csv $(or $(CSV),data/evaluation/activitynet_ablation_costs.csv) \
		--output-markdown $(or $(MARKDOWN),data/evaluation/activitynet_ablation_costs.md) \
		$(if $(JSON),--output-json $(JSON)) \
		$(foreach SETTING,$(WINDOW_STRIDES),--window-stride $(SETTING)) \
		$(foreach INTERVAL,$(KEYFRAME_INTERVALS),--keyframe-interval $(INTERVAL))

build-activitynet-paper-artifacts:
	$(MAKE) estimate-activitynet-ablation-costs JSON=data/evaluation/activitynet_ablation_costs.json
	$(MAKE) summarize-activitynet-results
	$(MAKE) summarize-activitynet-temporal-tradeoff LATEX=data/evaluation/activitynet_temporal_tradeoff.tex
	$(MAKE) check-activitynet-paper-artifacts

check-activitynet-paper-artifacts:
	python scripts/check_activitynet_paper_artifacts.py

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
