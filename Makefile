.PHONY: dev worker test lint format ingest eval build \
	audit-castle \
	build-castle-slice \
	build-castle-fixed-manifest \
	enrich-castle-transcripts \
	download-castle-slice \
	sample-castle-frames \
	build-visual-semantic-events \
	compare-visual-segmenters \
	evaluate-visual-boundaries \
	compare-visual-retrieval \
	sweep-transcript-boundary-weights \
	build-castle-timeline-inventory \
	enrich-castle-heart-rate \
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

audit-castle:
	python scripts/audit_castle_dataset.py \
		--output-dir $(or $(OUTPUT_DIR),processed/audit)

build-castle-slice:
	python scripts/build_castle_slice.py \
		--day $(or $(DAY),day1) \
		--participant-id $(or $(PARTICIPANT),Allie) \
		--output-dir $(or $(OUTPUT_DIR),processed/slices) \
		$(if $(MAX_RECORDINGS),--max-recordings $(MAX_RECORDINGS))

build-castle-fixed-manifest:
	python scripts/build_castle_fixed_manifest.py \
		$(or $(RECORDINGS),processed/timeline/recordings.jsonl) \
		--output-path $(or $(OUTPUT),processed/chunks/fixed_30s.jsonl) \
		--window $(or $(WINDOW),30s) \
		--processing-version $(or $(PROCESSING_VERSION),dev)

enrich-castle-transcripts:
	python scripts/enrich_castle_transcripts.py \
		$(or $(INPUT),processed/slices/day1_Allie/fixed_30s.jsonl) \
		--output-manifest $(or $(OUTPUT),processed/slices/day1_Allie/dev_08_10_fixed_30s_transcript.jsonl) \
		--day $(or $(DAY),day1) \
		--participant-id $(or $(PARTICIPANT),Allie) \
		$(foreach STEM,$(STEMS),--recording-stem $(STEM)) \
		--revision $(or $(REVISION),c8e7b5cd9e9c83d0ff42560fc1169bed7867abd4) \
		--cleaned-spans-path $(or $(CLEANED_SPANS),processed/slices/day1_Allie/dev_08_10_cleaned_transcripts.jsonl) \
		--rejected-spans-path $(or $(REJECTED_SPANS),processed/slices/day1_Allie/dev_08_10_rejected_transcripts.csv)

download-castle-slice:
	python scripts/download_castle_slice.py \
		--day $(or $(DAY),day1) \
		--participant-id $(or $(PARTICIPANT),Allie) \
		$(foreach STEM,$(STEMS),--recording-stem $(STEM)) \
		--revision $(or $(REVISION),c8e7b5cd9e9c83d0ff42560fc1169bed7867abd4) \
		--output-dir $(or $(OUTPUT_DIR),data/castle)

sample-castle-frames:
	python scripts/sample_castle_frames.py \
		$(or $(RECORDINGS),processed/slices/day1_Allie/recordings.jsonl) \
		$(foreach VIDEO_ID,$(VIDEO_IDS),--video-id $(VIDEO_ID)) \
		--output-dir $(or $(OUTPUT_DIR),processed/frames) \
		--interval-sec $(or $(INTERVAL_SEC),600)

build-visual-semantic-events:
	python scripts/build_visual_semantic_events.py \
		$(FRAME_DIR) \
		--video-id $(VIDEO_ID) \
		--participant-id $(PARTICIPANT) \
		--video-uri "$(VIDEO_URI)" \
		--output-manifest $(OUTPUT_MANIFEST) \
		--output-embeddings $(OUTPUT_EMBEDDINGS) \
		--output-scores $(OUTPUT_SCORES) \
		--processing-version $(PROCESSING_VERSION) \
		--model-name $(or $(MODEL),ViT-B-32-quickgelu) \
		--pretrained $(or $(PRETRAINED),openai) \
		$(if $(TRANSCRIPT_SPANS),--transcript-spans $(TRANSCRIPT_SPANS)) \
		$(if $(TRANSCRIPT_WEIGHT),--transcript-weight $(TRANSCRIPT_WEIGHT))

compare-visual-segmenters:
	python scripts/compare_visual_segmenters.py \
		$(FRAME_DIR) \
		--output-csv $(OUTPUT_CSV) \
		--output-json $(OUTPUT_JSON) \
		--model-name $(or $(MODEL),ViT-B-32-quickgelu) \
		--pretrained $(or $(PRETRAINED),openai)

evaluate-visual-boundaries:
	python scripts/evaluate_visual_boundaries.py \
		$(or $(COMPARISON),processed/semantic/dev_08_400_700_detector_comparison.json) \
		$(or $(REFERENCE),evaluation/fixtures/castle_dev08_400_700_boundaries.jsonl) \
		--output-csv $(or $(OUTPUT),processed/semantic/dev_08_400_700_boundary_evaluation.csv)

compare-visual-retrieval:
	python scripts/compare_visual_retrieval.py \
		$(or $(FRAME_DIR),processed/frames/dev_08_activity/day1_Allie_08) \
		$(or $(SEMANTIC_MANIFEST),processed/semantic/dev_08_400_700_v2_events.jsonl) \
		$(or $(QUERIES),evaluation/fixtures/castle_dev08_visual_queries.jsonl) \
		--output-results $(or $(RESULTS),processed/semantic/dev_08_400_700_visual_retrieval_results.csv) \
		--output-summary $(or $(SUMMARY),processed/semantic/dev_08_400_700_visual_retrieval_summary.csv)

sweep-transcript-boundary-weights:
	python scripts/sweep_transcript_boundary_weights.py \
		$(or $(CASES),evaluation/fixtures/castle_transcript_weight_cases.jsonl) \
		--transcript-spans $(or $(TRANSCRIPT_SPANS),processed/slices/day1_Allie/dev_08_10_cleaned_transcripts.jsonl) \
		--output-csv $(or $(OUTPUT),processed/semantic/transcript_weight_sweep.csv) \
		--output-summary $(or $(SUMMARY),processed/semantic/transcript_weight_sweep_summary.csv) \
		$(foreach WEIGHT,$(WEIGHTS),--weight $(WEIGHT))

build-castle-timeline-inventory:
	python scripts/build_castle_timeline_inventory.py \
		--day $(or $(DAY),day1) \
		--participant-id $(or $(PARTICIPANT),Allie) \
		$(foreach STEM,$(STEMS),--recording-stem $(STEM)) \
		--metadata-sensor $(or $(SENSOR),ACCL) \
		--output-csv $(or $(OUTPUT),processed/timeline/day1_Allie/source_timeline_inventory.csv) \
		--output-json $(or $(JSON),processed/timeline/day1_Allie/source_timeline_inventory.json)

enrich-castle-heart-rate:
	python scripts/enrich_castle_heart_rate.py \
		$(or $(INPUT),processed/semantic/dev_08_400_700_visual_text_events.jsonl) \
		--output-manifest $(or $(OUTPUT),processed/semantic/dev_08_400_700_visual_text_hr_events.jsonl) \
		--timeline-inventory $(or $(TIMELINE_INVENTORY),processed/timeline/day1_Allie/source_timeline_inventory.csv) \
		--output-summary $(or $(SUMMARY),processed/semantic/dev_08_400_700_visual_text_hr_summary.csv) \
		--day $(or $(DAY),day1) \
		--participant-id $(or $(PARTICIPANT),Allie) \
		--min-confidence $(or $(MIN_CONFIDENCE),1.0)

# Legacy ActivityNet pipeline retained for provenance during CASTLE migration.
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
