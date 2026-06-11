# Moment Search Checklist

This checklist captures the current product/research decisions so another agent can continue without relying on chat history.

## Decisions

- Primary goal: validate the WorldMM-inspired approach as semantic long-video retrieval.
- Canonical output: Top-10 ranked `Video Moment` results with `media_id`, `start_sec`, `end_sec`, `score`, `thumbnail_sec`, and source-specific `Evidence`.
- Benchmark path: no LLM reasoning. Return results only.
- Search scope: single selected video for the first validation flow.
- Baseline moment unit: 10-second windows with 5-second stride.
- Visual evidence baseline: sample one keyframe every 2 seconds; aggregate visual score per window with max frame score.
- Retrieval ranking: per-modality retrieval, normalize hits to `Video Moment`, then late fusion.
- ActivityNet Captions: the sole quantitative Evaluation Dataset and paper benchmark.
- CASTLE2024: not part of the benchmark scope because it has no Ground Truth Moments; keep only as a legacy/manual demo path if needed.
- ActivityNet captions must not be indexed. They are only Evaluation Queries and Ground Truth Moments.
- ASR generated from video audio may be indexed because it is video-derived evidence.
- Early metric: `Recall@10` with `tIoU >= 0.3`.
- ActivityNet evaluation unit: one caption sentence maps to one Ground Truth Moment.
- First manifest target: fixed-seed sample of 50 playable ActivityNet validation videos, capped at 500 Evaluation Queries if needed.

## Implementation Slices

- [x] Define `VideoMoment`, `Evidence`, and `MomentSearchResponse` schemas.
- [x] Add `POST /api/search/moments` as the primary search contract.
- [x] Keep `/api/search/episodic` for compatibility, but exclude it from benchmark flow.
- [x] Add ActivityNet Evaluation Manifest format and loader.
- [x] Add temporal metrics: tIoU, Recall@K, mAP@K for one-query-one-ground-truth evaluation.
- [x] Add fixed-window generation for media duration.
- [x] Add per-modality result normalization into `VideoMoment`.
- [x] Add visual-only weighted Video Moment ranking using Evaluation Profiles.
- [x] Add late-fusion ranking across visual and ASR evidence using Evaluation Profiles.
- [x] Add audio evidence to late-fusion ranking.
- [ ] Add summary evidence to late-fusion ranking if generated summaries are indexed separately.
- [x] Add ActivityNet Evaluation Profiles for visual-only and multimodal benchmark runs.
- [x] Update Search Interface to require selected video before search.
- [x] Render result timestamps and seek the video player to `start_sec` when clicked.
- [x] Add CASTLE Curated Query Set for manual inspection.
- [x] Add ActivityNet dev50 manifest generation script.
- [x] Add ActivityNet dev50 video downloader for playable YouTube subset.
- [x] Add evaluator runner that loads a manifest, calls Moment Search, and reports Recall@10/mAP@10.
- [x] Add ActivityNet ingestion command that preserves filename-based `media_id` for benchmark lookup.

## Completed Slices

- [x] ActivityNet manifest dataclasses/loader.
- [x] Temporal tIoU, Recall@K, and mAP@K metric functions.
- [x] Unit tests for manifest loading and temporal metrics.
- [x] Moment Search API schemas.
- [x] Empty-baseline `MomentSearchService` for the benchmark-facing contract.
- [x] `/api/v1/search/moments` and `/api/search/moments` endpoints.
- [x] Unit/integration tests for Moment Search API contract.
- [x] Fixed-window generation utility in `services/retrieval/moments.py`.
- [x] Evidence hit normalization into ranked `VideoMoment` results.
- [x] Evaluation Profile config and loader in `configs/evaluation_profiles.yaml` and `evaluation/profiles.py`.
- [x] `MomentSearchService` validates requested Evaluation Profile.
- [x] ActivityNet manifest builder in `evaluation/activitynet_manifest.py`.
- [x] CLI script `scripts/build_activitynet_manifest.py`.
- [x] Manifest-driven evaluator runner in `evaluation/moment_evaluator.py`.
- [x] Search Interface calls `/api/search/moments` with selected `media_id`, profile, query, and Top-K.
- [x] Moment Search request accepts optional `duration_sec`; UI and evaluator pass it when available.
- [x] Visual-only `MomentSearchService` uses CLIP text query embedding, searches `visual_keyframes`, and normalizes frame hits into fixed-window `VideoMoment` results.
- [x] ASR/text evidence search uses `TextEncoder`, searches `text_transcripts`, and merges transcript interval hits with visual frame hits.
- [x] Audio evidence search uses CLAP text query embedding, searches `audio_segments`, and merges audio interval hits with visual/ASR hits.
- [x] Moment Search isolates per-modality failures so one missing encoder/collection does not fail the whole request.
- [x] CASTLE smoke Curated Query Set in `data/curated_queries/castle_smoke.jsonl`.
- [x] Curated query loader in `evaluation/curated_queries.py`.
- [x] Manual inspection runner in `evaluation/castle_inspection.py`.
- [x] Ingestion keyframe sampling reads `configs/model_config.yaml` and defaults to 2 seconds, matching ADR-0006.
- [x] Single-query Moment Search CLI in `scripts/search_moments.py`.
- [x] Media index diagnostics in `evaluation/index_diagnostics.py` and `scripts/check_media_index.py`.
- [x] Local ActivityNet dev50 subset downloaded under `data/activitynet/videos/`.
- [x] ActivityNet dev50 manifest regenerated at `data/manifests/activitynet_dev50.jsonl`.
- [x] Manifest validation passed: 50 videos, 171 Evaluation Queries, 0 missing local video files.
- [x] `scripts/ingest_corpus.py` supports `--media-id-source filename`, so indexed `media_id` can match ActivityNet manifest IDs.
- [x] `make ingest-activitynet` ingests only video paths listed in the ActivityNet manifest, avoiding extra downloaded/intermediate files.
- [x] Video preprocessing no longer requires system `ffmpeg`/`ffprobe`; `services/ingestion/video_processor.py` uses PyAV.
- [x] VAD audio segmentation no longer depends on `librosa.load`; `services/ingestion/audio_processor.py` uses `soundfile` and `scipy.signal.resample_poly`.
- [x] Ingestion must run outside the sandbox/escalated when it needs local Docker Milvus at `localhost:19530`.
- [x] `make ingest-activitynet` sets `HF_HUB_DISABLE_XET=1` to avoid stalled Hugging Face xet downloads.
- [x] Text encoder `sentence-transformers/all-mpnet-base-v2` was preloaded successfully into `./model_cache`.
- [x] CLAP encoder supports current Transformers processor API (`audio=` and `pooler_output`).
- [x] Empty audio/text batches are skipped safely, so videos with no VAD speech can still index visual evidence.
- [x] `scripts/ingest_corpus.py` supports `--start-at-media-id` for resume.
- [x] `scripts/ingest_corpus.py` supports repeatable `--only-media-id` for targeted backfill.
- [x] ActivityNet dev50 ingestion completed in Milvus: 50/50 manifest media IDs indexed, 0 missing, 0 extra.
- [x] `v_T_5ANYuDWOA` validated as a no-speech case with visual-only evidence indexed successfully.
- [x] Moment Search response score normalization handles negative raw ANN scores without schema failures.
- [x] One ActivityNet Moment Search smoke query completed against indexed Milvus data and returned Top-10 results.
- [x] ActivityNet dev50 temporal evaluation completed with `activitynet_visual_heavy`: `Recall@10 = 0.47953216374269003`, `mAP@10 = 0.2946300937529008`, `tIoU >= 0.3`, 50 videos, 171 queries.
- [x] ActivityNet evaluator writes aggregate metrics to `data/evaluation/activitynet_dev50_summary.json`, per-query Top-K scores/tIoU/hit labels to `data/evaluation/activitynet_dev50_results.jsonl`, compact query table to `data/evaluation/activitynet_dev50_queries.csv`, and readable report to `data/evaluation/activitynet_dev50_report.md`.
- [x] ActivityNet dev200 manifest generated at `data/manifests/activitynet_dev200.jsonl`: 200 videos, 693 Evaluation Queries.
- [x] ActivityNet dev200 targeted ingestion checkpoint reached: 88/200 manifest media IDs indexed, covering 303 Evaluation Queries.
- [x] ActivityNet dev200 interim temporal evaluation completed on `data/manifests/activitynet_dev200_indexed88.jsonl` with `activitynet_visual_heavy`: `Recall@10 = 0.4884488448844885`, `mAP@10 = 0.3071114254282571`, `tIoU >= 0.3`, 88 videos, 303 queries.
- [x] ActivityNet dev200 interim evaluator outputs written to `data/evaluation/activitynet_dev200_indexed88_summary.json`, `data/evaluation/activitynet_dev200_indexed88_results.jsonl`, `data/evaluation/activitynet_dev200_indexed88_queries.csv`, and `data/evaluation/activitynet_dev200_indexed88_report.md`.
- [x] `scripts/ingest_corpus.py` supports selectable ingestion modalities via repeatable `--modality/--modalities` values: `visual`, `audio`, and `asr`.
- [x] `make ingest-activitynet` supports `MODALITIES="visual"` for faster visual-only scale-up runs.
- [x] `scripts/ingest_corpus.py` and `make ingest-activitynet` support per-run `KEYFRAME_INTERVAL_SEC` overrides for coarse visual scale-up tests without changing the default 2-second baseline config.
- [x] `scripts/ingest_corpus.py` and `make ingest-activitynet` support `--skip-indexed` / `SKIP_INDEXED=1` so resume runs skip media IDs already indexed for all selected modalities.
- [x] ActivityNet dev200 visual-only resume smoke indexed one more video before stopping the long batch: 89/200 manifest media IDs indexed, covering 308 Evaluation Queries.
- [x] ActivityNet dev200 current resume manifest regenerated at `data/manifests/activitynet_dev200_missing_resume.jsonl`: 111 videos, 385 Evaluation Queries still missing from Milvus.
- [x] ActivityNet dev200 current indexed manifest generated at `data/manifests/activitynet_dev200_indexed_current.jsonl`: 89 videos, 308 Evaluation Queries.
- [x] ActivityNet dev200 current checkpoint evaluation completed with `activitynet_visual_heavy`: `Recall@10 = 0.4935064935064935`, `mAP@10 = 0.3113249845392702`, `tIoU >= 0.3`, 89 videos, 308 queries.
- [x] ActivityNet dev200 current checkpoint evaluator outputs written to `data/evaluation/activitynet_dev200_indexed_current_summary.json`, `data/evaluation/activitynet_dev200_indexed_current_results.jsonl`, `data/evaluation/activitynet_dev200_indexed_current_queries.csv`, and `data/evaluation/activitynet_dev200_indexed_current_report.md`.
- [x] ActivityNet dev200 targeted coarse visual smoke completed with `KEYFRAME_INTERVAL_SEC=10`: `v_TX8FGTL1flw` indexed in 59s with 12 keyframes, down from the earlier 2-second smoke shape of 48 keyframes.
- [x] ActivityNet dev200 current manifests regenerated after coarse smoke: `data/manifests/activitynet_dev200_indexed_current.jsonl` has 90 videos/310 Evaluation Queries; `data/manifests/activitynet_dev200_missing_resume.jsonl` has 110 videos/383 Evaluation Queries remaining.
- [x] ActivityNet dev200 current checkpoint evaluation refreshed after coarse smoke: `Recall@10 = 0.49032258064516127`, `mAP@10 = 0.3093164362519201`, `tIoU >= 0.3`, 90 videos, 310 queries.
- [x] ActivityNet dev200 coarse visual ingestion is complete in the current Milvus volume: `data/manifests/activitynet_dev200_indexed_current.jsonl` has 200 videos and `data/manifests/activitynet_dev200_missing_resume.jsonl` has 0 remaining videos.
- [x] Added `activitynet_visual_only` Evaluation Profile for visual-only prototype runs without loading text/audio encoders.
- [x] ActivityNet full dev200 visual-only temporal evaluation completed: `Recall@10 = 0.455988455988456`, `mAP@10 = 0.283629950296617`, `tIoU >= 0.3`, 200 videos, 693 queries.
- [x] ActivityNet full dev200 visual-only evaluator outputs written to `data/evaluation/activitynet_dev200_visual_only_summary.json`, `data/evaluation/activitynet_dev200_visual_only_results.jsonl`, `data/evaluation/activitynet_dev200_visual_only_queries.csv`, and `data/evaluation/activitynet_dev200_visual_only_report.md`.
- [x] ActivityNet dev200 audio/ASR backfill completed in the current Milvus volume: 200/200 manifest media IDs have both `audio_segments` and `text_transcripts` evidence.
- [x] VAD no-speech fallback creates fixed full-track audio windows, so silent/non-speech videos can still index audio/ASR evidence.
- [x] ActivityNet full dev200 multimodal `activitynet_visual_heavy` evaluation completed: `Recall@10 = 0.455988455988456`, `mAP@10 = 0.28303499851118896`, `tIoU >= 0.3`, 200 videos, 693 queries.
- [x] ActivityNet full dev200 multimodal evaluator outputs written to `data/evaluation/activitynet_dev200_visual_heavy_summary.json`, `data/evaluation/activitynet_dev200_visual_heavy_results.jsonl`, `data/evaluation/activitynet_dev200_visual_heavy_queries.csv`, and `data/evaluation/activitynet_dev200_visual_heavy_report.md`.
- [x] CASTLE removed from benchmark planning; ActivityNet is the only dataset used for quantitative claims.
- [x] Added ActivityNet tuning profiles: `activitynet_visual_asr_light`, `activitynet_visual_asr_medium`, `activitynet_visual_audio_light`, and `activitynet_visual_audio_medium`.
- [x] Added `make eval-activitynet-profile-sweep` to run ActivityNet profile comparisons and write aggregate JSON/CSV.
- [x] ActivityNet dev200 profile sweep completed. `activitynet_visual_only`, `activitynet_visual_asr_light`, `activitynet_visual_asr_medium`, `activitynet_visual_audio_light`, and `activitynet_visual_audio_medium` tied at `Recall@10 = 0.455988455988456`, `mAP@10 = 0.283629950296617`; `activitynet_visual_heavy` was slightly lower at `mAP@10 = 0.28303499851118896`.

## Current Next Slice

- [ ] Add summary evidence search if/when generated summaries have their own indexed field.
- [ ] Use `scripts/check_media_index.py` before running manual or benchmark searches against real indexed media.
- [ ] Promote `activitynet_visual_only` as the current ActivityNet paper baseline unless a later architecture change makes multimodal evidence improve metrics.
- [ ] Inspect per-query regressions between the best multimodal profile and `activitynet_visual_only`.
- [ ] Decide whether to keep coarse 10-second visual sampling as the prototype baseline before larger runs.

## Command Cookbook

Make target shortcuts:

```bash
make build-activitynet-manifest
make eval-moments MANIFEST=data/manifests/activitynet_dev50.jsonl
make eval-moments MANIFEST=data/manifests/activitynet_dev200.jsonl PROFILE=activitynet_visual_only SUMMARY=data/evaluation/activitynet_dev200_visual_only_summary.json RESULTS=data/evaluation/activitynet_dev200_visual_only_results.jsonl QUERY_CSV=data/evaluation/activitynet_dev200_visual_only_queries.csv REPORT=data/evaluation/activitynet_dev200_visual_only_report.md
make eval-moments MANIFEST=data/manifests/activitynet_dev200.jsonl PROFILE=activitynet_visual_heavy SUMMARY=data/evaluation/activitynet_dev200_visual_heavy_summary.json RESULTS=data/evaluation/activitynet_dev200_visual_heavy_results.jsonl QUERY_CSV=data/evaluation/activitynet_dev200_visual_heavy_queries.csv REPORT=data/evaluation/activitynet_dev200_visual_heavy_report.md
make eval-activitynet-profile-sweep MANIFEST=data/manifests/activitynet_dev200.jsonl
make ingest-activitynet
make ingest-activitynet MANIFEST=data/manifests/activitynet_dev200_missing_resume.jsonl MODALITIES="visual"
make ingest-activitynet MANIFEST=data/manifests/activitynet_dev200_missing_resume.jsonl MODALITIES="visual" KEYFRAME_INTERVAL_SEC=10
make ingest-activitynet MANIFEST=data/manifests/activitynet_dev200_missing_resume.jsonl MODALITIES="visual" KEYFRAME_INTERVAL_SEC=10 SKIP_INDEXED=1
make ingest-activitynet MANIFEST=data/manifests/activitynet_dev200_missing_audio_asr.jsonl MODALITIES="audio asr" SKIP_INDEXED=1
make list-indexed-media
make check-media-index MEDIA_ID=v_123
make search-moments MEDIA_ID=v_123 DURATION_SEC=120 QUERY="woman doing sit ups"
```

Build an ActivityNet dev manifest after videos are available locally:

```bash
python scripts/build_activitynet_manifest.py \
  --video-root data/activitynet/videos \
  --output-path data/manifests/activitynet_dev50.jsonl
```

Download a playable ActivityNet dev subset and write a manifest:

```bash
python scripts/download_activitynet_dev.py \
  --split val1 \
  --target-videos 50 \
  --max-attempts 500 \
  --output-dir data/activitynet/videos \
  --manifest-path data/manifests/activitynet_dev50.jsonl
```

Run ActivityNet temporal evaluation against the Moment Search service:

```bash
python -m evaluation.moment_evaluator \
  data/manifests/activitynet_dev50.jsonl \
  --profile-name activitynet_visual_heavy
```

Run one Moment Search query from the shell:

```bash
python scripts/search_moments.py \
  --media-id v_123 \
  --duration-sec 120 \
  --query "woman doing sit ups" \
  --profile activitynet_visual_heavy
```

Check whether one media ID has indexed Moment Search evidence:

```bash
python scripts/list_indexed_media.py \
  --sample-limit 1000

python scripts/check_media_index.py \
  --media-id v_123 \
  --sample-limit 100
```

## Domain Docs

- Glossary: `CONTEXT.md`
- ADRs: `docs/adr/`
- Real-data validation runbook: `docs/MOMENT_SEARCH_RUNBOOK.md`
