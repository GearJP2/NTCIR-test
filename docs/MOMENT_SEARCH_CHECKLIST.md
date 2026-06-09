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
- ActivityNet Captions: quantitative Evaluation Dataset.
- CASTLE2024: Development Dataset for qualitative/manual inspection only.
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
- [x] Add `activitynet_visual_heavy` and `castle_lifelog_balanced` Evaluation Profiles.
- [x] Update Search Interface to require selected video before search.
- [x] Render result timestamps and seek the video player to `start_sec` when clicked.
- [ ] Add CASTLE Curated Query Set for manual inspection.
- [x] Add ActivityNet dev50 manifest generation script.
- [x] Add evaluator runner that loads a manifest, calls Moment Search, and reports Recall@10/mAP@10.

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

## Current Next Slice

- [ ] Add summary evidence search if/when generated summaries have their own indexed field.
- [ ] Add CASTLE Curated Query Set for manual inspection.

## Domain Docs

- Glossary: `CONTEXT.md`
- ADRs: `docs/adr/`
