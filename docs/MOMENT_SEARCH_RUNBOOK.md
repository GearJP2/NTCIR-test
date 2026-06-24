# Moment Search Runbook

Use this runbook after media has been ingested into Milvus.

## 1. Check One Indexed Media ID

For ActivityNet benchmark runs, ingest with filename-based media IDs so indexed evidence
matches manifest IDs:

```bash
make ingest-activitynet
```

For faster visual-only scale-up runs, skip audio extraction, VAD, ASR, and audio embeddings:

```bash
make ingest-activitynet \
  MANIFEST=data/manifests/activitynet_dev200_missing_resume.jsonl \
  MODALITIES="visual"
```

For a coarse scale-up run, increase the visual sampling interval:

```bash
make ingest-activitynet \
  MANIFEST=data/manifests/activitynet_dev200_missing_resume.jsonl \
  MODALITIES="visual" \
  KEYFRAME_INTERVAL_SEC=10 \
  SKIP_INDEXED=1
```

For a targeted smoke/backfill run, pass one or more media IDs:

```bash
make ingest-activitynet \
  MANIFEST=data/manifests/activitynet_dev200_missing_resume.jsonl \
  MODALITIES="visual" \
  KEYFRAME_INTERVAL_SEC=10 \
  ONLY_MEDIA_ID="v_123 v_456"
```

`SKIP_INDEXED=1` checks the selected modality collections before queuing work. For example, a visual-only run skips media IDs that already have at least one `visual_keyframes` row, while a full `visual audio asr` run only skips media IDs that have rows in all three modality collections.

To build a manifest for missing modality evidence and backfill audio/ASR only:

```bash
python scripts/build_missing_modality_manifest.py \
  --manifest-path data/manifests/activitynet_dev200.jsonl \
  --output-path data/manifests/activitynet_dev200_missing_audio_asr.jsonl \
  --modality audio \
  --modality asr

make ingest-activitynet \
  MANIFEST=data/manifests/activitynet_dev200_missing_audio_asr.jsonl \
  MODALITIES="audio asr" \
  SKIP_INDEXED=1
```

If VAD finds no speech but the video has an audio track, ingestion falls back to fixed full-track windows. This keeps non-speech audio evidence searchable and prevents audio/ASR completeness checks from staying missing only because VAD returned zero speech segments.

The dev200 visual-only checkpoint measured about 3m52s for the first remaining video at the default 2-second sampling interval on the current machine, so run the remaining resume manifest in controlled batches or use a coarser interval for prototype-scale checks.

List candidate media IDs sampled from Moment Search collections:

```bash
make list-indexed-media
```

Then inspect one candidate:

```bash
make check-media-index MEDIA_ID=v_123
```

At least one of `visual_keyframes`, `text_transcripts`, or `audio_segments` must report `ready: true`. If all are false, the media has not been indexed for Moment Search.

## 2. Run One Query

```bash
make search-moments \
  MEDIA_ID=v_123 \
  DURATION_SEC=120 \
  QUERY="woman doing sit ups" \
  PROFILE=activitynet_visual_heavy
```

The response should contain ranked `VideoMoment` results with source-specific `Evidence`. Empty results usually mean either the media has no indexed evidence or all modality searches failed.

## 3. ActivityNet Evaluation

Download a playable 50-video ActivityNet validation subset:

```bash
python scripts/download_activitynet_dev.py \
  --split val1 \
  --target-videos 50 \
  --max-attempts 500 \
  --output-dir data/activitynet/videos \
  --manifest-path data/manifests/activitynet_dev50.jsonl
```

Or build the manifest after ActivityNet videos already exist locally:

```bash
make build-activitynet-manifest \
  VIDEO_ROOT=data/activitynet/videos \
  MANIFEST=data/manifests/activitynet_dev50.jsonl
```

Run temporal evaluation:

```bash
make eval-moments MANIFEST=data/manifests/activitynet_dev50.jsonl
```

This reports `Recall@10` and `mAP@10` using the `activitynet_visual_heavy` profile and `tIoU >= 0.3`.
It also writes:

- `data/evaluation/activitynet_dev50_summary.json`: aggregate metrics.
- `data/evaluation/activitynet_dev50_results.jsonl`: one row per Evaluation Query with Ground Truth Moment, Top-K result scores, tIoU, and hit/miss labels.
- `data/evaluation/activitynet_dev50_queries.csv`: compact query-level table for spreadsheet inspection.
- `data/evaluation/activitynet_dev50_report.md`: human-readable report with summary metrics, hit-rank distribution, and closest misses.

For the coarse dev200 visual-only prototype baseline:

```bash
make eval-moments \
  MANIFEST=data/manifests/activitynet_dev200.jsonl \
  PROFILE=activitynet_visual_only \
  SUMMARY=data/evaluation/activitynet_dev200_visual_only_summary.json \
  RESULTS=data/evaluation/activitynet_dev200_visual_only_results.jsonl \
  QUERY_CSV=data/evaluation/activitynet_dev200_visual_only_queries.csv \
  REPORT=data/evaluation/activitynet_dev200_visual_only_report.md
```

The current full dev200 visual-only checkpoint is `Recall@10 = 0.455988455988456`, `mAP@10 = 0.283629950296617`, elapsed `153.31829674399887s`, throughput `4.5200084707249735 queries/sec`, `tIoU >= 0.3`, 200 videos, and 693 queries. When running from Codex sandbox, local Docker/Milvus access may require escalated execution; normal shell sessions can run the command directly.

For the dev200 temporal granularity ablation, run a coarser `20s/10s` window:

```bash
make eval-moments \
  MANIFEST=data/manifests/activitynet_dev200.jsonl \
  PROFILE=activitynet_visual_only \
  WINDOW_SEC=20 \
  STRIDE_SEC=10 \
  SUMMARY=data/evaluation/activitynet_dev200_visual_only_w20_s10_summary.json \
  RESULTS=data/evaluation/activitynet_dev200_visual_only_w20_s10_results.jsonl \
  QUERY_CSV=data/evaluation/activitynet_dev200_visual_only_w20_s10_queries.csv \
  REPORT=data/evaluation/activitynet_dev200_visual_only_w20_s10_report.md
```

The current `20s/10s` checkpoint is `Recall@10 = 0.6594516594516594`, `mAP@10 = 0.42309603976270643`, elapsed `133.52791842399893s`, and throughput `5.18992588351057 queries/sec`. The score is higher than `10s/5s`, but the returned moments are coarser; report window and stride with every temporal result.

Generate the paper-ready quality/cost/runtime table:

```bash
make summarize-activitynet-temporal-tradeoff \
  LATEX=data/evaluation/activitynet_temporal_tradeoff.tex
```

This writes:

- `data/evaluation/activitynet_temporal_tradeoff.csv`
- `data/evaluation/activitynet_temporal_tradeoff.md`
- `data/evaluation/activitynet_temporal_tradeoff.tex`

Estimate ablation compute size from the manifest:

```bash
make estimate-activitynet-ablation-costs \
  JSON=data/evaluation/activitynet_ablation_costs.json
```

Current dev200 estimate:

| Type | Setting | Units | Avg/video | Relative cost |
| --- | --- | ---: | ---: | ---: |
| moment windows | 10s/5s | 4454 | 22.27 | 1.000x |
| moment windows | 20s/10s | 2184 | 10.92 | 0.490x |
| visual keyframes | 2s | 11471 | 57.35 | 1.000x |
| visual keyframes | 5s | 4653 | 23.27 | 0.406x |
| visual keyframes | 10s | 2383 | 11.91 | 0.208x |

For the full dev200 multimodal profile after audio/ASR backfill:

```bash
make eval-moments \
  MANIFEST=data/manifests/activitynet_dev200.jsonl \
  PROFILE=activitynet_visual_heavy \
  SUMMARY=data/evaluation/activitynet_dev200_visual_heavy_summary.json \
  RESULTS=data/evaluation/activitynet_dev200_visual_heavy_results.jsonl \
  QUERY_CSV=data/evaluation/activitynet_dev200_visual_heavy_queries.csv \
  REPORT=data/evaluation/activitynet_dev200_visual_heavy_report.md
```

The current full dev200 `activitynet_visual_heavy` checkpoint is `Recall@10 = 0.455988455988456`, `mAP@10 = 0.28303499851118896`, `tIoU >= 0.3`, 200 videos, and 693 queries. This does not improve on `activitynet_visual_only`, so the next benchmark slice should tune modality weights or inspect per-query regressions before treating the multimodal profile as the baseline.

Run the ActivityNet profile sweep:

```bash
make eval-activitynet-profile-sweep \
  MANIFEST=data/manifests/activitynet_dev200.jsonl
```

This runs the default ActivityNet comparison set:

- `activitynet_visual_only`
- `activitynet_visual_asr_light`
- `activitynet_visual_asr_medium`
- `activitynet_visual_audio_light`
- `activitynet_visual_audio_medium`
- `activitynet_visual_heavy`

The sweep writes:

- `data/evaluation/activitynet_profile_sweep_summary.json`
- `data/evaluation/activitynet_profile_sweep_summary.csv`

Current dev200 sweep result:

| Profile | Recall@10 | mAP@10 |
| --- | ---: | ---: |
| `activitynet_visual_only` | 0.455988455988456 | 0.283629950296617 |
| `activitynet_visual_asr_light` | 0.455988455988456 | 0.283629950296617 |
| `activitynet_visual_asr_medium` | 0.455988455988456 | 0.283629950296617 |
| `activitynet_visual_audio_light` | 0.455988455988456 | 0.283629950296617 |
| `activitynet_visual_audio_medium` | 0.455988455988456 | 0.283629950296617 |
| `activitynet_visual_heavy` | 0.455988455988456 | 0.28303499851118896 |

Use `activitynet_visual_only` as the current ActivityNet component baseline. The light and medium ASR/audio variants tie visual-only, while the heavy multimodal profile slightly reduces mAP.

Build a paper-ready result table from existing summary JSON files:

```bash
make summarize-activitynet-results
```

This writes:

- `data/evaluation/activitynet_results_table.csv`
- `data/evaluation/activitynet_results_table.md`
- `data/evaluation/activitynet_results_table.tex`
- `data/evaluation/activitynet_findings.md`

Compare per-query behavior between the current baseline and a candidate profile:

```bash
make compare-activitynet-results \
  BASELINE=data/evaluation/activitynet_dev200_visual_only_results.jsonl \
  CANDIDATE=data/evaluation/activitynet_dev200_visual_heavy_results.jsonl \
  JSON=data/evaluation/activitynet_visual_only_vs_heavy_regressions.json
```

The current `visual_only` vs `visual_heavy` report shows 316 hits for both profiles, 0 lost hits, 0 gained hits, 4 worse hit ranks, and 2 better hit ranks. This explains the slightly lower multimodal mAP: heavy multimodal evidence mostly preserves hit/miss status but perturbs a few ranks downward.

Use `WRITE_DETAILS=1` if per-profile JSONL/CSV/Markdown reports are needed for regression analysis:

```bash
make eval-activitynet-profile-sweep \
  MANIFEST=data/manifests/activitynet_dev200.jsonl \
  WRITE_DETAILS=1
```

## 4. Paper Scope

ActivityNet Captions is the controlled quantitative benchmark for temporal grounding because it provides timestamped Ground Truth Moments. CASTLE remains the downstream lifelog application setting, but is not part of the tIoU benchmark path because it has no Ground Truth Moments; keep CASTLE tooling as manual inspection/demo support only.

The frozen paper protocol lives in `docs/ACTIVITYNET_EXPERIMENT_PROTOCOL.md`. Use that document as the source of truth for dataset scope, metrics, baseline profile, current ablation results, regression interpretation, and claim boundaries. Use `docs/PAPER_FRAMING.md` and `docs/REPORT_RESULTS_DRAFT.md` for report wording.

## 5. Known Limits

- Generated-summary evidence is not searched separately yet.
- Full `pytest tests/unit` has known legacy failures outside the Moment Search slice.
- Moment Search depends on indexed Milvus collections and locally available model weights.
