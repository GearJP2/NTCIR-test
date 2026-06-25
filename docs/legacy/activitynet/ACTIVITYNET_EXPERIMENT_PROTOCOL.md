# ActivityNet Experiment Protocol

This document freezes the current quantitative evaluation protocol for paper writing.

## Why ActivityNet

The target application setting is NTCIR-19 CASTLE-style lifelog retrieval. CASTLE
does not provide timestamped Ground Truth Moments, so it cannot support
tIoU-based quantitative temporal evaluation. ActivityNet Captions is therefore
used as a controlled proxy benchmark for the temporal grounding component.

This protocol does not claim superiority over prior long-video systems unless
they are evaluated on the same dataset and protocol. It validates the retrieval
component under a measurable single-video moment localization setting.

## Scope

- Dataset: ActivityNet Captions dev200 playable subset.
- Evaluation unit: one caption sentence mapped to one Ground Truth Moment.
- Current subset size: 200 videos and 693 Evaluation Queries.
- Search scope: single selected video.
- Benchmark path: Moment Search only; no LLM reasoning.
- CASTLE2024 is outside the quantitative benchmark scope because it has no Ground Truth Moments.

## Retrieval Output

Each query returns Top-10 ranked `VideoMoment` results with:

- `media_id`
- `start_sec`
- `end_sec`
- `score`
- source-specific evidence

The current baseline uses fixed temporal windows and ranks moments by late-fused evidence scores.

## Metrics

- `Recall@10`
- `mAP@10`
- tIoU threshold: `0.3`

A prediction is counted as a hit when a retrieved moment overlaps the Ground Truth Moment with `tIoU >= 0.3`.

## Current Baseline

Use `activitynet_visual_only` as the current paper baseline:

```bash
make eval-moments \
  MANIFEST=data/manifests/activitynet_dev200.jsonl \
  PROFILE=activitynet_visual_only \
  SUMMARY=data/evaluation/activitynet_dev200_visual_only_summary.json \
  RESULTS=data/evaluation/activitynet_dev200_visual_only_results.jsonl \
  QUERY_CSV=data/evaluation/activitynet_dev200_visual_only_queries.csv \
  REPORT=data/evaluation/activitynet_dev200_visual_only_report.md
```

Current result:

| Profile | Window | Stride | Videos | Queries | Recall@10 | mAP@10 | Elapsed | Queries/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `activitynet_visual_only` | 10s | 5s | 200 | 693 | 0.455988 | 0.283630 | 153.32s | 4.52 |

Temporal granularity is part of the protocol. Do not compare runs with different
window/stride settings without reporting those settings because wider windows can
raise tIoU-based metrics while returning coarser moments.

## Temporal Granularity Ablation

Run a coarse-window comparison:

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

Current result:

| Profile | Window | Stride | Candidate windows | Relative cost | Recall@10 | mAP@10 | Elapsed | Queries/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `activitynet_visual_only` | 10s | 5s | 4454 | 1.000x | 0.455988 | 0.283630 | 153.32s | 4.52 |
| `activitynet_visual_only` | 20s | 10s | 2184 | 0.490x | 0.659452 | 0.423096 | 133.53s | 5.19 |

Interpretation: `20s/10s` uses roughly half as many candidate windows as `10s/5s`
and scores higher at `tIoU >= 0.3`, but its returned moments are coarser. Treat
this as a temporal granularity ablation, not a replacement baseline unless the
paper explicitly prefers coarse localization.

Estimate candidate-window and keyframe costs:

```bash
make estimate-activitynet-ablation-costs \
  JSON=data/evaluation/activitynet_ablation_costs.json
```

Current estimate:

| Type | Setting | Units | Avg/video | Relative cost |
| --- | --- | ---: | ---: | ---: |
| moment windows | 10s/5s | 4454 | 22.27 | 1.000x |
| moment windows | 20s/10s | 2184 | 10.92 | 0.490x |
| visual keyframes | 2s | 11471 | 57.35 | 1.000x |
| visual keyframes | 5s | 4653 | 23.27 | 0.406x |
| visual keyframes | 10s | 2383 | 11.91 | 0.208x |

## Multimodal Ablation

Run the profile sweep:

```bash
make eval-activitynet-profile-sweep \
  MANIFEST=data/manifests/activitynet_dev200.jsonl
```

Current sweep:

| Profile | Recall@10 | mAP@10 |
| --- | ---: | ---: |
| `activitynet_visual_only` | 0.455988 | 0.283630 |
| `activitynet_visual_asr_light` | 0.455988 | 0.283630 |
| `activitynet_visual_asr_medium` | 0.455988 | 0.283630 |
| `activitynet_visual_audio_light` | 0.455988 | 0.283630 |
| `activitynet_visual_audio_medium` | 0.455988 | 0.283630 |
| `activitynet_visual_heavy` | 0.455988 | 0.283035 |

Interpretation: ASR/audio fusion did not improve this ActivityNet subset. Light and medium multimodal profiles tied visual-only, while the heavy profile slightly reduced mAP.

## Regression Analysis

Compare visual-only against heavy multimodal:

```bash
make compare-activitynet-results \
  BASELINE=data/evaluation/activitynet_dev200_visual_only_results.jsonl \
  CANDIDATE=data/evaluation/activitynet_dev200_visual_heavy_results.jsonl \
  JSON=data/evaluation/activitynet_visual_only_vs_heavy_regressions.json
```

Current result:

- Baseline hits: 316
- Candidate hits: 316
- Lost hits: 0
- Gained hits: 0
- Worse hit ranks: 4
- Better hit ranks: 2

Interpretation: heavy multimodal evidence mostly preserves hit/miss status but perturbs a few ranks downward, explaining the slightly lower mAP.

## Paper Exports

Generate paper-ready tables and findings:

```bash
make summarize-activitynet-results
```

This writes:

- `data/evaluation/activitynet_results_table.csv`
- `data/evaluation/activitynet_results_table.md`
- `data/evaluation/activitynet_results_table.tex`
- `data/evaluation/activitynet_findings.md`

Use `docs/legacy/activitynet/PAPER_FRAMING.md` for claim wording, `docs/legacy/activitynet/REPORT_RESULTS_DRAFT.md`
for a report-ready experiment/results/discussion draft, and
`docs/legacy/activitynet/PAPER_ARTIFACTS.md` for the artifact manifest.

## Claim Boundary

Use this framing until a later architecture or retrieval change improves multimodal metrics:

- Main quantitative claim: component-level temporal grounding on ActivityNet Captions.
- Multimodal evidence: exploratory ablation.
- CASTLE: downstream lifelog application setting, not quantitative tIoU benchmark.
- Do not claim ASR/audio improves ActivityNet performance with the current results.
- Do not claim superiority over WorldMM or any prior long-video system without matching its dataset and protocol.
- Keep generated-summary/semantic-memory improvements as future work until they are indexed, searched, and evaluated.
