# Paper Artifacts

Run this command after evaluation summaries are available:

```bash
make build-activitynet-paper-artifacts
```

It regenerates the current paper-facing ActivityNet artifacts from existing
summary/result files and runs a consistency check at the end.

To check existing artifacts without regenerating them:

```bash
make check-activitynet-paper-artifacts
```

## Result Tables

- `data/evaluation/activitynet_results_table.csv`: spreadsheet-friendly
  profile-ablation table.
- `data/evaluation/activitynet_results_table.md`: Markdown profile-ablation
  table.
- `data/evaluation/activitynet_results_table.tex`: LaTeX profile-ablation table
  for the report.
- `data/evaluation/activitynet_findings.md`: paper-ready findings and claim
  boundary for the multimodal profile sweep.

## Temporal Trade-off Tables

- `data/evaluation/activitynet_temporal_tradeoff.csv`: spreadsheet-friendly
  quality/cost/runtime table.
- `data/evaluation/activitynet_temporal_tradeoff.md`: Markdown temporal
  granularity table with interpretation.
- `data/evaluation/activitynet_temporal_tradeoff.tex`: LaTeX temporal
  granularity table for the report.

## Cost Estimates

- `data/evaluation/activitynet_ablation_costs.csv`: candidate-window and
  keyframe cost estimates.
- `data/evaluation/activitynet_ablation_costs.md`: Markdown cost table.
- `data/evaluation/activitynet_ablation_costs.json`: machine-readable cost
  table used by the temporal trade-off summarizer.

## Consistency Check

`scripts/check_activitynet_paper_artifacts.py` verifies that:

- the profile table matches the current visual-only summary JSON,
- the temporal trade-off table matches the current 10s/5s and 20s/10s summary
  JSON files,
- candidate-window counts and relative costs match
  `activitynet_ablation_costs.json`,
- findings and report draft text preserve the current claim boundary.

## Writing Guides

- `docs/PAPER_FRAMING.md`: Do/Don't claim boundary and reusable report
  paragraphs.
- `docs/REPORT_RESULTS_DRAFT.md`: draft experiment/results/discussion text.
- `docs/ACTIVITYNET_EXPERIMENT_PROTOCOL.md`: source of truth for the frozen
  protocol, metric definitions, current numbers, and claim boundary.

## Current Main Numbers

- ActivityNet dev200: 200 videos, 693 sentence-level queries.
- Baseline `activitynet_visual_only`, `10s/5s`: Recall@10 = 0.455988, mAP@10 =
  0.283630, elapsed = 153.32s, throughput = 4.52 queries/sec.
- Coarse `activitynet_visual_only`, `20s/10s`: Recall@10 = 0.659452, mAP@10 =
  0.423096, elapsed = 133.53s, throughput = 5.19 queries/sec.
- `20s/10s` uses 2184 candidate windows, or 0.490x the `10s/5s` candidate-window
  count.

Interpret the coarse setting as a temporal granularity trade-off, not as a strict
localization improvement.
