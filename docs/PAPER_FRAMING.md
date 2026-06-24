# Paper Framing Notes

This note defines the current claim boundary for writing the report.

## Core Framing

The target application is multimodal memory and retrieval for lifelog-style search
in the NTCIR-19 CASTLE setting. CASTLE is not used as the quantitative benchmark
because it does not provide timestamped Ground Truth Moments. ActivityNet Captions
is used as a controlled proxy benchmark for evaluating the temporal grounding
component with reproducible tIoU-based metrics.

ActivityNet should be described as component-level validation, not as proof that
the full lifelog system outperforms prior long-video systems.

## Contribution Wording

Use these claims:

- We implement a reproducible ActivityNet Captions evaluation pipeline for
  single-video moment retrieval.
- We evaluate temporal grounding with Recall@10 and mAP@10 at tIoU >= 0.3.
- We report multimodal fusion as an ablation. Current ASR/audio settings tie or
  slightly underperform the visual-only baseline.
- We report temporal window/stride as a granularity-cost trade-off. Coarser
  windows can improve tIoU-based metrics while returning less precise moments.
- We position ActivityNet as a stepping stone toward CASTLE-style lifelog
  retrieval, where quantitative temporal evaluation is blocked by missing
  timestamped ground truth.

Do not use these claims:

- Do not say ActivityNet is used because prior long-video work did not evaluate
  long videos.
- Do not claim superiority over WorldMM or any prior long-video method unless the
  same dataset, task, and evaluation protocol have been run.
- Do not claim ASR/audio improves ActivityNet performance with the current
  results.
- Do not claim semantic/generated-summary memory improves temporal grounding
  until summary evidence is indexed, searched, and evaluated in a controlled
  ablation.
- Do not compare `10s/5s` and `20s/10s` as if they have the same localization
  precision. Always report window and stride.

## Introduction Paragraph

This work targets multimodal memory and retrieval for lifelog-style search in the
NTCIR-19 CASTLE setting. Since CASTLE does not provide timestamped ground-truth
moments for quantitative temporal evaluation, we use ActivityNet Captions as a
controlled proxy benchmark for evaluating the temporal grounding component.
ActivityNet enables reproducible measurement with tIoU-based Recall@K and mAP,
while CASTLE remains the intended downstream application setting.

## Experiment Paragraph

We evaluate the retrieval component on an ActivityNet Captions dev200 subset with
200 videos and 693 sentence-level queries. Each query is evaluated against a
single selected video and one timestamped Ground Truth Moment. The system returns
Top-10 ranked Video Moments, and performance is measured with Recall@10 and
mAP@10 at tIoU >= 0.3. ActivityNet captions are used only as evaluation queries
and ground truth, not as indexed retrieval content.

## Results Paragraph

On ActivityNet dev200, the visual-only profile reaches Recall@10 = 0.455988 and
mAP@10 = 0.283630 with 10-second windows and 5-second stride. Light ASR/audio
fusion ties the visual-only baseline, while the heavy multimodal profile slightly
reduces mAP. A coarser 20-second window with 10-second stride reaches Recall@10 =
0.659452 and mAP@10 = 0.423096 while using 0.490x as many candidate windows, but
this result should be interpreted as a temporal granularity trade-off rather than
as strictly better localization.

## Limitation Paragraph

The ActivityNet experiments evaluate the temporal grounding component under a
controlled single-video setting. They do not constitute a direct comparison to
prior long-video or lifelog systems unless those systems are evaluated on the
same dataset and protocol. CASTLE remains the downstream application setting, but
without timestamped Ground Truth Moments it currently supports qualitative
inspection rather than tIoU-based quantitative claims.
