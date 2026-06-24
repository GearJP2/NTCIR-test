# Report Results Draft

This draft is written for the report's experiment and discussion sections.

## Experimental Framing

This work targets multimodal memory and retrieval for lifelog-style search in the
NTCIR-19 CASTLE setting. Since CASTLE does not provide timestamped Ground Truth
Moments for quantitative temporal evaluation, ActivityNet Captions is used as a
controlled proxy benchmark for the temporal grounding component. ActivityNet
enables reproducible measurement with tIoU-based Recall@K and mAP, while CASTLE
remains the intended downstream application setting.

The ActivityNet experiments should therefore be interpreted as component-level
validation of temporal moment retrieval, not as a direct comparison against prior
long-video or lifelog systems.

## Protocol

We evaluate on an ActivityNet Captions dev200 subset with 200 videos and 693
sentence-level queries. Each query is evaluated against a single selected video
and one timestamped Ground Truth Moment. The system returns Top-10 ranked Video
Moments. Performance is measured with Recall@10 and mAP@10 at tIoU >= 0.3.
ActivityNet captions are used only as evaluation queries and ground truth, not as
indexed retrieval content.

## Main ActivityNet Result

| Profile | Videos | Queries | Recall@10 | mAP@10 |
| --- | ---: | ---: | ---: | ---: |
| activitynet_visual_only | 200 | 693 | 0.455988 | 0.283630 |
| activitynet_visual_asr_light | 200 | 693 | 0.455988 | 0.283630 |
| activitynet_visual_asr_medium | 200 | 693 | 0.455988 | 0.283630 |
| activitynet_visual_audio_light | 200 | 693 | 0.455988 | 0.283630 |
| activitynet_visual_audio_medium | 200 | 693 | 0.455988 | 0.283630 |
| activitynet_visual_heavy | 200 | 693 | 0.455988 | 0.283035 |

The visual-only profile achieves Recall@10 = 0.455988 and mAP@10 = 0.283630.
Light ASR/audio fusion ties the visual-only baseline, while the heavy multimodal
profile slightly reduces mAP. These results do not support a claim that ASR/audio
improves ActivityNet temporal grounding under the current fusion settings.

## Temporal Granularity Trade-off

| Setting | Candidate windows | Relative cost | Recall@10 | mAP@10 | Time (s) | Queries/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10s/5s | 4454 | 1.000x | 0.455988 | 0.283630 | 153.32 | 4.52 |
| 20s/10s | 2184 | 0.490x | 0.659452 | 0.423096 | 133.53 | 5.19 |

The coarser 20-second window with 10-second stride reaches higher tIoU-based
scores while using approximately half as many candidate windows. This does not
mean the localization is strictly more precise: the returned moments are wider.
Window and stride must therefore be reported with every temporal result.

## Discussion

The ActivityNet results support the current system as a measurable temporal
grounding pipeline for video-derived evidence. The strongest supported claim is
that the system can produce ranked temporal moments and can be evaluated
reproducibly using tIoU-based metrics. The current multimodal fusion settings are
best treated as exploratory ablations because they do not improve over
visual-only retrieval on this subset.

The temporal granularity experiment highlights a metric/design trade-off. Wider
windows reduce candidate-window cost and improve tIoU-based Recall@10/mAP@10 in
this setup, but they return coarser moments. If the report prioritizes precise
localization, the 10s/5s run should remain the baseline. If the report discusses
coarse event retrieval, the 20s/10s result can be reported as a separate
granularity ablation.

## Claim Boundary

Do not claim that this system outperforms WorldMM or any prior long-video system
unless the same dataset, task, and protocol are evaluated. Do not claim that
semantic/generated-summary memory improves temporal grounding until summary
evidence is indexed, searched, and evaluated in a controlled ablation. CASTLE
should be described as the downstream lifelog application setting rather than as
the source of quantitative tIoU-based evidence.
