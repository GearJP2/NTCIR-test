# Use ActivityNet as the Controlled Temporal Benchmark

ActivityNet Captions is the controlled quantitative benchmark for temporal grounding because it provides sentence-level timestamp annotations. Paper claims should report ActivityNet metrics as component-level evidence, using Recall@10 and mAP@10 with tIoU >= 0.3.

CASTLE2024 remains the downstream lifelog application setting, but it is removed from tIoU benchmark planning because it has no Ground Truth Moments. Existing CASTLE scripts can remain as manual inspection/demo support, but CASTLE results should not be used as quantitative temporal-localization evidence in the paper.

This decision does not imply that prior long-video systems lack long-video evaluation. Do not claim superiority over WorldMM or related systems unless the same dataset, task, and evaluation protocol are run.
