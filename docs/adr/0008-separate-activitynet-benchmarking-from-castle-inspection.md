# Use ActivityNet as the Paper Benchmark

ActivityNet Captions is the quantitative Evaluation Dataset because it provides sentence-level timestamp annotations. Paper claims should report ActivityNet metrics only, using Recall@10 and mAP@10 with tIoU >= 0.3.

CASTLE2024 is removed from benchmark planning because it has no Ground Truth Moments. Existing CASTLE scripts can remain as legacy/manual demo support, but CASTLE results should not be used as evaluation evidence in the paper.
