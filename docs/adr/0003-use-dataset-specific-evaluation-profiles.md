# Use Dataset-Specific Evaluation Profiles

Evaluation settings are profile-specific rather than global. ActivityNet Captions is the only quantitative benchmark for paper claims because it provides timestamped Ground Truth Moments. ActivityNet profiles use single-video search, Top-10 results, and tIoU >= 0.3 for Recall@10 and mAP@10. CASTLE tooling may remain for legacy/manual demos, but CASTLE2024 is not an evaluation profile target because it does not provide Ground Truth Moments.
