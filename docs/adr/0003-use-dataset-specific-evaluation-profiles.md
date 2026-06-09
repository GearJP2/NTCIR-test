# Use Dataset-Specific Evaluation Profiles

Evaluation settings are dataset-specific rather than global. ActivityNet validation uses a visual-heavy profile with single-video search, Top-10 results, and tIoU >= 0.3 for early Recall@10, while CASTLE qualitative testing uses a balanced lifelog profile without temporal ground-truth scoring because CASTLE2024 does not provide benchmark annotations.
