# Keep ActivityNet Captions Out of the Index

ActivityNet Captions annotations are used only to create Evaluation Queries and Ground Truth Moments, not as indexed retrieval content. This prevents label leakage: the system must retrieve moments from video-derived evidence such as frames, audio, ASR, or generated summaries rather than matching directly against the benchmark answer text.
