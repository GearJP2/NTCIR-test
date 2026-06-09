# Use Late Fusion for Video Moment Ranking

Retrieval results from visual, audio, ASR, and generated-summary evidence are normalized into Video Moments before ranking is finalized. We use late fusion rather than a single fused embedding so evaluation can explain which evidence type contributed to each result and so modality weights can be adjusted, especially for ActivityNet action captions where visual evidence may matter more than ASR.
