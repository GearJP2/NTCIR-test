# NTCIR-test — WorldMM Validation

This repo tests the WorldMM (Dynamic Multimodal Memory Agent) approach against long-form video lifelog data. CASTLE2024 is used to build and explore the pipeline; ActivityNet Captions is used to measure retrieval quality where ground truth exists.

## Language

**WorldMM Approach**:
The dynamic multimodal memory agent design from the WorldMM paper — episodic memory indexing plus retrieval over long video, with optional LLM reasoning over retrieved context.
_Avoid_: CSAT pipeline, NTCIR submission system

**Development Dataset**:
CASTLE2024 — the primary lifelog corpus for building and qualitatively testing the memory agent. No ground-truth retrieval labels are available.
_Avoid_: Benchmark dataset, eval set, test set

**Evaluation Dataset**:
ActivityNet Captions — the held-out corpus with annotated timestamp intervals and sentence captions, used to score whether retrieved moments match known events.
_Avoid_: Training data, CASTLE substitute

**Semantic Query**:
A natural-language text description of the moment the user wants to find (e.g. "the woman does sit ups").
_Avoid_: Keyword search, topic ID, qid

**Retrieval**:
A single ranked moment returned by search — a time interval within a video, paired with a relevance score. (Granularity and search scope still to be confirmed.)
_Avoid_: Hit, chunk, segment (until defined precisely)
