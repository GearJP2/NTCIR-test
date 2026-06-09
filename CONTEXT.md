# NTCIR-test — WorldMM Validation

This repo tests the WorldMM (Dynamic Multimodal Memory Agent) approach against long-form video lifelog data. CASTLE2024 is used to build and explore the pipeline; ActivityNet Captions is used to measure retrieval quality where ground truth exists.

## Language

**WorldMM Approach**:
The dynamic multimodal memory agent design from the WorldMM paper as adapted for this repo: multimodal memory indexing plus retrieval over long video. Benchmark evaluation focuses on ranked Video Moments only, without LLM reasoning.
_Avoid_: CSAT pipeline, NTCIR submission system

**Development Dataset**:
CASTLE2024 — the primary lifelog corpus for building and qualitatively testing the memory agent. No ground-truth retrieval labels are available.
_Avoid_: Benchmark dataset, eval set, test set

**Evaluation Dataset**:
ActivityNet Captions — the held-out corpus with annotated timestamp intervals and sentence captions, used to score whether retrieved moments match known events.
_Avoid_: Training data, CASTLE substitute

**Curated Query Set**:
A hand-written set of Semantic Queries used for manual CASTLE2024 inspection through the Search Interface. It guides tuning but is not a benchmark because CASTLE2024 has no Ground Truth Moments.
_Avoid_: CASTLE benchmark, qrels, labels

**Semantic Query**:
A natural-language text description of the moment the user wants to find (e.g. "the woman does sit ups").
_Avoid_: Keyword search, topic ID, qid

**Evaluation Query**:
A Semantic Query derived from one ActivityNet Captions sentence, paired with that sentence's timestamp interval as its single Ground Truth Moment.
_Avoid_: Topic, prompt, benchmark question

**Indexable Video Evidence**:
The video-derived material that may be indexed for retrieval, such as frames, audio, ASR output generated from audio, or model-derived summaries. ActivityNet Captions annotations are not Indexable Video Evidence because they are reserved for Evaluation Queries and Ground Truth Moments. Each item of evidence should preserve its source type, such as visual, audio, ASR, or generated summary.
_Avoid_: Ground truth captions, labels, answer text

**Retrieval**:
A single ranked Video Moment returned by search, paired with a relevance score.
_Avoid_: Hit, chunk, segment

**Search Result Set**:
The ranked list of retrieved video moments returned for one Semantic Query, normally limited to the top 10 results.
_Avoid_: Search output, result batch, hits list

**Video Moment**:
A contiguous time interval within a video, identified by media ID, start timestamp, and end timestamp. This is the canonical output unit of the Search Interface and the unit compared against ActivityNet Captions annotations.
_Avoid_: Audio chunk, keyframe, clip, segment

**Evidence**:
The source-specific support behind a Video Moment score, such as a visual keyframe match, audio embedding match, ASR transcript match, or generated-summary match.
_Avoid_: Debug data, raw hit

**Moment Search Response**:
The canonical search API response: one selected media ID, one Semantic Query, and a ranked Search Result Set of Video Moments with scores, thumbnail timestamps, and Evidence.
_Avoid_: Episodic search response, ranked hits

**Ground Truth Moment**:
An annotated ActivityNet Captions time interval and caption pair used as the expected answer for retrieval evaluation.
_Avoid_: Label, benchmark row, answer key

**Temporal Match**:
A retrieved Video Moment is considered correct when its temporal intersection-over-union with a Ground Truth Moment meets the chosen threshold. Early validation uses tIoU >= 0.3.
_Avoid_: Exact timestamp match, segment ID match

**Recall@10**:
The early validation metric: whether at least one of the top 10 retrieved Video Moments temporally matches the Ground Truth Moment for a Semantic Query.
_Avoid_: Accuracy, success rate

**Evaluation Profile**:
A named evaluation setting that defines dataset-specific retrieval assumptions, especially modality weights and matching thresholds. ActivityNet validation should use a visual-heavy profile, while CASTLE qualitative testing may use a more balanced lifelog profile.
_Avoid_: Config preset, experiment mode

**Evaluation Manifest**:
A reproducible file that lists the videos selected for evaluation, their local media paths, durations, and Evaluation Queries with Ground Truth Moments.
_Avoid_: Dataset dump, annotation file, loader cache

**Search Interface**:
The web experience where a user selects and watches one video, enters a Semantic Query, and receives ranked timestamped moments with scores for that selected video.
_Avoid_: Demo page, dashboard, UI

**Search Scope**:
The set of video moments eligible to be returned for a Semantic Query. For the first validation flow, this is expected to be moments within the currently selected long video.
_Avoid_: Corpus filter, media filter, dataset slice
