# Research Plan: Semantic Multimodal Event Retrieval for CASTLE

## 1. Working Title

**Hierarchical Semantic Video Indexing with Temporally Grounded Auxiliary
Evidence for Multimodal Event Retrieval**

## 2. Research Objective

Develop and evaluate an event-retrieval system for long CASTLE recordings that:

1. Divides video into semantically coherent macro and micro events.
2. Indexes events directly with video embeddings instead of relying only on
   generated captions.
3. Aligns transcripts, gaze, heart rate, and thermal imagery to the event
   timeline.
4. Uses auxiliary signals during candidate retrieval and reranking.
5. Returns ranked video identifiers and event timestamps.

The main contribution is not the use of an agent or a particular embedding
model. It is the construction and evaluation of a hierarchical, temporally
grounded multimodal event index.

## 3. Main Hypothesis

> Direct video embeddings and semantic event boundaries preserve visual and
> temporal evidence lost by caption-first, fixed-window systems. Temporally
> grounded gaze and physiological-thermal evidence provide complementary
> improvements for suitable event topics.

## 4. Research Questions

### RQ1: Semantic Chunking

Does semantic event chunking improve retrieval over fixed 30-second and
120-second windows?

### RQ2: Direct Video Indexing

Do direct video embeddings retrieve relevant events that caption and transcript
indexes miss?

### RQ3: Hierarchical Retrieval

Does searching macro events, micro events, keyframes, and fallback windows
together improve candidate recall and timestamp precision?

### RQ4: Gaze Grounding

Does converting gaze coordinates into attended objects improve retrieval over
using raw gaze statistics or ignoring gaze?

### RQ5: Physiological-Thermal Fusion

Does joint heart-rate and thermal activity-state estimation improve retrieval
for exertion, passive-arousal, and heat-related topics?

### RQ6: Event-Level Integration

Does indexing auxiliary evidence as aligned event features outperform using it
only as selectively requested evidence at answer or reranking time?

## 5. Scope

### Core Contribution

- Semantic video chunking
- Direct video-vector indexing
- Hierarchical event retrieval
- Timestamp-aware multimodal fusion
- Event boundary refinement

### Secondary Contribution

- Gaze-to-object grounding
- Heart-rate and thermal activity-state fusion

### Optional Extension

- Query-routing agent
- Learned fusion weights
- Auxiliary personal or Narrative Clip photo evidence

Auxiliary photos should remain optional because they may be sparse, randomly
captured, or weakly associated with the primary video timeline.

## 6. System Stages

### Stage A: Shared Timeline

Create one canonical timeline and common identifiers:

```text
video_id
event_id
start_timestamp
end_timestamp
participant_id
source_confidence
```

Map transcripts, gaze, heart-rate samples, and thermal images onto this
timeline. Record missing data instead of filling it with unsupported values.

### Stage B: Semantic Video Segmentation

1. Sample short video windows.
2. Extract lightweight visual features.
3. calculate visual-semantic differences between adjacent windows.
4. Add transcript changes when transcripts are available.
5. Detect candidate event boundaries.
6. Merge chunks below a minimum duration.
7. Split chunks above a maximum duration.
8. Preserve overlapping fixed windows as a recall fallback.

Initial suggested limits:

- Feature window: 2-5 seconds
- Minimum event: 5-10 seconds
- Maximum micro event: 30-60 seconds
- Macro event: up to several minutes
- Boundary context: 3-5 seconds
- Fixed fallback: 30 seconds with 10-second overlap

### Stage C: Event Index Construction

For each event, store:

- Direct video embedding
- Macro-event and micro-event relationship
- Keyframe embeddings
- Caption and caption embedding
- Transcript text and embedding
- OCR and detected objects
- Gaze-derived attended objects
- Heart-rate/thermal activity-state probabilities
- Start/end timestamps and confidence

### Stage D: Query Processing

Decompose each topic into:

- Objects
- Actions
- People
- Scene/location
- Speech or named entities
- Attention requirements
- Physiological intensity
- Temporal order
- Positive and negative evidence

Use the decomposition to select retrieval channels and weights.

### Stage E: Retrieval and Fusion

Retrieve candidates independently from:

- Direct video index
- Caption index
- Transcript dense and lexical indexes
- Keyframe/OCR index
- Gaze-derived attended-object index
- Activity-state metadata

Begin with Reciprocal Rank Fusion. Introduce learned or query-dependent weights
only after the baseline is stable.

### Stage F: Temporal Refinement and Reranking

1. Expand each candidate to neighbouring events.
2. Search shorter windows inside the candidate interval.
3. Analyze the candidate video with a video-capable model.
4. Present structured evidence to the multimodal reranker.
5. Penalize contradictions and unsupported evidence.
6. Refine start/end boundaries.
7. Remove duplicate or heavily overlapping results.

## 7. Implementation Phases

### Phase 0: Dataset Audit

**Goal:** Establish exactly which modalities, timestamps, participants, and
missing periods are available.

Deliverables:

- File and modality inventory
- Timestamp-format table
- Participant/modality coverage matrix
- Sample synchronized timeline
- Formal-run output validator

Decision gate:

> Do not train auxiliary models until modality coverage and temporal alignment
> are understood.

### Phase 1: Reproducible Retrieval Baselines

Implement:

- Fixed 120-second direct-video retrieval
- Fixed 30-second overlapping retrieval
- Caption-only retrieval
- Transcript-only retrieval
- Caption + transcript fusion

Deliverables:

- Indexed baseline corpus
- Candidate `Recall@K`
- Final accuracy
- Retrieval latency and storage measurements

Decision gate:

> Continue only when evaluation is deterministic and errors can be traced to
> candidate retrieval or reranking.

### Phase 2: Semantic Event Chunking

Implement:

- Visual boundary score
- Minimum/maximum duration constraints
- Macro/micro hierarchy
- Fixed-window fallback

Compare:

```text
Fixed 120 s
Fixed 30 s overlapping
Semantic events
Semantic events + fixed fallback
```

Success criterion:

> Semantic-plus-fallback retrieval improves candidate recall or timestamp
> precision without an unacceptable indexing-cost increase.

### Phase 3: Direct Video and Text Complementarity

Implement and compare:

- Video vectors only
- Captions only
- Transcript only
- Video + captions
- Video + captions + transcript

Analyze failures by topic category. Identify events retrieved only by direct
video vectors and events retrieved only by text.

Primary paper evidence:

> Direct video indexing contributes complementary recall rather than merely
> duplicating caption retrieval.

### Phase 4: Gaze-to-Object Grounding

Pipeline:

```text
Valid fixation
  + synchronized frame
  + object/region detection
  -> attended object with confidence
```

Use fields such as:

- `FPOGX`, `FPOGY`
- `FPOGD`
- `FPOGV`
- `FPOGID`
- Associated frame/timestamp

Compare:

- No gaze
- Raw fixation statistics
- Gaze heatmaps
- Gaze-grounded attended objects

Only use valid fixations. Account for eye-tracker uncertainty by expanding the
gaze point into a small region rather than treating it as an exact pixel.

### Phase 5: Heart-Rate and Thermal Fusion

Heart-rate features:

- Personalized mean and percentile
- Baseline difference
- Variability
- Rising/falling trend
- Recovery trend
- Missing-data confidence

Thermal features:

- Human/body-region presence
- Posture and motion
- Heat distribution and change
- People count
- Environmental heat-source evidence

Predict coarse states:

- Passive rest
- Passive arousal
- Light activity
- Physical exertion
- Recovery
- Heat-source interaction

Compare:

- Heart rate only
- Thermal only
- Heart rate + thermal
- Heart rate + thermal + RGB evidence

Do not claim exact activity recognition from heart rate and thermal alone.

### Phase 6: Full Event Retrieval System

Combine:

- Semantic hierarchical video index
- Fixed-window fallback
- Transcript and caption retrieval
- Gaze-grounded evidence
- Physiological-thermal state evidence
- Timestamp-aware fusion
- Neighbour expansion
- Video-aware reranking

Produce the formal-run submission and save all component scores for analysis.

## 8. Experimental Matrix

| Run | Configuration |
|---|---|
| B1 | Fixed 120 s + video vectors |
| B2 | Fixed 30 s overlapping + video vectors |
| B3 | Captions only |
| B4 | Transcripts only |
| S1 | Semantic chunks + video vectors |
| S2 | Semantic + fixed fallback + video vectors |
| S3 | S2 + captions + transcripts |
| A1 | S3 + raw gaze statistics |
| A2 | S3 + gaze-grounded objects |
| A3 | S3 + heart rate |
| A4 | S3 + thermal |
| A5 | S3 + heart-rate/thermal fusion |
| F1 | Full system without reranking |
| F2 | Full system with temporal refinement and reranking |

## 9. Evaluation

### Primary Measures

- Official CSAT accuracy
- Candidate `Recall@10`, `Recall@50`, and `Recall@100`
- Mean reciprocal rank, if ranked relevance can be measured
- Temporal overlap or boundary error, when reference intervals are available

### Efficiency Measures

- Indexing time
- Index size
- Query latency
- Reranking cost per topic

### Topic Categories

- Objects
- Actions
- Locations/scenes
- Conversations/entities
- Attention or looking behaviour
- Exercise/physiological activity
- Heat-source interaction
- Multi-step or ordered events

### Statistical Analysis

- Report per-topic paired differences between runs.
- Use bootstrap confidence intervals where the number of topics permits.
- Avoid claiming improvements from a small number of favourable examples.

## 10. Comparison with MARS

The comparison should remain precise:

| Dimension | MARS-style approach | Proposed approach |
|---|---|---|
| Primary video representation | Captioned/summarized evidence | Direct video vectors plus text |
| Segmentation | Fixed-stride evidence units | Semantic macro/micro events plus fallback |
| Auxiliary data | Selectively requested evidence | Pre-aligned event features used in retrieval |
| Gaze | Auxiliary evidence | Fixation-to-object grounding |
| Heart rate/thermal | Separate evidence sources | Joint activity-state estimation |
| Output focus | Question answering | Ranked event IDs and timestamps |

The research should not claim that MARS lacks multimodality. The claim is that
the proposed method integrates modalities at a different stage and with
stronger temporal grounding.

## 11. Risks and Fallbacks

### Semantic Boundaries Do Not Improve Retrieval

Fallback: retain semantic chunks as one channel and fuse them with fixed
overlapping windows.

### Direct Video Embeddings Underperform Captions

Fallback: position them as complementary recall evidence and analyze the topic
classes where they help.

### Auxiliary Coverage Is Sparse

Fallback: make auxiliary scores optional and confidence-weighted. Evaluate only
on topics/events with valid coverage as a secondary analysis.

### No Labels for Activity-State Training

Fallback options:

- Weak labels derived from clearly identifiable RGB events
- Rule-based physiological states
- Small manually annotated development subset
- Unsupervised state clustering followed by human interpretation

Avoid training and testing on overlapping time windows from the same event.

### Gaze Does Not Match Primary Video View

Fallback: verify the gaze reference frame first. If alignment is unavailable,
use fixation behavior statistics rather than claiming object-level grounding.

## 12. Suggested 12-Week Schedule

| Week | Work |
|---|---|
| 1 | Dataset audit, timestamp mapping, evaluation script |
| 2 | Fixed-window video baseline |
| 3 | Caption and transcript baselines |
| 4 | Semantic boundary detector |
| 5 | Macro/micro event index and fallback windows |
| 6 | Direct-video/text fusion and error analysis |
| 7 | Gaze alignment and fixation processing |
| 8 | Gaze-to-object grounding experiment |
| 9 | Heart-rate feature pipeline |
| 10 | Thermal features and joint state classifier |
| 11 | Full-system fusion, temporal refinement, reranking |
| 12 | Ablations, statistical analysis, formal run, writing |

If time is limited, complete Phases 0-3 first. They contain the core research
claim. Gaze and physiological-thermal fusion should be added only after that
claim has a stable experimental foundation.

## 13. Minimum Publishable System

The minimum credible contribution is:

1. Semantic macro/micro video segmentation
2. Hierarchical direct-video index
3. Fixed-window fallback
4. Caption and transcript fusion
5. Temporal refinement
6. Careful ablation and topic-level analysis

Gaze grounding and heart-rate/thermal fusion strengthen the paper but should
not be allowed to delay the core system.

## 14. Immediate Next Actions

1. Inventory the CASTLE files and timestamp formats.
2. Build the official-topic parser and submission validator.
3. Select a small development subset containing diverse event types.
4. Implement fixed 30-second and 120-second retrieval baselines.
5. Measure candidate recall before implementing semantic chunking.
6. Create a common event schema used by every modality.

