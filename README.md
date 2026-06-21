# Semantic Event Video Indexing for Multimodal Lifelog Retrieval

## 1. Research Idea

This project proposes an event-retrieval method for long, untrimmed CASTLE
lifelog videos. Instead of dividing every video into fixed two-minute clips,
the system first detects meaningful semantic event boundaries. Each resulting
event is then indexed directly using video embeddings, with captions,
transcripts, OCR, and sensor information retained as complementary evidence.

The central hypothesis is:

> Semantic event-level video indexing preserves visual information that may be
> lost during captioning, while reducing the representation dilution caused by
> embedding long clips containing several unrelated activities.

The goal is not to replace caption or transcript retrieval. The system combines
direct video retrieval with textual and auxiliary retrieval to improve candidate
recall and final timestamp accuracy.

## 2. Motivation

Many existing video-retrieval systems convert video into captions and then use
mature text-retrieval methods. This approach is efficient and interpretable,
but captioning can discard information before retrieval:

- Brief or secondary actions may not appear in the caption.
- Fine visual details, spatial relationships, and unusual objects may be lost.
- A captioner's vocabulary may not match the topic wording.
- Multiple actions may be compressed into a generic description.
- Temporal order, such as entering versus leaving, may be misunderstood.

Direct video indexing can preserve latent visual evidence. However, indexing a
long fixed window with one vector also compresses too many events. Semantic
chunking is intended to create more coherent units for direct video embedding.

## 3. Proposed Method

### 3.1 Multimodal Semantic Chunking

The video is initially processed in short windows, such as two to five seconds.
The system calculates possible event-boundary signals from:

- Visual embedding changes
- Scene and camera-motion changes
- Object, person, or action changes
- Transcript topic and speaker changes
- Audio-event or silence changes
- Location and accelerometer changes
- Gaze-target changes
- Heart-rate intensity or trend changes

A general boundary score can be written as:

```text
B(t) = wv*Dv(t) + wt*Dt(t) + wa*Da(t) + ws*Ds(t)
```

Here, `Dv`, `Dt`, `Da`, and `Ds` represent visual, textual, audio, and sensor
changes around time `t`. The weights can be tuned using development data.

Very short chunks should be merged, while very long chunks should be split.
Small overlaps or context margins should be retained around detected
boundaries so that actions occurring at chunk edges are not lost.

### 3.2 Hierarchical Event Representation

The system should maintain representations at several temporal levels:

```text
Recording/session
  -> Macro event: broad activity or episode
      -> Micro events: short actions within the episode
          -> Keyframes or short fallback windows
```

Suggested levels:

- Macro-event vectors for broad contextual retrieval
- Micro-event vectors for precise action retrieval
- Overlapping 15- or 30-second vectors as a segmentation fallback
- Keyframe vectors for objects, people, places, and visible text

Fixed overlapping windows remain important because semantic boundary detection
will not always be correct.

### 3.3 Multi-Vector Event Index

Each event should have linked but separate representations:

```json
{
  "event_id": "E102",
  "video_id": "V001",
  "start_time": 610,
  "end_time": 648,
  "parent_event_id": "E100",
  "video_vector": "...",
  "caption_vector": "...",
  "transcript_vector": "...",
  "keyframe_vectors": ["...", "..."],
  "caption": "A participant reads a menu while sitting at a table.",
  "transcript": "I think I will order this one.",
  "ocr": ["MENU", "COFFEE"],
  "objects": ["menu", "cup", "table"],
  "actions": ["reading", "sitting"],
  "scene": "restaurant",
  "heart_rate_state": "resting",
  "heart_rate_trend": "stable"
}
```

Direct video embeddings, potentially produced with VLM2Vec-V2, are the main
visual retrieval representation. Text and sensor fields are not collapsed into
the same vector because their importance depends on the query.

### 3.4 Query Decomposition and Routing

Each topic is decomposed into retrieval requirements:

```json
{
  "objects": ["stairs"],
  "actions": ["walking upward"],
  "scene": ["indoor"],
  "speech_clues": [],
  "physiological_intensity": "moderate",
  "temporal_order": ["approach stairs", "climb stairs"]
}
```

The system selects modalities and weights based on these requirements:

- Visual action topics emphasize direct video embeddings.
- Conversation topics emphasize transcripts and audio.
- Sign, screen, and product topics emphasize OCR.
- Exercise topics use video, motion, and heart-rate evidence.
- Multi-step topics emphasize temporal neighbours and macro-event context.

### 3.5 Retrieval and Fusion

The query is searched against multiple indexes:

1. Direct video-event embeddings
2. Caption embeddings
3. Transcript dense and lexical indexes
4. OCR and metadata indexes
5. Sensor-derived activity or intensity fields

Candidate rankings can initially be combined with Reciprocal Rank Fusion:

```text
RRF(event) = sum(weight_m / (k + rank_m(event)))
```

Later, a learned or query-dependent fusion model can be introduced:

```text
Score(event, query) =
    wv(query)*Svideo +
    wc(query)*Scaption +
    wt(query)*Stranscript +
    wo(query)*SOCR +
    ws(query)*Ssensor
```

Sensor evidence should generally provide a bonus or penalty, rather than act as
a strict filter.

### 3.6 Temporal Refinement and Reranking

After coarse retrieval:

1. Retrieve the candidate event and neighbouring events.
2. Inspect shorter windows within the candidate interval.
3. Collect video, caption, transcript, OCR, and sensor evidence.
4. Use a video-aware model to score whether the event satisfies every topic
   requirement.
5. Refine the start and end timestamps.
6. Return the highest-ranked intervals in the required submission format.

This stage connects corpus-level retrieval with temporal grounding.

## 4. Proposed Architecture

```text
Raw multimodal recording
        |
        v
Short-window feature extraction
        |
        v
Multimodal boundary detection
        |
        v
Macro events + micro events + fallback windows
        |
        +--> Direct video embeddings
        +--> Captions and caption embeddings
        +--> Transcript and OCR indexes
        +--> Sensor and metadata summaries
        |
        v
Hierarchical multimodal event index

Topic query
        |
        v
Query decomposition and modality routing
        |
        v
Parallel retrieval from relevant indexes
        |
        v
Timestamp-aware candidate fusion
        |
        v
Neighbour expansion and short-window refinement
        |
        v
Video-aware multimodal reranking
        |
        v
Ranked video IDs and timestamps
```

## 5. Potential Contributions

Using an existing video embedding model alone is an implementation choice. A
stronger methodological contribution would consist of:

1. **Multimodal semantic event segmentation**

   Event boundaries are detected using visual, textual, audio, and sensor
   transitions rather than only shot changes.

2. **Hierarchical direct video indexing**

   The index preserves macro-event, micro-event, fallback-window, and keyframe
   representations instead of assigning one vector to a long recording.

3. **Query-adaptive multimodal retrieval**

   The method changes retrieval channels and fusion weights according to the
   evidence requested by each topic.

4. **Boundary-aware temporal refinement**

   Retrieved semantic chunks are expanded and rescored at a finer scale to
   produce accurate event timestamps.

5. **Analysis of direct video and caption complementarity**

   The study identifies which topic categories benefit from direct video
   vectors, captions, transcripts, and sensors.

An appropriate research framing is:

> A hierarchical, boundary-aware direct video indexing framework for event
> retrieval in long, untrimmed multimodal lifelog recordings.

## 6. Role of Heart-Rate and Auxiliary Data

Heart rate is better treated as an activity-intensity signal than as an exact
activity label. It may classify personalized states such as:

- Resting
- Light activity
- Moderate activity
- Vigorous activity
- Recovery
- Elevated heart rate without corresponding physical motion

Useful features include:

- Mean, minimum, maximum, and variation
- Difference from the participant's baseline
- Personalized percentile
- Increasing or decreasing slope
- Recovery rate
- Missing-data and reliability indicators

Heart rate becomes valuable when combined with video and accelerometer signals.
For example, rising heart rate, increasing motion, and a visual transition to
stairs together provide stronger evidence of physical activity. Heart rate
should have little or no weight for topics such as reading a sign.

## 7. Experimental Plan

### 7.1 Main Comparisons

| Run | Segmentation and retrieval configuration |
|---|---|
| A | Fixed 120-second chunks with direct video vectors |
| B | Fixed 30-second overlapping chunks with direct video vectors |
| C | Semantic chunks with caption retrieval only |
| D | Semantic chunks with direct video retrieval only |
| E | Semantic chunks with video and caption fusion |
| F | Hierarchical semantic chunks plus fixed fallback windows |
| G | Full multimodal retrieval with transcripts, OCR, and sensors |
| H | Full system with temporal refinement and video-aware reranking |

### 7.2 Ablation Studies

- Visual-only boundaries versus multimodal boundaries
- Flat versus hierarchical indexing
- With and without fixed fallback windows
- With and without direct video embeddings
- With and without captions
- Static versus query-adaptive fusion weights
- With and without heart-rate and motion features
- With and without neighbouring-event expansion

### 7.3 Evaluation Measures

- Candidate `Recall@10`, `Recall@50`, and `Recall@100`
- Final task accuracy
- Temporal localization accuracy or overlap, if reference boundaries exist
- Accuracy by topic category
- Storage requirements
- Indexing time
- Query latency

Candidate recall must be measured before reranking. If the correct event is not
retrieved, the reranker cannot recover it.

### 7.4 Topic-Level Analysis

Results should be separated into:

- Object-centric topics
- Fine-grained action topics
- Scene and location topics
- Conversation and named-entity topics
- Multi-step or temporally ordered events
- Physiological or exercise-related events
- Visually implicit events that captions frequently omit

The expected result is not necessarily that direct video retrieval wins every
category. A likely and valuable finding is that direct video vectors improve
visual-action recall, while text retrieval performs better on explicit entities
and dialogue. Their fusion may provide the highest overall accuracy.

## 8. Risks and Mitigations

### Incorrect Semantic Boundaries

Boundary errors can hide events or join unrelated activities.

**Mitigation:** retain overlapping fixed-window and keyframe indexes, and add
context around chunk boundaries.

### Representation Dilution

Long semantic events may still contain several actions.

**Mitigation:** use macro and micro events, maximum chunk lengths, and
multi-vector representations.

### Video-Model Domain Mismatch

General models may be trained on short, edited, third-person Internet videos
rather than repetitive lifelog recordings.

**Mitigation:** use development-topic adaptation, hard-negative mining, query
expansion, and hybrid text-video retrieval.

### Auxiliary-Data Noise

Heart rate, OCR, transcripts, and other sensors may be missing or incorrect.

**Mitigation:** store reliability values, avoid strict sensor filters, and treat
missing evidence as neutral.

### Computational Cost

Overlapping multi-scale video embeddings can be expensive.

**Mitigation:** use lightweight features for boundary detection, encode semantic
events offline, and apply expensive video analysis only to top candidates.

## 9. Practical Development Order

1. Implement fixed 30-second overlapping video indexing as a reproducible
   baseline.
2. Implement visual semantic boundary detection and event chunk generation.
3. Add minimum and maximum event lengths plus boundary context.
4. Build direct video, caption, transcript, and keyframe indexes.
5. Add Reciprocal Rank Fusion and timestamp alignment.
6. Evaluate semantic chunks against fixed windows using candidate recall.
7. Add transcript, OCR, motion, and heart-rate features incrementally.
8. Introduce macro/micro hierarchy and neighbouring-event expansion.
9. Add video-aware reranking and fine timestamp refinement.
10. Complete ablations, topic-level error analysis, and efficiency reporting.

## 10. Possible Title

**Semantic Event Video Indexing for Multimodal Lifelog Retrieval**

Alternative:

**Hierarchical Boundary-Aware Video Indexing for Event Retrieval in Long
Multimodal Lifelogs**

## 11. Short Proposal Summary

This work proposes a hierarchical event-retrieval framework for long,
untrimmed multimodal lifelog video. It first segments recordings into semantic
macro and micro events using changes across video, transcript, audio, and
sensor streams. Each event is indexed directly with video embeddings while
captions, transcripts, OCR, keyframes, and physiological signals are preserved
as complementary searchable evidence. At query time, the system decomposes the
topic, selects relevant modalities, fuses candidates across indexes, expands
temporal context, and refines event boundaries through video-aware reranking.
The study evaluates whether semantic direct-video indexing preserves evidence
lost by caption-only systems and investigates its complementarity with textual
and sensor-based retrieval on the CASTLE dataset.
