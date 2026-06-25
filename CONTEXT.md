# NTCIR-test — CASTLE Semantic Event Retrieval

This repository develops a hierarchical multimodal event-retrieval system for
long CASTLE2024 lifelog recordings. CASTLE is the only active research corpus.
ActivityNet code and documents are retained only as legacy implementation
evidence and must not drive new interfaces or claims.

## Language

**CASTLE Recording**:
A long primary lifelog video and the multimodal sources associated with its
participant and session.
_Avoid_: ActivityNet video, benchmark clip

**Canonical Timeline**:
The documented absolute millisecond timeline onto which video, transcript,
heart-rate, gaze, and optional thermal evidence are aligned.
_Avoid_: inferred clock, local modality time

**Event Record**:
The canonical manifest entry describing one time interval in a CASTLE
Recording. It owns identifiers, core timestamps, hierarchy, source coverage,
metadata summaries, confidence, and references to raw evidence.
_Avoid_: chunk row, retrieval hit

**Core Event Interval**:
The non-expanded start and end timestamps assigned by fixed-window or semantic
segmentation. Evaluation and reported timestamps use this interval.
_Avoid_: context window

**Retrieval Context**:
Optional time before and after a Core Event Interval used to preserve evidence
near event edges. It never changes the reported Core Event Interval.
_Avoid_: corrected boundary

**Micro Event**:
A short semantically coherent Event Record, initially constrained to roughly
5–60 seconds. It may have one Macro Event parent.
_Avoid_: clip, segment

**Macro Event**:
A broader Event Record grouping related Micro Events over up to several
minutes.
_Avoid_: session, arbitrary long window

**Fallback Window**:
A fixed-duration overlapping Event Record retained as a recall channel when
semantic segmentation fails. Initial baselines are 30-second overlapping and
120-second non-overlapping windows.
_Avoid_: semantic event

**Semantic Event**:
A Micro Event or Macro Event created from documented visual and optional
transcript transition signals, with duration post-processing and a boundary
confidence score.
_Avoid_: fixed window

**Aligned Evidence**:
Transcript spans, heart-rate samples, valid gaze fixations, attended objects,
and optional thermal images associated with an Event Record through the
Canonical Timeline.
_Avoid_: silently interpolated metadata

**Coverage**:
Explicit per-modality availability for an Event Record. Missing evidence stays
missing and is never represented by fabricated default measurements.
_Avoid_: completeness score

**Event Manifest**:
A versioned JSONL or Parquet collection of validated Event Records. It is the
handoff between dataset preparation, indexing, retrieval, and evaluation.
_Avoid_: ActivityNet evaluation manifest

**Direct Video Evidence**:
Video-derived embeddings for an Event Record, Micro Event, Macro Event,
Fallback Window, or keyframe. It is indexed separately from textual evidence.
_Avoid_: caption vector

**Auxiliary Evidence**:
Transcript, OCR, gaze-grounded objects, heart-rate state, and optional sparse
thermal information aligned to an Event Record.
_Avoid_: primary boundary signal

**Candidate Event**:
An Event Record returned by one retrieval channel before fusion, neighbour
expansion, or temporal refinement.
_Avoid_: final answer

**Ranked Event Result**:
A CASTLE Recording identifier, Core Event Interval, score, and source-specific
evidence returned for a topic query.
_Avoid_: ActivityNet Video Moment

**Topic Query**:
A natural-language CASTLE task description decomposed into requested objects,
actions, people, scenes, speech clues, attention, physiological state, and
temporal order.
_Avoid_: ActivityNet caption query

## Active Research Rules

- Video determines event boundaries initially; metadata follows those
  boundaries.
- Heart rate, gaze, and thermal evidence do not create precise event boundaries
  in the baseline system.
- Thermal evidence is optional and confidence-weighted because coverage may be
  sparse.
- Fixed windows remain reproducible baselines and a fallback retrieval channel.
- Direct video, caption, transcript, keyframe/OCR, gaze, and physiological
  evidence remain separate representations until fusion.
- Every generated Event Manifest includes a processing version.
- Quantitative claims require an explicit CASTLE task protocol, labels, or
  official evaluation output. Do not reuse ActivityNet metrics as CASTLE
  evidence.
