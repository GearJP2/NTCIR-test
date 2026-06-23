# Focused Work Plan: Metadata Preparation and Semantic Video Chunking

## 1. Scope

This work package owns two parts of the CASTLE event-retrieval system:

1. Preparing and aligning multimodal metadata.
2. Dividing long videos into meaningful semantic events.

The downstream retrieval team should receive clean event records and media
references. Retrieval models and final reranking are outside this work package.

## 2. Available Inputs

- Primary videos
- Timestamped transcripts
- Heart-rate CSV files
- Gaze/eye-tracking CSV files
- Sparse timestamped thermal images, if useful

Personal-device and Narrative Clip photos should remain optional. They should
not be treated as frames extracted from the primary video.

## 3. Main Output

Produce one event manifest in JSONL or Parquet:

```json
{
  "event_id": "P01_V003_E0012",
  "participant_id": "P01",
  "video_id": "V003",
  "start_ms": 610000,
  "end_ms": 648000,
  "duration_ms": 38000,
  "parent_event_id": "P01_V003_M0003",
  "boundary_confidence": 0.82,
  "video_uri": "...",
  "transcript": "...",
  "heart_rate": {
    "mean_bpm": 105.2,
    "min_bpm": 92.0,
    "max_bpm": 117.0,
    "std_bpm": 6.4,
    "slope_bpm_s": 0.21,
    "baseline_delta": 31.0,
    "valid_ratio": 0.96
  },
  "gaze": {
    "valid_fixation_count": 14,
    "total_fixation_duration_ms": 6200,
    "mean_fixation_duration_ms": 443,
    "attended_objects": ["laptop"],
    "valid_ratio": 0.81
  },
  "thermal": {
    "image_ids": [],
    "image_count": 0,
    "nearest_time_delta_ms": null,
    "valid": false,
    "note": "optional sparse evidence only"
  },
  "coverage": {
    "video": true,
    "transcript": true,
    "heart_rate": true,
    "gaze": true,
    "thermal": false
  }
}
```

Keep raw measurements in separate files. The event manifest should contain
summaries, confidence values, and pointers to raw evidence.

## 4. Canonical Timeline

Use Unix time in milliseconds, or another single documented absolute format,
for every modality.

Create a synchronization table:

| source | original time format | timezone | offset | canonical field |
|---|---|---|---:|---|
| video | dataset timestamp | documented zone | 0 ms | `timestamp_ms` |
| transcript | dataset timestamp | documented zone | measured | `timestamp_ms` |
| heart rate | CSV timestamp | documented zone | measured | `timestamp_ms` |
| gaze | `TIME`/`TIME_TICK` plus session start | session zone | measured | `timestamp_ms` |
| thermal | filename/EXIF/manifest timestamp, if available | documented zone | measured | `timestamp_ms` |

Do not assume `TIME` in the gaze CSV is an absolute date. It may be elapsed
recording time and require the recording/session start timestamp.

## 5. Metadata Preparation

### 5.1 Heart Rate

Preserve:

- Timestamp
- BPM
- Device validity fields
- Participant/session identifier

Clean:

- Remove impossible values according to documented device limits.
- Mark missing and duplicated timestamps.
- Do not silently interpolate long gaps.
- Resample only for feature calculation, not as a replacement for raw data.

Compute over short overlapping windows:

- Mean, minimum, maximum, and standard deviation
- Median and interquartile range
- Slope
- Difference from participant baseline
- Participant percentile
- Valid-sample ratio

Suggested starting window: 15 seconds with a 5-second step.

### 5.2 Gaze

Use fixation fields when valid:

- `FPOGX`, `FPOGY`
- `FPOGS`, `FPOGD`
- `FPOGID`, `FPOGV`
- Associated frame or timestamp

Processing:

1. Recover the absolute timestamp for every gaze sample.
2. Keep only valid fixation samples for semantic attention features.
3. Convert normalized coordinates into frame coordinates.
4. Expand each gaze point into an uncertainty radius.
5. Intersect the gaze region with detected objects or semantic regions.
6. Aggregate attended objects and fixation durations per event.

Keep raw gaze position and validity so results can be audited.

### 5.3 Thermal Images

Thermal should be treated as optional sparse evidence, not a core metadata
stream. If the dataset only contains about 50 thermal images, it will usually
be too sparse to support reliable activity classification or event chunking.
Its best role is as a weak contextual cue for a small number of events.

Preserve, when available:

- Image identifier/path
- Capture timestamp
- Participant/session identifier
- Original thermal values when available

Possible exploratory features:

- Human/body presence
- Heat distribution
- Environmental heat-source evidence
- Confidence

Do not reduce the entire image to one average temperature. Thermal images may
represent environmental objects rather than body temperature.

Associate sparse thermal images with events using a documented tolerance. A
starting value could be the event interval plus or minus 5-10 seconds, but it
must be validated against the dataset capture process. If coverage is very low,
report thermal coverage and exclude it from the main system comparison.

### 5.4 Transcript

Preserve:

- Original text
- Start and end timestamps
- Speaker, when available
- Original record identifier

Attach all transcript spans that overlap an event. Preserve partial overlap
instead of forcing each transcript record into exactly one event.

## 6. Semantic Chunking Method

### 6.1 Baselines

Always build these first:

- 120-second non-overlapping windows
- 30-second windows with 10-second overlap

They provide a fair comparison and protect retrieval recall when semantic
boundaries are incorrect.

### 6.2 Visual Boundary Signal

Process the video in 2-5 second units and calculate:

- Visual embedding distance between adjacent units
- Scene-change score
- Object/person-set change
- Motion change
- Optional transcript-semantic change

Initial boundary score:

```text
B(t) =
    w_visual * visual_distance(t)
  + w_scene  * scene_change(t)
  + w_object * object_change(t)
  + w_motion * motion_change(t)
  + w_text   * transcript_change(t)
```

Heart rate should not define precise boundaries because physiological response
can lag behind visible activity. Gaze may support interpretation of an event,
but should initially be metadata rather than a primary boundary input. Thermal
should not be used for boundary detection unless later inspection shows much
better coverage than expected.

### 6.3 Boundary Post-Processing

- Merge events shorter than 5-10 seconds.
- Split events longer than 30-60 seconds into micro events.
- Group related micro events into macro events lasting several minutes.
- Add 3-5 seconds of retrieval context around each event.
- Keep reported event boundaries separate from expanded retrieval context.
- Assign a boundary-confidence score.

### 6.4 Metadata Aggregation

After final video boundaries are produced:

1. Select heart-rate samples overlapping each event.
2. Select valid gaze fixations overlapping each event.
3. Select thermal images inside the interval or tolerance, only as optional evidence.
4. Attach overlapping transcript spans.
5. Calculate event-level features and coverage.

Metadata should follow the video-event boundaries. It should not create
independent incompatible chunks.

## 7. Data Products

Create these versioned outputs:

```text
processed/
  timeline/
    source_alignment.csv
  heart_rate/
    cleaned_samples.parquet
    window_features.parquet
  gaze/
    cleaned_fixations.parquet
    attended_objects.parquet
  thermal/
    optional_image_manifest.parquet
    optional_event_links.parquet
  transcript/
    aligned_transcripts.parquet
  chunks/
    fixed_30s.jsonl
    fixed_120s.jsonl
    semantic_micro_events.jsonl
    semantic_macro_events.jsonl
  events/
    multimodal_event_manifest.jsonl
```

Every generated dataset should include a processing-version field.

## 8. Validation Checks

### Timeline Validation

- Timestamps are monotonic within each source/session.
- Samples fall inside the recording period.
- Clock offsets are documented.
- Random events are manually checked across all modalities.

### Chunk Validation

- No negative or zero-length intervals.
- No unintended gaps in the semantic timeline.
- No overlapping core semantic events at the same hierarchy level.
- Every micro event has a valid macro parent.
- Minimum and maximum duration rules are satisfied.

### Metadata Validation

- Coverage rates are reported by participant and modality.
- Missing metadata remains explicitly missing.
- Validity flags are respected.
- Thermal-image tolerances and sparse coverage are recorded when thermal is used.
- Gaze points remain inside valid frame coordinates.

## 9. Experiments Owned by This Work Package

### Chunking Experiment

Compare:

1. Fixed 120 seconds
2. Fixed 30 seconds overlapping
3. Visual semantic chunks
4. Visual + transcript semantic chunks
5. Semantic chunks + fixed-window fallback

Measure:

- Candidate retrieval recall using the same downstream embedding/retriever
- Event duration distribution
- Boundary error or overlap when manual annotations exist
- Index size and processing cost

### Metadata Experiment

Compare downstream retrieval with:

1. No metadata
2. Heart rate only
3. Gaze only
4. Heart rate + gaze
5. Heart rate + gaze + optional sparse thermal

Report results only for topics/events with valid modality coverage, as well as
overall results.

Thermal should be reported as an exploratory ablation only. If coverage is too
low, the honest result is to state that thermal was inspected but not included
in the main retrieval pipeline.

## 10. Recommended Order

1. Audit files and timestamp formats.
2. Build the canonical timeline.
3. Clean heart-rate and gaze CSV files.
4. Inspect thermal coverage and decide whether it is worth keeping as optional evidence.
5. Implement fixed-window chunk baselines.
6. Implement visual semantic chunking.
7. Add transcript transitions to chunking.
8. Aggregate metadata into finalized events.
9. Validate synchronized examples manually.
10. Export the multimodal event manifest.
11. Run chunking and metadata ablations.

## 11. Immediate Information Needed

To begin implementation, collect a small representative sample containing:

- One complete primary video
- Its timestamp or filename metadata
- Matching transcript file
- Matching heart-rate CSV
- Matching gaze CSV with the complete header
- Any matching thermal images and their timestamp source, if available

One short synchronized session is enough to build and test the first end-to-end
preprocessing prototype.
