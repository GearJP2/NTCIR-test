# CASTLE Dataset Audit

Audit date: June 25, 2026
Dataset revision: `c8e7b5cd9e9c83d0ff42560fc1169bed7867abd4`

## Repository inventory

The `CASTLE-Dataset/CASTLE2024` Hugging Face repository is public and contains
49,189 repository entries. After checksum sidecars are excluded, the audit
found:

| Source | Logical files | Approximate size |
|---|---:|---:|
| Main video | 780 | 7,616.25 GiB |
| Main metadata | 12,738 | 46.70 GiB |
| Main transcripts | 659 | 0.04 GiB |
| Auxiliary photos | 10,006 | 2.73 GiB |
| Auxiliary gaze | 7 | 0.18 GiB |
| Auxiliary heart rate | 39 | 0.01 GiB |
| Auxiliary thermal | 39 | 0.01 GiB |
| Auxiliary video | 16 | 0.68 GiB |

The repository spans four days and 16 recording-source directory names. Five
names—`Kitchen`, `Living1`, `Living2`, `Meeting`, and `Reading`—represent static
views rather than participants. Participant identity and camera assignment
must be confirmed from the dataset documentation before cross-view fusion.

## Timestamp formats observed

| Source | Observed format | Required conversion |
|---|---|---|
| Video | Hour-like recording stem, approximately one-hour MP4 | Establish recording start date, timezone, and clock time |
| Transcript | Recording-relative `[start_sec, end_sec]` | Add recording start timestamp |
| Main metadata | Clock of day, e.g. `08:05:17.180` | Combine CASTLE day date and timezone |
| Heart rate | Elapsed time, e.g. `00:00:02.0` | Add participant/day session anchor |
| Gaze | Session start in `TIME(...)` header plus elapsed `TIME` rows | Parse header start and add elapsed seconds |
| Thermal | Sequential BMP filename | Requires an external capture-time mapping |

No modality should be converted to the Canonical Timeline until its required
anchor is documented.

## Representative slice

`day1/Allie` was selected because path-level coverage includes:

- 11 available primary videos and two `.novideo` markers;
- 11 transcript files;
- heart rate;
- gaze;
- GPS and dense device metadata.

Video durations were obtained through HTTP range requests. No primary videos
were downloaded.

Results:

- 11 remotely probed recordings;
- approximately 11.00 hours total;
- 2,354 transcript chunks;
- 2,283 structurally valid intervals;
- 65 reversed intervals;
- 49 non-monotonic transitions;
- 33 empty transcript chunks;
- 10 of 11 transcript files require cleaning or quarantine.

Transcript text may still be useful, but malformed intervals must not enter an
Event Manifest unchanged.

## Generated baseline artifacts

The local ignored directory `processed/slices/day1_Allie/` contains:

- `recordings.jsonl`: 11 remote recording records;
- `source_inventory.csv`: paired video, transcript, and metadata paths;
- `transcript_quality.csv`: interval-quality measurements;
- `fixed_30s.jsonl`: 1,991 overlapping Fallback Windows;
- `fixed_120s.jsonl`: 341 non-overlapping Fallback Windows.

## Reproduction

```bash
make audit-castle
make build-castle-slice DAY=day1 PARTICIPANT=Allie

make build-castle-fixed-manifest \
  RECORDINGS=processed/slices/day1_Allie/recordings.jsonl \
  OUTPUT=processed/slices/day1_Allie/fixed_30s.jsonl \
  WINDOW=30s \
  PROCESSING_VERSION=castle-c8e7b5c-day1-allie-v1

make build-castle-fixed-manifest \
  RECORDINGS=processed/slices/day1_Allie/recordings.jsonl \
  OUTPUT=processed/slices/day1_Allie/fixed_120s.jsonl \
  WINDOW=120s \
  PROCESSING_VERSION=castle-c8e7b5c-day1-allie-v1
```

## Next decisions

1. Confirm the calendar date and timezone represented by each CASTLE day.
2. Confirm whether video stems such as `08` mean an exact `08:00` start.
3. Establish the heart-rate session anchor for each participant/day.
4. Map gaze sessions to participant/day recordings.
5. Find a capture-time source for thermal images or exclude thermal from the
   aligned baseline.
6. Define deterministic transcript cleaning: reject reversed intervals, remove
   empty chunks, sort valid spans, and retain an audit trail.

## Development transcript baseline

Recordings `day1/Allie/08`, `09`, and `10` form the first three-hour
development slice.

- 673 transcript spans accepted;
- 15 spans rejected with explicit reasons;
- 2 spans clipped to recording duration;
- 543 fixed 30-second events, 508 with transcript evidence;
- 93 fixed 120-second events, 92 with transcript evidence.

The transcript for recording `10` states that the machine-readable QR-code
clock is the experiment's timing reference. This is a useful alignment clue,
but the implementation must still verify the QR clock representation and
timezone from video or dataset documentation before assigning absolute
timestamps.

## Development visual sampling

The three selected UHD recordings total 40.8 GiB. A full download was tested,
but the available connection transferred at approximately 2–3 MB/s, implying a
four-to-five-hour acquisition. The transfer was stopped after approximately
609 MB of reported transfer progress. Cancellation did not leave a usable
local video, so future acquisition should run to completion for one recording
at a time.

For initial validation, HTTP range seeking sampled seven frames per recording
at ten-minute intervals:

- 21 frames across three recordings;
- less than 1 MiB of generated JPEG data;
- visual states include the CASTLE calibration card, driving activity, and
  indoor group activity;
- remote timestamp seeking works and is suitable for sparse inspection.

Sparse ten-minute sampling is not sufficient for semantic boundary detection.
The next visual experiment should use a dense but bounded interval—for example,
the first 10–15 minutes of one recording sampled every 2–5 seconds—before
processing all three hours.

## First visual-only semantic chunking baseline

The first five minutes of recording `08` contain only a static CASTLE
calibration card. This exposed an important detector failure: a percentile-only
threshold selected zero-valued score plateaus as boundaries. The detector now
requires both a percentile threshold and an absolute visual-change floor.

A second five-minute interval, from 400 to 700 seconds, was sampled every five
seconds and encoded directly from video frames with CLIP
`ViT-B-32-quickgelu`. No captions or transcripts influenced boundary
selection.

The corrected baseline produced:

- 60 direct visual samples;
- six learned visual boundaries;
- four maximum-duration fallback splits;
- ten Semantic Micro Events between 10 and 60 seconds;
- one Semantic Macro Event covering the five-minute experiment;
- one normalized pooled visual embedding per event.

Manual inspection confirmed that strong boundaries include real changes such as
a kitchen overview changing to cabinet inspection. The detector is also
sensitive to rapid egocentric camera motion within one activity. It is a valid
visual-only baseline, not yet the final multimodal semantic segmenter.

## Contextual visual detector and development evaluation

A contextual `V2` detector compares pooled visual embeddings before and after a
candidate boundary. This suppresses one-frame excursions while retaining
sustained visual changes. The original adjacent-frame detector remains frozen
as `V1`.

Seven approximate manual boundaries were annotated for the 400–700 second
development interval. Descriptions are evaluation-only and are never provided
to the detector. With a ±10-second matching tolerance:

| Detector | Precision | Recall | F1 | Mean error |
|---|---:|---:|---:|---:|
| V1 adjacent | 0.667 | 0.571 | 0.615 | 5.0 s |
| V2 contextual radius 1 | 0.571 | 0.571 | 0.571 | 6.25 s |
| V2 contextual radius 2 | 0.833 | 0.714 | **0.769** | 6.0 s |
| V2 contextual radius 3 | **1.000** | 0.571 | 0.727 | **2.5 s** |

Radius 2 is the current development setting because it has the highest F1. It
produces nine Semantic Micro Events, six learned boundaries, and two
maximum-duration splits over the five-minute interval.

## Direct-visual retrieval check

Three small visual queries were evaluated using CLIP text embeddings only at
query time. Indexed candidates contain pooled direct frame embeddings; no
caption or transcript embeddings are indexed in this comparison.

| Candidate unit | Recall@1 | Recall@3 | MRR | Mean best tIoU | Mean top-1 tIoU |
|---|---:|---:|---:|---:|---:|
| Semantic V2 | 0.667 | **1.000** | **0.833** | **0.594** | 0.449 |
| Fixed 30 s | 0.667 | 0.667 | 0.667 | 0.565 | **0.565** |
| Fixed 120 s | 0.667 | **1.000** | **0.833** | 0.453 | 0.300 |

This three-query check is diagnostic, not a research result. Semantic V2 finds
well-localized candidates within the top three, but fixed 30-second windows
currently have better top-1 timestamp precision. More queries and improved
event ranking are required before claiming a retrieval advantage.

## Transcript-aware semantic chunking diagnostic

The cleaned transcript spans can now contribute an optional contextual
transition score. Video remains primary: the combined boundary score is the
visual score plus a weighted transcript score where transcript context exists.
Missing or unchanged transcript evidence cannot suppress a visual boundary.
Generated Event Records also include the overlapping transcript text and
explicit transcript coverage.

On the same 400–700 second interval, V2 with transcript weight `0.25` produced
six learned boundaries and nine Semantic Micro Events. Against the seven
approximate manual boundaries at ±10 seconds, it matched the visual-only V2
radius-2 result:

| Segmenter | Precision | Recall | F1 | Mean error |
|---|---:|---:|---:|---:|
| V2 visual only | 0.833 | 0.714 | 0.769 | 6.0 s |
| V2 visual + transcript | 0.833 | 0.714 | 0.769 | 4.0 s |

The three-query direct-visual retrieval diagnostic improved from Recall@1
`0.667`, MRR `0.833`, and mean top-1 tIoU `0.449` to Recall@1 `1.000`, MRR
`1.000`, and mean top-1 tIoU `0.754`. This is development evidence only: the
query set is too small for a research claim, and transcript quality is uneven.

## Cross-interval transcript-weight sweep

Three additional five-minute intervals were sampled every five seconds:

- recording `08`, 700–1000 seconds: kitchen preparation with several clear
  sub-activity transitions;
- recording `09`, 400–700 seconds: continuous breakfast conversation used as
  a negative boundary control;
- recording `10`, 400–700 seconds: a workshop presentation with one clear
  presenter handoff.

Together with the original interval, the development sweep contains four cases,
14 positive manual boundaries, one negative control, and ten visual queries.
The references and queries are approximate manual diagnostics, not official
CASTLE labels.

| Transcript weight | Boundary precision | Boundary recall | Boundary F1 | Mean events | Recall@1 | MRR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.450 | 0.643 | 0.529 | 8.0 | 0.250 | 0.438 |
| 0.10 | 0.500 | 0.643 | 0.563 | 7.5 | 0.333 | **0.465** |
| 0.25 | **0.563** | 0.643 | **0.600** | 7.5 | 0.333 | 0.458 |
| 0.50 | 0.500 | 0.500 | 0.500 | 7.0 | 0.333 | **0.465** |

Weight `0.25` is the provisional development setting because boundary quality
is the primary objective of this component. Weight `0.50` is rejected because
it suppresses valid visual boundaries and lowers recall. The setting is not
final: the continuous recording `09` control still produces four learned
boundaries, showing that camera motion and viewpoint changes remain false
positive sources.

Run the reproducible sweep with:

```bash
make sweep-transcript-boundary-weights WEIGHTS="0 0.1 0.25 0.5"
```

## Development timeline source inventory

The first timeline inventory command inspects one clock-bearing main metadata
stream (`ACCL`) for recordings `08`, `09`, and `10`, plus Allie's day-1
heart-rate and gaze CSV files:

```bash
make build-castle-timeline-inventory STEMS="08 09 10"
```

The generated development output is
`processed/timeline/day1_Allie/source_timeline_inventory.csv`.

Key findings:

- `day1/Allie/08.ACCL` starts at `08:05:17.180` and ends at `08:59:59.998`.
- `day1/Allie/09.ACCL` starts at `09:00:00.003` and ends at `09:59:59.997`.
- `day1/Allie/10.ACCL` starts at `10:00:00.002` and ends at `10:59:59.996`.
- `auxiliary/heartrate/Allie/day1.csv` uses elapsed `HH:MM:SS.s` time,
  from `00:00:02.0` to `23:59:59.0`, and still needs a participant/day
  session anchor before event-level aggregation.
- `auxiliary/gaze/Allie.csv` has header start
  `TIME(2024/12/04 16:54:18.272)` and elapsed row times from `0.00000` to
  `1233.54395` seconds. This is an anchor candidate, but timezone and mapping
  to CASTLE `day1` recordings remain unresolved.

This confirms that video recording stems are not reliable absolute start
times: recording `08` begins about five minutes after 08:00 according to the
metadata clock stream. Until the calendar date and timezone are confirmed,
Event Records should continue to report recording-relative core intervals and
store clock metadata as an unresolved alignment source.

## Development heart-rate enrichment

Heart-rate summaries can now be attached to Event Records with the timeline
inventory:

```bash
make enrich-castle-heart-rate \
  INPUT=processed/semantic/dev_08_400_700_visual_text_events.jsonl \
  OUTPUT=processed/semantic/dev_08_400_700_visual_text_hr_events.jsonl \
  SUMMARY=processed/semantic/dev_08_400_700_visual_text_hr_summary.csv
```

The enrichment maps each event interval as:

1. recording-relative `start_ms`/`end_ms`;
2. plus the recording's first metadata clock offset from
   `source_timeline_inventory.csv`;
3. onto `auxiliary/heartrate/Allie/day1.csv` elapsed `HH:MM:SS.s` samples.

Attached summaries include mean/min/max/std BPM, linear BPM slope, baseline
delta against the participant/day median BPM, and valid-sample ratio. Samples
outside `30..220` BPM or below the configured confidence threshold are excluded
from statistics but still count toward valid-sample coverage.
The generated QA CSV records the mapped clock interval and sample counts for
each event so alignment and coverage can be inspected without parsing JSONL.

This is intentionally narrower than full multimodal alignment: it provides a
clock-of-day heart-rate join for development events, while calendar date,
timezone, and gaze-session mapping remain unresolved.

## Development gaze alignment diagnostics

Gaze is not yet attached to Event Records. The current diagnostic command is:

```bash
make build-castle-gaze-alignment-diagnostics
```

It writes:

- `processed/timeline/day1_Allie/gaze_stream_summary.csv`
- `processed/timeline/day1_Allie/gaze_alignment_candidates.csv`

For Allie's current gaze file, the stream summary reports one media stream
(`NewMedia0`) from `0.00000` to `1233.54395` seconds with roughly 65% valid
fixation rows and no AOI labels. Candidate clock checks compare:

1. the header clock-of-day anchor from `TIME(2024/12/04 16:54:18.272)`;
2. elapsed seconds treated as a day-clock diagnostic;
3. recording metadata clock windows from the timeline inventory.

Neither candidate clock interpretation overlaps the `08`, `09`, or `10`
recording clock windows. That is evidence against attaching gaze summaries to
Event Records in the development slice without an additional participant/day
session mapping source.

## Development thermal provenance inventory

Thermal evidence is also excluded from Event Records until it can be assigned
to participant/day recording intervals. The lightweight repository-metadata
inventory is:

```bash
make build-castle-thermal-inventory
```

It writes `processed/timeline/thermal_inventory.csv` without downloading image
payloads. The diagnostic records each thermal BMP path, file size, any sequence
number inferred from the filename, and whether the path exposes day,
participant, or timestamp evidence. Current CASTLE audit findings indicate the
thermal paths are flat sequential BMP filenames, so they remain unassigned and
should not be attached to Event Records without an external capture manifest or
image-level timestamp evidence.

## Auxiliary modality readiness decision

The current readiness report is built from the timeline inventory, gaze
alignment diagnostics, and thermal inventory:

```bash
make build-castle-modality-readiness
```

It writes `processed/timeline/day1_Allie/modality_readiness.csv`. For the
development slice:

- `heart_rate` is attachable to Event Records through the documented
  clock-of-day join and QA summary.
- `gaze` is blocked because no candidate clock interpretation overlaps the
  recording windows.
- `thermal` is blocked because image paths lack participant/day/timestamp
  assignment.

This report is the gate for future multimodal enrichment: no modality should
be attached to Event Records unless it has an attachable readiness row and a
corresponding coverage/QA artifact.

The full auxiliary diagnostic sequence is:

```bash
make build-castle-auxiliary-diagnostics
```

It rebuilds the source timeline inventory, gaze alignment diagnostics, thermal
provenance inventory, and modality readiness report in order.

Event Manifests can be checked against the readiness gate:

```bash
make check-castle-manifest-modality-readiness \
  MANIFEST=processed/semantic/dev_08_400_700_visual_text_hr_events.jsonl
```

The check writes `modality_readiness_violations.csv` and exits non-zero if any
EventRecord has `coverage.heart_rate`, `coverage.gaze`, or `coverage.thermal`
enabled while that participant/day/modality is blocked or missing from the
readiness report.
