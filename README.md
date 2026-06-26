# CASTLE Hierarchical Multimodal Event Retrieval

Research implementation for semantic event retrieval over long CASTLE2024
lifelog recordings.

The active system direction is:

```text
CASTLE recording + timestamped modalities
        |
        v
Canonical Timeline
        |
        v
Semantic Micro Events + Macro Events + fixed Fallback Windows
        |
        v
Versioned Event Manifest
        |
        +--> direct video and keyframe evidence
        +--> caption, transcript, and OCR evidence
        +--> gaze-grounded attended objects
        +--> heart-rate and optional thermal evidence
        |
        v
Hierarchical retrieval, fusion, and temporal refinement
        |
        v
Ranked CASTLE recording IDs and event timestamps
```

## Current implementation stage

The repository is being reorganized around the canonical CASTLE `EventRecord`.
The immediate work is dataset audit, timestamp alignment, fixed-window
baselines, and semantic chunk construction. Existing ActivityNet tooling is
legacy and is not an active research path.

## Research invariants

- CASTLE is the only active corpus.
- Every modality maps onto one documented millisecond timeline.
- Video determines baseline semantic boundaries; auxiliary evidence follows.
- Core event timestamps remain separate from expanded retrieval context.
- Missing evidence remains explicitly missing.
- Fixed 30-second overlapping and 120-second non-overlapping windows remain
  reproducible baselines.
- Thermal evidence remains optional unless the dataset audit demonstrates
  sufficient coverage.

## Repository map

```text
app/schemas/event.py         Canonical Event Record interface
services/events/             Event Manifest validation and future construction
services/ingestion/          Existing media extraction modules to be adapted
services/retrieval/          Existing retrieval modules to be adapted
storage/                     Milvus and MinIO adapters
scripts/                     Dataset preparation and command-line workflows
frontend/                    Search and inspection interface
evaluation/                  CASTLE protocol and result tooling
docs/adr/                    Architectural decisions
docs/legacy/activitynet/     Retained historical ActivityNet work
```

## Work plans

- [Research plan](research_plan.md)
- [Metadata and semantic chunking work plan](metadata_and_chunking_workplan.md)
- [Architecture overview](Architecture.md)
- [Domain language](CONTEXT.md)
- [CASTLE-only architecture decision](docs/adr/0009-focus-research-and-active-architecture-on-castle.md)
- [Verified CASTLE dataset audit](docs/CASTLE_DATASET_AUDIT.md)

## Next executable milestone

Phase 0 must inventory a representative CASTLE sample:

- primary video and recording timestamps;
- transcript spans;
- heart-rate samples;
- gaze CSV fields and session start;
- thermal files and their timestamp source.

The first generated artifact will be a validated dataset audit and source
alignment table. The first retrieval artifacts will be fixed 30-second and
120-second Event Manifests.

Run the current Phase 0 workflow:

```bash
make audit-castle
make build-castle-slice DAY=day1 PARTICIPANT=Allie
```

The semantic-event builder accepts cleaned transcript spans as an optional
boundary signal while retaining video as the primary signal:

```bash
make build-visual-semantic-events \
  FRAME_DIR=processed/frames/dev_08_activity/day1_Allie_08 \
  VIDEO_ID=day1_Allie_08 \
  PARTICIPANT=Allie \
  VIDEO_URI="..." \
  OUTPUT_MANIFEST=processed/semantic/dev_08_visual_text_events.jsonl \
  OUTPUT_EMBEDDINGS=processed/semantic/dev_08_visual_text_embeddings.npz \
  OUTPUT_SCORES=processed/semantic/dev_08_visual_text_scores.csv \
  PROCESSING_VERSION=dev \
  TRANSCRIPT_SPANS=processed/slices/day1_Allie/dev_08_10_cleaned_transcripts.jsonl \
  TRANSCRIPT_WEIGHT=0.25
```

Run the four-case development sweep:

```bash
make sweep-transcript-boundary-weights WEIGHTS="0 0.1 0.25 0.5"
```

The current provisional transcript weight is `0.25`. It improves aggregate
manual boundary F1, but the segmenter still over-segments a continuous-activity
control interval, so this is not a frozen final configuration.

Inspect the development metadata timeline sources:

```bash
make build-castle-timeline-inventory STEMS="08 09 10"
```

This writes `processed/timeline/day1_Allie/source_timeline_inventory.csv` and
captures current clock, heart-rate, and gaze anchor status.

Attach provisional heart-rate summaries to an Event Manifest:

```bash
make enrich-castle-heart-rate \
  INPUT=processed/semantic/dev_08_400_700_visual_text_events.jsonl \
  OUTPUT=processed/semantic/dev_08_400_700_visual_text_hr_events.jsonl \
  SUMMARY=processed/semantic/dev_08_400_700_visual_text_hr_summary.csv
```

This uses recording metadata clock offsets from the timeline inventory to map
recording-relative Event Records onto the heart-rate CSV's elapsed day clock.
The summary CSV records mapped clock intervals, overlapping sample counts,
valid sample counts, and BPM statistics. It does not resolve absolute calendar
date, timezone, or gaze-session alignment.

Build gaze alignment diagnostics before attaching gaze to Event Records:

```bash
make build-castle-gaze-alignment-diagnostics
```

This writes stream-level gaze quality stats and candidate clock-overlap checks
under `processed/timeline/day1_Allie/`. Gaze remains diagnostic-only until a
participant/day recording anchor is demonstrated rather than assumed.

Inventory thermal provenance without downloading image payloads:

```bash
make build-castle-thermal-inventory
```

This writes `processed/timeline/thermal_inventory.csv` and records whether
thermal BMP paths expose participant, day, or timestamp evidence.

Summarize auxiliary modality readiness for Event Records:

```bash
make build-castle-modality-readiness
```

For the current development slice, heart-rate is attachable with documented
clock caveats; gaze and thermal remain blocked until their anchors are proven.

Run the full auxiliary diagnostic sequence:

```bash
make build-castle-auxiliary-diagnostics
```

This rebuilds the timeline inventory, gaze diagnostics, thermal inventory, and
final modality readiness report for the default `day1/Allie` development slice.

Check an Event Manifest against the readiness gate:

```bash
make check-castle-manifest-modality-readiness \
  MANIFEST=processed/semantic/dev_08_400_700_visual_text_hr_events.jsonl
```

This fails if Event Records attach a modality that the readiness report still
marks blocked.

Build a compact reviewer-facing auxiliary report:

```bash
make build-castle-auxiliary-report
```

This emits Markdown and JSON summaries under `processed/timeline/day1_Allie/`.
