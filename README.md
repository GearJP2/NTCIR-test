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
