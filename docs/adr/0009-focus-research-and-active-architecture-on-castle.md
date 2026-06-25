# Focus Research and Active Architecture on CASTLE

## Status

Accepted. Supersedes ADR-0003 and ADR-0008 for new work.

## Decision

CASTLE2024 is the only active research corpus and architecture target.

The system will construct a Canonical Timeline and versioned Event Manifest
containing hierarchical Semantic Events, fixed Fallback Windows, aligned
transcripts, heart-rate summaries, gaze summaries, and optional thermal
evidence.

New ingestion, retrieval, frontend, and evaluation interfaces must consume
CASTLE Event Records. ActivityNet-specific scripts, profiles, reports, and
metrics may remain in a legacy area for provenance, but they must not define
active defaults or research claims.

The implementation order is:

1. Audit CASTLE files, modalities, timestamps, and coverage.
2. Define and validate the Event Record interface.
3. Produce fixed 30-second and 120-second Event Manifests.
4. Produce Semantic Micro Events, Macro Events, and fixed-window fallback.
5. Align auxiliary evidence to finalized Core Event Intervals.
6. Build hierarchical retrieval and temporal refinement over those records.

## Consequences

- Existing ActivityNet defaults and terminology must be removed from active
  interfaces.
- ActivityNet evaluation remains historical work, not a second supported
  research path.
- No quantitative CASTLE claim is valid until its labels or official
  evaluation protocol are documented.
- The Event Manifest becomes the seam between data preparation and retrieval.
