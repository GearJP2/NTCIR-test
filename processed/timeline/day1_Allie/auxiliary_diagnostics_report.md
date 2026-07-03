# CASTLE Auxiliary Diagnostics Report

Participant/day: `Allie/day1`

## Decision

- Attachable modalities: heart_rate
- Blocked modalities: gaze, thermal
- Manifest readiness violations: 0

## Readiness gate

| Modality | Attach | Status | Blocker |
|---|---:|---|---|
| heart_rate | True | attachable_with_clock_day_join | - |
| gaze | False | blocked_no_clock_overlap | no candidate gaze clock interpretation overlaps recordings |
| thermal | False | blocked_unassigned | thermal files lack participant/day/timestamp assignment |

## Gaze diagnostics

- Streams: 1
- Rows: 88478
- Valid fixation ratio: 0.655
- Alignment candidates: 6
- Overlapping candidates: 0

## Thermal diagnostics

- Thermal files: 39
- Unassigned files: 39
- Files without path timestamps: 39
