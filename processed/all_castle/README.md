# Full CASTLE Fixed-Window Chunk Manifests

This directory contains whole-dataset fixed-window Event Manifests for all
CASTLE main MP4 recordings discovered at dataset revision
`c8e7b5cd9e9c83d0ff42560fc1169bed7867abd4`.

## Outputs

```text
recordings.jsonl
fixed_30s.jsonl
fixed_120s.jsonl
recording_failures.csv
```

## Counts

```text
recordings: 666
fixed_30s events: 120530
fixed_120s events: 20630
duration probe failures: 0
```

Both fixed-window manifests were loaded and validated with
`services.events.manifest.validate_event_manifest`.

## Notes

- These are fixed-window chunks, not semantic chunks.
- Fixed 30-second windows use the repository's existing 30s/10s-overlap
  fallback definition.
- Fixed 120-second windows are non-overlapping coarse recall chunks.
- Some final tail windows can be shorter than the nominal window size because
  recording durations are not always exact multiples of the window stride.
- Semantic chunking for the entire dataset requires dense frame sampling and
  CLIP embedding over all recordings. That is a heavier follow-up job; the
  current semantic proof-of-concept remains under `processed/semantic/`.
