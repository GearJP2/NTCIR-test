# LANTA Semantic Chunking Package

This package prepares the full CASTLE fixed-window chunks for semantic chunking
on LANTA.

## Input

Use the full recording manifest already prepared locally:

```text
processed/all_castle/recordings.jsonl
```

It contains:

```text
666 main MP4 recordings
```

The fixed-window baseline chunks are already done:

```text
processed/all_castle/fixed_30s.jsonl
processed/all_castle/fixed_120s.jsonl
```

## What LANTA Should Build

For each recording:

```text
remote video
  -> sample frames every 5 seconds
  -> embed sampled frames with CLIP on GPU
  -> detect semantic visual boundaries
  -> split into micro events
  -> pool frame embeddings per event
  -> write per-video semantic manifest + embeddings + boundary scores
```

Expected per-video outputs:

```text
processed/lanta_semantic/manifests/<video_id>.jsonl
processed/lanta_semantic/embeddings/<video_id>.npz
processed/lanta_semantic/scores/<video_id>.csv
processed/lanta_semantic/done/<video_id>.json
```

Merged outputs:

```text
processed/lanta_semantic/semantic_events.jsonl
processed/lanta_semantic/semantic_embeddings.npz
processed/lanta_semantic/merge_summary.json
```

## Recommended First Run

Local smoke test:

```bash
lanta/local_smoke_test.sh
```

Current smoke-test result:

```text
Completed day1_Allie_08: 7 records
{"manifest_files": 1, "embedding_files": 1, "records": 7, "embeddings": 7}
```

Run a small array first:

```bash
sbatch --array=1-10 lanta/slurm_semantic_array.sh
```

If that works, run all recordings:

```bash
sbatch --array=1-666 lanta/slurm_semantic_array.sh
```

Then merge:

```bash
sbatch lanta/slurm_merge_semantic.sh
```

## Important Notes

- This performs semantic chunking. It is heavier than fixed-window chunking.
- The default frame interval is `5s` to make the first full run practical.
- The dev experiments used `5s` for 300-second intervals and showed useful
  event-first behavior.
- If LANTA runtime is acceptable, a later higher-quality pass can use `2s`.
- Transcript boundary signals are optional. The current whole-dataset transcript
  cleaning is not complete, so the default LANTA job uses visual-only semantic
  boundaries.
- Fixed 120-second chunks should remain as the high-recall fallback.

## Resource Tuning

Defaults in `slurm_semantic_array.sh`:

```text
1 GPU
8 CPU cores
32 GB RAM
4 hour time limit per recording
frame interval: 5 seconds
batch size: 128
```

If memory is tight, lower:

```text
BATCH_SIZE=64
```

If jobs are too slow, keep `FRAME_INTERVAL_SEC=5` and increase batch size.

If quality is insufficient and runtime is acceptable, try:

```text
FRAME_INTERVAL_SEC=2
```
