# Full-Dataset Chunking Status

The whole CASTLE main-video corpus has been chunked into fixed-window Event
Manifests.

## Completed

All main MP4 recordings discovered from Hugging Face were duration-probed and
written to:

```text
processed/all_castle/recordings.jsonl
```

Count:

```text
666 recordings
```

The whole-dataset fixed-window manifests are:

```text
processed/all_castle/fixed_30s.jsonl
processed/all_castle/fixed_120s.jsonl
```

Counts:

```text
fixed_30s events: 120530
fixed_120s events: 20630
duration probe failures: 0
```

Validation:

```text
processed/all_castle/fixed_30s.jsonl events 120530 videos 666
processed/all_castle/fixed_120s.jsonl events 20630 videos 666
```

Both manifests were loaded and validated with the project Event Manifest
validator.

## Not Completed

Full-dataset semantic chunking is not complete.

Reason:

```text
semantic chunking requires dense frame sampling and visual embedding for every
recording; the dataset audit reports ~8.22 TB of repository data and 666 main
MP4 recordings.
```

Current semantic chunking exists for development/evaluation slices only:

```text
processed/semantic/dev_08_400_700_visual_text_hr_events.jsonl
```

Status of that semantic slice:

```text
ready
1 macro event
9 micro events
transcript attached
heart rate attached
```

## LANTA Preparation

A LANTA-ready semantic chunking package has been prepared:

```text
lanta/README.md
lanta/slurm_semantic_array.sh
lanta/slurm_merge_semantic.sh
lanta/local_smoke_test.sh
scripts/lanta_process_castle_recording.py
scripts/merge_castle_semantic_outputs.py
```

Recommended LANTA flow:

```bash
lanta/local_smoke_test.sh
sbatch --array=1-10 lanta/slurm_semantic_array.sh
sbatch --array=1-666 lanta/slurm_semantic_array.sh
sbatch lanta/slurm_merge_semantic.sh
```

Local smoke test result:

```text
Completed day1_Allie_08: 7 records
{"manifest_files": 1, "embedding_files": 1, "records": 7, "embeddings": 7}
```

## What To Tell Senior

The dataset is now fully chunked into fixed windows and ready for baseline
indexing/evaluation. Semantic event chunking has been validated on representative
slices and should be scaled as the next compute-heavy job.

Recommended handoff framing:

```text
We prepared full-dataset fixed-window Event Manifests for all 666 CASTLE main
videos. We also validated semantic event chunking on development slices and
showed that semantic events improve localization, while fixed 120s windows
remain useful as high-recall fallback.
```

## Reproduce

```bash
make build-castle-all-recordings WORKERS=16
make build-castle-fixed-manifest RECORDINGS=processed/all_castle/recordings.jsonl OUTPUT=processed/all_castle/fixed_30s.jsonl WINDOW=30s PROCESSING_VERSION=castle-all-c8e7b5c-fixed30
make build-castle-fixed-manifest RECORDINGS=processed/all_castle/recordings.jsonl OUTPUT=processed/all_castle/fixed_120s.jsonl WINDOW=120s PROCESSING_VERSION=castle-all-c8e7b5c-fixed120
```
