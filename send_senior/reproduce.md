# Reproduce Commands

Run from repository root:

```bash
cd /home/rojsak/Coding/TonKitLab/NTCIR-test
```

## Validate Current Chunking Artifact

Build all-recording duration manifest:

```bash
make build-castle-all-recordings WORKERS=16
```

Build whole-dataset fixed-window chunks:

```bash
make build-castle-fixed-manifest RECORDINGS=processed/all_castle/recordings.jsonl OUTPUT=processed/all_castle/fixed_30s.jsonl WINDOW=30s PROCESSING_VERSION=castle-all-c8e7b5c-fixed30
make build-castle-fixed-manifest RECORDINGS=processed/all_castle/recordings.jsonl OUTPUT=processed/all_castle/fixed_120s.jsonl WINDOW=120s PROCESSING_VERSION=castle-all-c8e7b5c-fixed120
```

Validate current semantic chunking artifact:

```bash
make finalize-castle-semantic-chunking
```

Expected result:

```text
Semantic chunking status: ready
```

## Rebuild Auxiliary Diagnostics

```bash
make build-castle-auxiliary-diagnostics
```

Expected current readiness:

```text
heart_rate: attachable_with_clock_day_join
gaze: blocked_no_clock_overlap
thermal: blocked_unassigned
```

## Re-run Boundary Weight Sweep

```bash
make sweep-transcript-boundary-weights WEIGHTS="0 0.1 0.25 0.5"
```

Outputs:

```text
processed/semantic/transcript_weight_sweep.csv
processed/semantic/transcript_weight_sweep_summary.csv
```

## Re-run Retrieval Evaluation

```bash
make compare-castle-event-retrieval-cases
```

Outputs:

```text
processed/semantic/castle_event_retrieval_cases.csv
processed/semantic/castle_event_retrieval_summary.csv
```

This evaluates:

```text
semantic events
semantic events + transcript rerank
fixed 30s
fixed 30s + transcript rerank
fixed 120s
fixed 120s + transcript rerank
fused semantic + fixed 120s
fused semantic + fixed 120s + transcript/HR gated rerank
```

## Run Focused Tests

```bash
pytest tests/unit/test_visual_retrieval_comparison.py tests/unit/test_sweep_transcript_boundary_weights.py -q
```

Expected current result:

```text
8 passed
```

## Notes

The evaluation loads the CLIP/open_clip model from local cache when available.
If model cache is missing, the run may need network access or model-cache
preparation.
