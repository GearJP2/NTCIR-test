# Artifact Index

## Primary Chunking Artifacts

Whole-dataset fixed-window Event Manifests:

```text
processed/all_castle/recordings.jsonl
processed/all_castle/fixed_30s.jsonl
processed/all_castle/fixed_120s.jsonl
processed/all_castle/recording_failures.csv
```

Final ready Event Manifest:

```text
processed/semantic/dev_08_400_700_visual_text_hr_events.jsonl
```

Final chunking report:

```text
processed/semantic/final_semantic_chunking_report.md
processed/semantic/final_semantic_chunking_report.json
```

Semantic boundary scores and embeddings:

```text
processed/semantic/dev_08_400_700_visual_text_boundary_scores.csv
processed/semantic/dev_08_400_700_visual_text_embeddings.npz
processed/semantic/dev_08_400_700_visual_text_events.jsonl
```

Heart-rate-enriched semantic manifest:

```text
processed/semantic/dev_08_400_700_visual_text_hr_events.jsonl
processed/semantic/dev_08_400_700_visual_text_hr_summary.csv
```

## Evaluation Artifacts

Four-case retrieval summary:

```text
processed/semantic/castle_event_retrieval_summary.csv
```

Four-case per-case retrieval results:

```text
processed/semantic/castle_event_retrieval_cases.csv
```

Larger evaluation report:

```text
processed/semantic/event_first_embedding_larger_eval_report.md
```

Initial decision report:

```text
processed/semantic/event_first_embedding_decision_report.md
```

Transcript-weight sweep:

```text
processed/semantic/transcript_weight_sweep.csv
processed/semantic/transcript_weight_sweep_summary.csv
```

## Modality Readiness

Auxiliary diagnostics report:

```text
processed/timeline/day1_Allie/auxiliary_diagnostics_report.md
processed/timeline/day1_Allie/auxiliary_diagnostics_report.json
```

Modality readiness:

```text
processed/timeline/day1_Allie/modality_readiness.csv
processed/timeline/day1_Allie/modality_readiness_violations.csv
```

Gaze diagnostics:

```text
processed/timeline/day1_Allie/gaze_stream_summary.csv
processed/timeline/day1_Allie/gaze_alignment_candidates.csv
```

Thermal inventory:

```text
processed/timeline/thermal_inventory.csv
```

## Source Data Products

Representative CASTLE slice:

```text
processed/slices/day1_Allie/recordings.jsonl
processed/slices/day1_Allie/fixed_30s.jsonl
processed/slices/day1_Allie/fixed_120s.jsonl
processed/slices/day1_Allie/dev_08_10_cleaned_transcripts.jsonl
```

Sampled frames:

```text
processed/frames/dev_08_activity/day1_Allie_08/
processed/frames/dev_08_700_1000/day1_Allie_08/
processed/frames/dev_09_400_700/day1_Allie_09/
processed/frames/dev_10_400_700/day1_Allie_10/
```

## Code Entry Points

Semantic event builder:

```text
scripts/build_visual_semantic_events.py
```

Final chunking validator:

```text
scripts/finalize_castle_semantic_chunking.py
```

Four-case retrieval evaluator:

```text
scripts/compare_castle_event_retrieval_cases.py
```

Transcript-weight sweep:

```text
scripts/sweep_transcript_boundary_weights.py
```

Core visual retrieval helpers:

```text
evaluation/visual_retrieval.py
```

## LANTA Package

Semantic chunking package for LANTA:

```text
lanta/README.md
lanta/slurm_semantic_array.sh
lanta/slurm_merge_semantic.sh
lanta/local_smoke_test.sh
scripts/lanta_process_castle_recording.py
scripts/merge_castle_semantic_outputs.py
```
