# Event-First Embedding Decision Report

Question: should CASTLE video be segmented into Event Records before embedding and
retrieval?

Verdict: provisionally yes for visually distinct CASTLE events, but not proven
as a full-system win yet.

## Evidence Supporting Event-First Embedding

On the `day1_Allie_08` `400s-700s` development slice, semantic event intervals
were embedded by pooling sampled frame embeddings inside each event, then
compared against fixed 30-second and fixed 120-second interval baselines.

Results from `processed/semantic/dev_08_400_700_visual_text_retrieval_summary.csv`:

| Candidate type | Queries | Recall@1 | Recall@3 | MRR | Mean best tIoU | Mean top1 tIoU |
|---|---:|---:|---:|---:|---:|---:|
| semantic_visual_text | 3 | 1.000 | 1.000 | 1.000 | 0.754 | 0.754 |
| fixed_30s | 3 | 0.667 | 0.667 | 0.667 | 0.565 | 0.565 |
| fixed_120s | 3 | 0.667 | 1.000 | 0.833 | 0.453 | 0.300 |

Interpretation:

- Semantic events ranked the correct interval first for all 3 visual queries.
- Fixed 30-second windows sometimes split a real activity across adjacent
  windows, lowering hit quality.
- Fixed 120-second windows often contain the event but dilute timestamp
  precision because they include unrelated activity.

This supports the core approach: segment first, then embed event-level evidence.

## Boundary Quality

The semantic chunking artifact is structurally ready:

- Manifest: `processed/semantic/dev_08_400_700_visual_text_hr_events.jsonl`
- Status: `ready`
- Macro events: 1
- Micro events: 9
- Micro duration range: `10000..60000 ms`
- Modality readiness violations: 0

Boundary evaluation on `dev08_400_700`:

| Detector | Precision | Recall | F1 | Mean absolute error |
|---|---:|---:|---:|---:|
| v1-adjacent | 0.667 | 0.571 | 0.615 | 5000 ms |
| v2-contextual-r1 | 0.571 | 0.571 | 0.571 | 6250 ms |
| v2-contextual-r2 | 0.833 | 0.714 | 0.769 | 6000 ms |
| v2-contextual-r3 | 1.000 | 0.571 | 0.727 | 2500 ms |

The current finalized manifest uses visual boundaries plus transcript boundary
weight `0.25`. The four-case sweep gives this aggregate:

| Transcript weight | Boundary F1 micro | Retrieval Recall@1 | Retrieval Recall@3 | Mean best tIoU | Mean top1 tIoU |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.529 | 0.250 | 0.583 | 0.523 | 0.165 |
| 0.10 | 0.563 | 0.333 | 0.542 | 0.588 | 0.253 |
| 0.25 | 0.600 | 0.333 | 0.458 | 0.598 | 0.253 |
| 0.50 | 0.500 | 0.333 | 0.542 | 0.660 | 0.265 |

Interpretation:

- Adding transcript boundary signal improves boundary F1 from `0.529` to
  `0.600` at weight `0.25`.
- Retrieval ranking does not monotonically improve with better boundary F1.
  Boundary quality helps, but embedding/query quality and event granularity also
  matter.

## Reasons It May Work

1. Event intervals reduce irrelevant visual evidence inside each embedding.
2. Event intervals preserve activity boundaries better than fixed windows.
3. Event records provide a clean place to attach transcript and heart-rate
   summaries after segmentation.
4. Long fixed windows can get high overlap but poor timestamp precision.
5. Short fixed windows can miss complete activities by splitting them.

## Reasons It May Not Work Yet

1. Current proof is small: 4 labeled intervals and 10 curated visual queries.
2. The strongest fixed-window comparison is only available for one dev slice
   with 3 visual queries.
3. `dev09_400_700` is a continuous-activity negative control. The segmenter
   still creates several events there, which means over-segmentation remains a
   risk.
4. Gaze and thermal are not attachable yet, so this is not a full multimodal
   proof.
5. Event-level pooled frame embeddings may lose brief object evidence inside a
   longer event unless keyframe retrieval is retained as a parallel channel.

## Decision

Continue with event-first embedding as the primary CASTLE path, but keep fixed
30-second and 120-second windows as fallback retrieval channels.

This approach is working on the current visually distinct development slice. It
is not yet proven enough to claim a general CASTLE-wide improvement.

## Next Required Experiment

Run the same semantic-vs-fixed retrieval comparison across all four existing
development cases:

- `dev08_400_700`
- `dev08_700_1000`
- `dev09_400_700`
- `dev10_400_700`

The acceptance rule should be:

- semantic events improve or match fixed windows on Recall@3;
- semantic events improve mean top1 tIoU over fixed 120-second windows;
- semantic events do not over-segment continuous controls enough to harm
  retrieval or manifest size;
- fixed windows remain available when semantic confidence is low.

Commands rerun for this report:

```bash
make finalize-castle-semantic-chunking
make compare-visual-retrieval SEMANTIC_MANIFEST=processed/semantic/dev_08_400_700_visual_text_events.jsonl RESULTS=processed/semantic/dev_08_400_700_visual_text_retrieval_results.csv SUMMARY=processed/semantic/dev_08_400_700_visual_text_retrieval_summary.csv
make evaluate-visual-boundaries COMPARISON=processed/semantic/dev_08_400_700_detector_comparison.json REFERENCE=evaluation/fixtures/castle_dev08_400_700_boundaries.jsonl OUTPUT=processed/semantic/dev_08_400_700_boundary_evaluation.csv
make sweep-transcript-boundary-weights WEIGHTS="0 0.1 0.25 0.5"
pytest tests/unit/test_build_visual_semantic_events.py tests/unit/test_visual_retrieval_comparison.py tests/unit/test_semantic_chunking_report.py tests/unit/test_event_manifest.py -q
```

Verification: all commands completed successfully; selected unit tests passed
with `15 passed`.
