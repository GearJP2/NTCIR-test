# Technical Summary

## Objective

Evaluate whether CASTLE long video should be segmented into semantic events
before embedding and retrieval.

The tested approach is:

```text
real CASTLE video
  -> sampled frames
  -> visual boundary detection
  -> transcript-weighted boundary adjustment
  -> semantic micro/macro Event Records
  -> event-level pooled visual embeddings
  -> retrieval comparison against fixed windows
```

## Current Chunking Artifact

The finalized manifest is:

```text
processed/semantic/dev_08_400_700_visual_text_hr_events.jsonl
```

It covers:

```text
participant: Allie
day: day1
video: 08
interval: 400s-700s
```

Validation output:

```text
Status: ready
Macro events: 1
Micro events: 9
Micro duration range: 10000..60000 ms
Mean micro duration: 33333.3 ms
Modality readiness violations: 0
```

## Boundary Configuration

The selected transcript boundary weight is:

```text
0.25
```

Four-case transcript-weight sweep:

```text
weight 0.00: boundary F1 micro 0.529, Recall@1 0.250, Recall@10 0.750
weight 0.10: boundary F1 micro 0.563, Recall@1 0.333, Recall@10 0.875
weight 0.25: boundary F1 micro 0.600, Recall@1 0.333, Recall@10 0.875
weight 0.50: boundary F1 micro 0.500, Recall@1 0.333, Recall@10 0.875
```

Interpretation:

- `0.25` gives the best tested boundary F1.
- Retrieval does not monotonically track boundary F1, so boundary quality is
  necessary but not sufficient.

## Retrieval Evaluation

The larger evaluation uses four development cases:

```text
dev08_400_700
dev08_700_1000
dev09_400_700
dev10_400_700
```

Total visual queries:

```text
10
```

Aggregate results:

| Candidate type | Recall@1 | Recall@3 | Recall@10 | MRR | Best tIoU | Top1 tIoU |
|---|---:|---:|---:|---:|---:|---:|
| semantic_visual_text_w0.25 | 0.400 | 0.500 | 0.900 | 0.517 | 0.616 | 0.299 |
| semantic_visual_text_w0.25_transcript_rerank | 0.400 | 0.600 | 0.900 | 0.533 | 0.616 | 0.290 |
| fixed_30s | 0.300 | 0.400 | 0.700 | 0.406 | 0.604 | 0.255 |
| fixed_30s_transcript_rerank | 0.300 | 0.500 | 0.700 | 0.418 | 0.604 | 0.265 |
| fixed_120s | 0.600 | 1.000 | 1.000 | 0.767 | 0.439 | 0.255 |
| fixed_120s_transcript_rerank | 0.500 | 1.000 | 1.000 | 0.717 | 0.439 | 0.164 |
| fused_semantic_fixed120_rrf | 0.400 | 0.500 | 1.000 | 0.528 | 0.641 | 0.299 |
| fused_semantic_fixed120_rrf_transcript_hr_gated | 0.400 | 0.500 | 1.000 | 0.528 | 0.641 | 0.299 |

## Interpretation

Event-first embedding is useful:

- It beats fixed 30-second windows on `Recall@1`, `Recall@3`, `Recall@10`, MRR,
  best tIoU, and top1 tIoU.
- It localizes better than fixed 120-second windows.

Fixed 120-second windows are still necessary:

- They give perfect coarse `Recall@10`.
- They are broad and imprecise, so they should be fallback/context rather than
  the preferred final timestamp.

Transcript is useful but risky:

- It improves semantic-event `Recall@3` from `0.500` to `0.600`.
- Stronger transcript weights can hurt coverage on continuous-control cases.
- Current guarded transcript reranking keeps fused coverage intact but does not
  improve fused early rank.

Heart rate is prepared but not evaluated fairly yet:

- It is attached with a clock caveat.
- Current visual queries do not ask about exertion, stress, walking, resting,
  pulse, or similar physiology signals.

## Current Recommendation

Proceed with:

```text
semantic event index
+ fixed 120-second fallback index
+ weighted RRF fusion
+ second-stage reranker/refiner
```

Do not proceed with:

```text
semantic events only
```

The next important engineering work is per-query diagnostics and a reranker that
prefers tight semantic intervals inside high-scoring fixed windows.
