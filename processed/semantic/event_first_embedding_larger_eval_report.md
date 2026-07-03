# Larger Event-First Retrieval Evaluation

Question: does segmenting CASTLE video into semantic events before embedding
hold up under a larger, fairer visual retrieval comparison?

Verdict: partially. Event-first retrieval is better for localization and better
than fixed 30-second windows on the current four-case set. A fused
semantic-event + fixed-120s fallback improves Recall@10 and best localization,
but fixed 120-second windows still win coarse Recall@1/3 because they use very
broad intervals.

## Evaluation Setup

Cases:

- `dev08_400_700`
- `dev08_700_1000`
- `dev09_400_700`
- `dev10_400_700`

Total visual queries: 10

Candidate types:

- `semantic_visual_text_w0.25`
- `semantic_visual_text_w0.25_transcript_rerank`
- `fixed_30s`
- `fixed_30s_transcript_rerank`
- `fixed_120s`
- `fixed_120s_transcript_rerank`
- `fused_semantic_fixed120_rrf`
- `fused_semantic_fixed120_rrf_transcript_hr_gated`

All candidate types used the same sampled frame embeddings, the same CLIP model,
and the same query embeddings. A hit requires at least `0.5` expected-interval
coverage. Fusion uses Reciprocal Rank Fusion over semantic-event and fixed-120s
rankings with semantic weight `1.0`, fixed-120s weight `0.85`, and RRF `k=60`.
Transcript reranking uses prepared transcript spans as secondary evidence. The
heart-rate gate is implemented but inactive for this query set because none of
the 10 visual queries asks for exertion, stress, pulse, walking, resting, or
similar physiological state.

## Aggregate Result

Source: `processed/semantic/castle_event_retrieval_summary.csv`

| Candidate type | Queries | Recall@1 | Recall@3 | Recall@10 | MRR | Mean best tIoU | Mean top1 tIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| semantic_visual_text_w0.25 | 10 | 0.400 | 0.500 | 0.900 | 0.517 | 0.616 | 0.299 |
| semantic_visual_text_w0.25_transcript_rerank | 10 | 0.400 | 0.600 | 0.900 | 0.533 | 0.616 | 0.290 |
| fixed_30s | 10 | 0.300 | 0.400 | 0.700 | 0.406 | 0.604 | 0.255 |
| fixed_30s_transcript_rerank | 10 | 0.300 | 0.500 | 0.700 | 0.418 | 0.604 | 0.265 |
| fixed_120s | 10 | 0.600 | 1.000 | 1.000 | 0.767 | 0.439 | 0.255 |
| fixed_120s_transcript_rerank | 10 | 0.500 | 1.000 | 1.000 | 0.717 | 0.439 | 0.164 |
| fused_semantic_fixed120_rrf | 10 | 0.400 | 0.500 | 1.000 | 0.528 | 0.641 | 0.299 |
| fused_semantic_fixed120_rrf_transcript_hr_gated | 10 | 0.400 | 0.500 | 1.000 | 0.528 | 0.641 | 0.299 |

## Interpretation

Semantic events beat fixed 30-second windows on ranking and localization:

- higher Recall@1, Recall@3, and Recall@10;
- higher MRR;
- higher mean best tIoU;
- higher mean top1 tIoU.

Fixed 120-second windows beat semantic events on coarse recall:

- Recall@1: `0.600` vs `0.400`;
- Recall@3: `1.000` vs `0.500`;
- Recall@10: `1.000` vs `0.900`.

Fusing semantic events with fixed 120-second fallback changes the tradeoff:

- Recall@10 improves from semantic-only `0.900` to `1.000`;
- mean best tIoU improves from semantic-only `0.616` to `0.641`;
- mean top1 tIoU stays at `0.299`, preserving semantic top-rank precision;
- Recall@1 and Recall@3 do not improve over semantic-only in this configuration.

Transcript reranking helps narrow candidate sets but is not yet useful on the
fused set:

- semantic-event Recall@3 improves from `0.500` to `0.600`;
- fixed-30s Recall@3 improves from `0.400` to `0.500`;
- fixed-120s Recall@1 regresses from `0.600` to `0.500`;
- fused metrics are unchanged after guarding transcript evidence behind the
  original fused rank.

An earlier stronger transcript setting improved fused `Recall@1/@3`, but it
dropped fused `Recall@10` on the continuous-control case. The current guarded
setting keeps coverage intact, which is the safer default.

That result is expected because each 300-second case has only three
120-second candidates. A correct broad window is easy to include in the top
three, but the timestamp is imprecise. This is visible in mean best tIoU:

- semantic: `0.616`;
- fixed 120s: `0.439`.

So fixed 120-second windows are a strong recall fallback, not a better final
timestamping strategy.

## Case-Level Pattern

Source: `processed/semantic/castle_event_retrieval_cases.csv`

- `dev08_400_700`: semantic wins clearly with Recall@1/3/10 all `1.000` and
  mean top1 tIoU `0.754`. Fusion preserves this.
- `dev08_700_1000`: semantic reaches Recall@10 `1.000`, but fixed 120s ranks
  broad candidates better at @1/@3. Fusion preserves Recall@10 but does not
  improve @1/@3.
- `dev09_400_700`: semantic fails the continuous-control case at @1/@3 and only
  reaches Recall@10 `0.500`; fixed 120s wins coarse recall. Fusion lifts
  Recall@10 to `1.000` but still misses @1/@3.
- `dev10_400_700`: semantic matches fixed 30s or better and reaches Recall@10
  `1.000`, but fixed 120s again wins coarse recall. Fusion improves best tIoU
  over all individual candidate types.

## Decision

Keep event-first embedding as the primary path for final timestamp precision,
and keep fixed 120-second fallback windows as a high-recall auxiliary channel.

Current best system shape:

1. Search semantic events for precise candidate intervals.
2. Search fixed 120-second windows as a high-recall safety channel.
3. Fuse both candidate sets with weighted RRF.
4. Rerank/refine timestamps inside any fixed-window hit.

This means the approach is not "semantic events only." The evidence supports
"semantic events plus fixed-window fallback."

Current fusion is useful for Recall@10 and candidate coverage, but not enough
for early-rank quality. The next technical step is a second-stage reranker that
prefers tighter semantic intervals when semantic and fixed candidates cover the
same expected activity, while keeping the fixed window available as context.
Transcript should be used as a secondary signal for semantic events, not as a
global override on fused candidates. Heart rate still needs physiology-specific
queries before it can be evaluated fairly.

## Commands

```bash
make compare-visual-retrieval SEMANTIC_MANIFEST=processed/semantic/dev_08_400_700_visual_text_events.jsonl RESULTS=processed/semantic/dev_08_400_700_visual_text_retrieval_results.csv SUMMARY=processed/semantic/dev_08_400_700_visual_text_retrieval_summary.csv
make sweep-transcript-boundary-weights WEIGHTS="0 0.1 0.25 0.5"
make compare-castle-event-retrieval-cases
pytest tests/unit/test_visual_retrieval_comparison.py tests/unit/test_sweep_transcript_boundary_weights.py -q
```

Verification:

- `compare-visual-retrieval` completed.
- `sweep-transcript-boundary-weights` completed with 16 case-weight rows.
- `compare-castle-event-retrieval-cases` completed with 32 case rows and 8
  summary rows.
- Unit tests: `8 passed`.
