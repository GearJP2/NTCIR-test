# CASTLE Event-First Chunking Handoff

This folder summarizes the current CASTLE chunking work and the evidence for
using event-first embedding.

## Status

The whole CASTLE main-video corpus has now been chunked into fixed windows.

Whole-dataset fixed-window manifests:

```text
processed/all_castle/fixed_30s.jsonl
processed/all_castle/fixed_120s.jsonl
```

Whole-dataset counts:

```text
recordings: 666
fixed_30s events: 120530
fixed_120s events: 20630
duration probe failures: 0
```

The current semantic real-video chunking artifact is also ready for senior
review.

Main chunk manifest:

```text
processed/semantic/dev_08_400_700_visual_text_hr_events.jsonl
```

Validated report:

```text
processed/semantic/final_semantic_chunking_report.md
```

Current status from validation:

```text
Status: ready
Video: day1_Allie_08
Interval: 400s-700s
Macro events: 1
Micro events: 9
Micro duration range: 10s-60s
Transcript-covered events: 8
Heart-rate-covered records: 10
Gaze-covered events: 0
Thermal-covered events: 0
Modality readiness violations: 0
```

## Main Finding

Segmenting video into semantic events before embedding is promising for CSAT
because it produces more precise event intervals, but it should not replace
fixed-window fallback yet.

Current best system shape:

1. Build semantic micro/macro events from visual boundaries plus light transcript
   boundary signal.
2. Embed semantic events for precise retrieval.
3. Also keep fixed 120-second fallback windows for high recall.
4. Fuse semantic and fixed-window candidates with weighted RRF.
5. Use transcript as secondary reranking evidence, not as a global override.

## Retrieval Evidence

Four-case evaluation, 10 visual queries:

```text
semantic events:       Recall@1 0.400, Recall@3 0.500, Recall@10 0.900, best tIoU 0.616, best temporal precision 0.810
semantic + transcript: Recall@1 0.400, Recall@3 0.600, Recall@10 0.900, best tIoU 0.616, best temporal precision 0.810
fixed 30s:             Recall@1 0.300, Recall@3 0.400, Recall@10 0.700, best tIoU 0.604, best temporal precision 0.967
fixed 120s:            Recall@1 0.600, Recall@3 1.000, Recall@10 1.000, best tIoU 0.439, best temporal precision 0.454
fused semantic+120s:   Recall@1 0.400, Recall@3 0.500, Recall@10 1.000, best tIoU 0.641, best temporal precision 0.727
semantic-refined RRF:  Recall@1 0.400, Recall@3 0.500, Recall@10 1.000, best tIoU 0.641, best temporal precision 0.727
```

Interpretation:

- Semantic events beat fixed 30s and give better localization.
- Fixed 120s wins coarse recall because each case has only broad windows.
- Fusion gives fixed-window coverage while improving localization and preserving
  substantially better temporal precision than fixed 120s alone.
- Semantic-refined fusion currently matches plain RRF, but it is the safer final
  answer policy because it promotes semantic intervals without deleting the
  fixed-window fallback.
- Transcript helps semantic events at `Recall@3`, but can damage coverage if it
  is too strong.

CSAT framing:

- Fixed 120-second windows are useful recall candidates but are too broad for
  final key-event answers.
- Semantic events should be promoted as the final interval when they overlap a
  high-scoring fixed-window hit, while the fixed window remains fallback context.

## Prepared Modalities

Ready enough to use:

- Video frames and event intervals.
- Transcript spans attached to events.
- Heart-rate summaries attached to events with a documented clock caveat.

Not ready:

- Gaze: blocked because no candidate gaze clock interpretation overlaps
  recordings.
- Thermal: blocked because files lack participant/day/timestamp assignment.

## Senior Review Ask

Please review whether this architecture is acceptable:

```text
semantic event retrieval + fixed 120s fallback + weighted RRF + guarded transcript rerank

Model details:
  - CLIP visual encoder: `ViT-B-32-quickgelu`
  - pretrained weights: `openai`
```

The main open design question is the second-stage reranker:

- Should final answers prefer semantic intervals whenever they sit inside a
  high-scoring fixed 120s window?
- How aggressively should transcript evidence rerank visual candidates?
- Should fixed 120s windows be returned as final intervals, or only used as
  context for refinement?

## Files In This Folder

- `technical_summary.md`: detailed implementation and results summary.
- `artifact_index.md`: exact files to inspect.
- `reproduce.md`: commands to regenerate chunking and evaluation artifacts.
- `full_dataset_chunking.md`: whole-dataset fixed-window chunking status.
