# CASTLE implementation comparison

Compared on 2026-11-15 against local `../NTCIR-19-CASTLE-TASK`
(HEAD `90ebae4`) and `NTCIR_19_Proceedings_LLFL26.pdf`, especially §§3–4.
The sibling repository and PDF describe different pipelines. Neither should be
copied wholesale over this workspace's event preparation and alignment work.

## Findings

| Stage | This workspace | Sibling repository | PDF target |
| --- | --- | --- | --- |
| Boundaries | Adjacent (`v1`) or contextual (`v2`) OpenCLIP distances; smoothing, minimum gaps, duration constraints; optional transcript fusion | Imports precomputed visual JSONL | Frames every 5s; adjacent cosine distance exceeding recording's 85th percentile; transcripts do not set boundaries (§3.2) |
| Fixed windows | Canonical 120s non-overlapping, truncated tail already implemented; 30s baseline also available | Imports event kind from JSONL | 120s windows for every recording, indexed alongside semantic chunks (§3.3) |
| Retrieval representation | Pooled OpenCLIP event vectors in preparation tools; live API searches keyframes, CLAP audio, and MPNet transcripts | Precomputed VLM2Vec-V2 visual events; BGE-M3 transcripts | VLM2Vec/VLM2Vec-V2.0 video and query vectors in one shared collection (§§3.4–3.5) |
| Transcript role | Can affect experimental boundaries and multimodal retrieval; timestamp overlap attachment exists | Transcript hits promote visual events; overlapping speech supports reranking | Overlapping speech supports video reranking only (§3.6) |
| Reranking | Text-only MS MARCO cross-encoder | Exact preview clips, local VLM or hosted OpenRouter scorer | Native Gemini API, Gemini-3.5-flash-lite, scores 0–1 (§3.7) |
| Final selection | Existing TREC exporter has no paper-style selection | Reranked head plus unjudged visual backfill | Score >=0.5, stable descending score order, suppress any overlap per day/camera; adjacent intervals retained; up to 100 per topic (§§3.7/4.1) |
| Export | Existing `scripts/export_results.py` writes six-column TREC rows | Eight-column CSAT rows with recording clock times | Ranked day/source recording-time intervals |

The PDF's strict percentile rule is not exactly reproduced by choosing `v1`:
`select_boundary_points` also applies local maxima, `>=` threshold, minimum gap,
and minimum score; interval construction additionally splits/merges for duration.
Existing defaults are retained to preserve the documented development baseline.

## Implemented update

`evaluation/csat_submission.py` implements the PDF's final selection independently
of model inference. `scripts/export_csat_results.py` exposes an offline export
command. It accepts the sibling repository's Result fields (extra metadata is
ignored) and requires an explicit `reranked` boolean. Unjudged backfill is excluded;
raw cosine scores or cross-encoder logits must not be presented as video-reranker
scores. Semantic and fixed windows must be supplied together in the same topic list.

Input JSON maps topic IDs to result lists:

```json
{
  "CSAT26001": [
    {
      "source_relative_path": "day1/Allie/video/08.mp4",
      "start_ms": 120000,
      "end_ms": 140000,
      "score": 0.94,
      "reranked": true
    }
  ]
}
```

```bash
python -m scripts.export_csat_results results/video_reranked.json \
  --output-path results/LLFL26-LLFL26CSAT02-Automatic.txt \
  --group-id LLFL26 --run-id LLFL26CSAT02
```

Output (no header):

```text
LLFL26, LLFL26CSAT02, CSAT26001, 1, 08:02:00, 08:02:20, Allie, 0.940000
```

The exporter validates all rows and renders before creating the output, refuses
to overwrite a file, rejects invalid/non-finite scores and intervals, and caps
each topic at 100 retained results. Ties preserve input order. Overlap comparisons
use recording time, including events extending across adjacent hourly files.
Times are floored to whole seconds, matching the sibling exporter; intervals that
collapse at that precision are rejected. `HH.mp4` is assumed to start at `HH:00:00`,
as in the sibling implementation; verify this against audited recording metadata
before formal submission. Midnight-crossing intervals are rejected.

The existing TREC export remains available for existing callers and evaluation.
The new exporter is an explicit offline path, not connected to the live search API.

## Remaining work for full paper reproduction

1. Produce video-level VLM2Vec-V2 embeddings for both canonical semantic and 120s
   windows; ingest them into a new, dimension-compatible collection. Pooled
   OpenCLIP vectors cannot be relabelled as VLM2Vec or mixed into its vector space.
2. Add a video-query encoder for that same checkpoint and joint top-100 retrieval,
   without independent modality fusion or transcript promotion.
3. Add exact-range video extraction and native Gemini video scoring with aligned
   transcript context, saving per-candidate scores and run configuration before
   this exporter. Existing cross-encoder output is not a substitute.
4. Add a separately versioned paper boundary configuration if exact segmentation
   reproduction is required; preserve current experimental manifests.
5. Validate full-corpus retrieval and submission output against official topics,
   judgements, and audited clock anchors. Offline export tests do not establish
   the paper's reported retrieval performance.

The adjacent `submitted_runs_and_official_results.txt` describes an older two-run
manuscript variant, whereas this PDF reports six submissions and depths of 30,
35, 40, 50, and 100. Do not use the older text to set reproduction defaults.
