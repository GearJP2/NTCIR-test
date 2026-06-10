# Moment Search Runbook

Use this runbook after media has been ingested into Milvus.

## 1. Check One Indexed Media ID

For ActivityNet benchmark runs, ingest with filename-based media IDs so indexed evidence
matches manifest IDs:

```bash
make ingest-activitynet
```

List candidate media IDs sampled from Moment Search collections:

```bash
make list-indexed-media
```

Then inspect one candidate:

```bash
make check-media-index MEDIA_ID=v_123
```

At least one of `visual_keyframes`, `text_transcripts`, or `audio_segments` must report `ready: true`. If all are false, the media has not been indexed for Moment Search.

## 2. Run One Query

```bash
make search-moments \
  MEDIA_ID=v_123 \
  DURATION_SEC=120 \
  QUERY="woman doing sit ups" \
  PROFILE=activitynet_visual_heavy
```

The response should contain ranked `VideoMoment` results with source-specific `Evidence`. Empty results usually mean either the media has no indexed evidence or all modality searches failed.

## 3. ActivityNet Evaluation

Download a playable 50-video ActivityNet validation subset:

```bash
python scripts/download_activitynet_dev.py \
  --split val1 \
  --target-videos 50 \
  --max-attempts 500 \
  --output-dir data/activitynet/videos \
  --manifest-path data/manifests/activitynet_dev50.jsonl
```

Or build the manifest after ActivityNet videos already exist locally:

```bash
make build-activitynet-manifest \
  VIDEO_ROOT=data/activitynet/videos \
  MANIFEST=data/manifests/activitynet_dev50.jsonl
```

Run temporal evaluation:

```bash
make eval-moments MANIFEST=data/manifests/activitynet_dev50.jsonl
```

This reports `Recall@10` and `mAP@10` using the `activitynet_visual_heavy` profile and `tIoU >= 0.3`.

## 4. CASTLE Manual Inspection

Run the curated query set against one indexed CASTLE recording:

```bash
make inspect-castle \
  MEDIA_ID=castle_recording_001 \
  DURATION_SEC=3600 \
  OUTPUT=data/inspection/castle_smoke_results.jsonl
```

Review the JSONL output manually in the Search Interface or by inspecting timestamps. CASTLE has no Ground Truth Moments, so these results are qualitative only.

## 5. Known Limits

- Generated-summary evidence is not searched separately yet.
- Full `pytest tests/unit` has known legacy failures outside the Moment Search slice.
- Moment Search depends on indexed Milvus collections and locally available model weights.
