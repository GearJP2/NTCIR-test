#!/usr/bin/env bash
set -euo pipefail

python scripts/lanta_process_castle_recording.py \
  processed/all_castle/recordings.jsonl \
  --task-id 1 \
  --output-root processed/lanta_smoke \
  --frame-interval-sec 600 \
  --processing-version castle-lanta-smoke \
  --device cpu \
  --batch-size 8 \
  --detector v2 \
  --max-event-sec 600 \
  --transcript-weight 0.0

python scripts/merge_castle_semantic_outputs.py processed/lanta_smoke \
  --output-manifest processed/lanta_smoke/semantic_events.jsonl \
  --output-embeddings processed/lanta_smoke/semantic_embeddings.npz
