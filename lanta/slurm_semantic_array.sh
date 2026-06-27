#!/usr/bin/env bash
#SBATCH --job-name=castle-semantic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lanta_semantic_%A_%a.out
#SBATCH --error=logs/lanta_semantic_%A_%a.err

set -euo pipefail

mkdir -p logs

# Edit these for the LANTA project environment.
REPO_DIR="${REPO_DIR:-$PWD}"
RECORDINGS="${RECORDINGS:-processed/all_castle/recordings.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-processed/lanta_semantic}"
FRAME_INTERVAL_SEC="${FRAME_INTERVAL_SEC:-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DEVICE="${DEVICE:-auto}"
PROCESSING_VERSION="${PROCESSING_VERSION:-castle-lanta-semantic-v1}"

cd "$REPO_DIR"

python scripts/lanta_process_castle_recording.py "$RECORDINGS" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --output-root "$OUTPUT_ROOT" \
  --frame-interval-sec "$FRAME_INTERVAL_SEC" \
  --processing-version "$PROCESSING_VERSION" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --detector v2 \
  --transcript-weight 0.0
