#!/usr/bin/env bash
#SBATCH --job-name=castle-merge-semantic
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/lanta_merge_%j.out
#SBATCH --error=logs/lanta_merge_%j.err

set -euo pipefail

mkdir -p logs

REPO_DIR="${REPO_DIR:-$PWD}"
OUTPUT_ROOT="${OUTPUT_ROOT:-processed/lanta_semantic}"

cd "$REPO_DIR"

python scripts/merge_castle_semantic_outputs.py "$OUTPUT_ROOT" \
  --output-manifest "$OUTPUT_ROOT/semantic_events.jsonl" \
  --output-embeddings "$OUTPUT_ROOT/semantic_embeddings.npz"
