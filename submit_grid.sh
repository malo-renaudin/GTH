#!/usr/bin/env bash
# SLURM array submission wrapper for grid configs produced by `scripts/make_grid_configs.py`.
# Usage (submit as an array job):
#   sbatch --array=1-N submit_grid.sh
# Or run locally for a single index: ./submit_grid.sh 1

# Example SBATCH header (uncomment and adapt to your cluster):
#SBATCH --job-name=grid-pretrain
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/grid_%A_%a.out
#SBATCH --error=logs/grid_%A_%a.err

set -euo pipefail

MANIFEST="configs/grid/manifest.tsv"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST"
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-}}
if [[ -z "$TASK_ID" ]]; then
  echo "Provide SLURM_ARRAY_TASK_ID or pass index as first arg" >&2
  exit 1
fi

LINE=$(sed -n "${TASK_ID}p" "$MANIFEST")
if [[ -z "$LINE" ]]; then
  echo "No manifest line for index $TASK_ID" >&2
  exit 1
fi

CONFIG=$(echo "$LINE" | awk -F"\t" '{print $1}')
OUT_DIR=$(echo "$LINE" | awk -F"\t" '{print $2}')

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

echo "Running config: $CONFIG" >&2
echo "Out dir: $OUT_DIR" >&2

# Run training (assumes `litgpt` is on PATH and env is set up)
litgpt pretrain --config "$CONFIG"

# After successful run, keep only the final step-* checkpoint directory
if [[ -d "$OUT_DIR" ]]; then
  cd "$OUT_DIR"
  # find last step directory (lexicographic numeric sort)
  last=$(ls -d step-* 2>/dev/null | sort -V | tail -n1 || true)
  if [[ -n "$last" ]]; then
    echo "Keeping final checkpoint: $last" >&2
    for d in step-*; do
      if [[ "$d" != "$last" ]]; then
        echo "Removing $d" >&2
        rm -rf "$d"
      fi
    done
  else
    echo "No step-* directories found in $OUT_DIR" >&2
  fi
else
  echo "Out dir does not exist: $OUT_DIR" >&2
fi

echo "Job finished for index $TASK_ID" >&2
