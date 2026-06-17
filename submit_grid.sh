#!/usr/bin/bash
#SBATCH --account=ywa@h100
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --constraint=h100
#SBATCH --job-name=grid-pretrain
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
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

# Record final metrics (train/valid perplexity) into CSV
PY=python3
RESULTS_CSV="configs/grid/results.csv"
if command -v $PY >/dev/null 2>&1; then
  echo "Recording final metrics to $RESULTS_CSV" >&2
  $PY scripts/record_final_metrics.py --config "$CONFIG" --out-dir "$OUT_DIR" --results-csv "$RESULTS_CSV" || echo "Failed to record metrics" >&2
else
  echo "Python not found; skipping metrics recording" >&2
fi
