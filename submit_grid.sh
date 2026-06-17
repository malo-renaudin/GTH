#!/usr/bin/env bash
#SBATCH --account=ywa@h100
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --constraint=h100
#SBATCH --job-name=grid-pretrain
#SBATCH --array=1-48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=logs/grid_%A_%a.out
#SBATCH --error=logs/grid_%A_%a.err

source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate litgpt_jz

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

MANIFEST="configs/grid/manifest.tsv"

if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest not found: $MANIFEST" >&2
    exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID}"

LINE=$(sed -n "${TASK_ID}p" "$MANIFEST")

if [[ -z "$LINE" ]]; then
    echo "No manifest line for index $TASK_ID" >&2
    exit 1
fi

IFS=$'\t' read -r CONFIG OUT_DIR <<< "$LINE"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

echo "[$(date)] Task $TASK_ID"
echo "Config: $CONFIG"
echo "Out dir: $OUT_DIR"

litgpt pretrain --config "$CONFIG"

if [[ -d "$OUT_DIR" ]]; then
    cd "$OUT_DIR"

    last=$(find . -maxdepth 1 -type d -name 'step-*' \
        | sed 's|^\./||' \
        | sort -V \
        | tail -n1)

    if [[ -n "$last" ]]; then
        echo "Keeping final checkpoint: $last"

        find . -maxdepth 1 -type d -name 'step-*' ! -name "$last" \
            -exec rm -rf {} +
    else
        echo "No step-* directories found"
    fi
else
    echo "Out dir does not exist: $OUT_DIR" >&2
fi

echo "Job finished for index $TASK_ID"

RESULTS_CSV="${SLURM_SUBMIT_DIR}/configs/grid/results.csv"

if command -v python3 >/dev/null 2>&1; then
    python3 "${SLURM_SUBMIT_DIR}/scripts/record_final_metrics.py" \
        --config "$CONFIG" \
        --out-dir "$OUT_DIR" \
        --results-csv "$RESULTS_CSV" \
    || echo "Failed to record metrics" >&2
else
    echo "Python not found; skipping metrics recording" >&2
fi