#!/usr/bin/env bash
#SBATCH --account=ywa@h100
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --constraint=h100
#SBATCH --job-name=grid-pretrain
#SBATCH --array=0-54
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=logs/grid_%A_%a.out
#SBATCH --error=logs/grid_%A_%a.err

set -e

# -------------------------
# ENV
# -------------------------
source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate litgpt_jz

mkdir -p logs
mkdir -p results

# -------------------------
# CONFIG LIST
# -------------------------
CONFIG_DIR="configs/hf/grid_search"
CONFIGS=($(ls $CONFIG_DIR/*.yaml | sort))

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Using config: $CONFIG"

# extract run name (important for metrics merging)
RUN_NAME=$(basename "$CONFIG" .yaml)

# each run gets its own folder
RUN_DIR="results/${RUN_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RUN_DIR"

echo "Run directory: $RUN_DIR"

# -------------------------
# TRAIN
# -------------------------
echo "Starting training..."

python scripts/train/train.py \
  --config "$CONFIG" \
  --dataset-name baseline \
  --model-name gpt2 \
  --cache-dir scripts/train/.cache \
  --output-dir "$RUN_DIR"

echo "Training done."

# -------------------------
# PICK FINAL CHECKPOINT
# -------------------------
CKPT="$RUN_DIR/final"

if [ ! -d "$CKPT" ]; then
    echo "Final checkpoint not found, falling back to latest checkpoint..."

    CKPT=$(ls -td "$RUN_DIR"/checkpoint-* 2>/dev/null | head -1)
fi

echo "Using checkpoint: $CKPT"

# -------------------------
# EVAL (METRICS OUTPUT)
# -------------------------
echo "Starting evaluation..."

python scripts/eval/eval_nested.py \
  --ckpt "$CKPT" \
  --eval-dataset scripts/eval/eval_data/short_nested_inner.json \
  --out-metrics "$RUN_DIR/eval_metrics.json"

echo "Done."
echo "Saved run in: $RUN_DIR"