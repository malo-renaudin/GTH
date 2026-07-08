#!/bin/bash
#SBATCH --account=ywa@h100
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00

# Set this to the array range you need: 0-(N-1)
# Replace N-1 with number_of_checkpoints-1 before submitting.
#SBATCH --array=0-9

#SBATCH --error=results/eval_filler_gap/error_%A_%a.log
#SBATCH --output=results/eval_filler_gap/output_%A_%a.log

# Environment / conda setup (adapt to your env)
source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate litgpt_jz

# User-configurable variables (edit)
CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-results/baseline_wh_augmented}
INPUT_CSV=${INPUT_CSV:-generate_simple_datasets/generated_simple_datasets/filler_gap_wh.csv}
OUT_DIR=${OUT_DIR:-results/baseline_wh_augmented/eval_filler_gap}
TOKENIZER_DIR=${TOKENIZER_DIR:-}
BATCH_SIZE=${BATCH_SIZE:-32}

# Build list of checkpoint subdirs
mapfile -t CKPTS < <(find "${CHECKPOINTS_DIR}" -maxdepth 1 -mindepth 1 -type d | sort)
NUM=${#CKPTS[@]}
echo "Found $NUM checkpoints in ${CHECKPOINTS_DIR}"
if [ ${SLURM_ARRAY_TASK_ID} -ge ${NUM} ]; then
  echo "SLURM_ARRAY_TASK_ID (${SLURM_ARRAY_TASK_ID}) out of range (0..$((NUM-1)))"
  exit 0
fi

CKPT="${CKPTS[${SLURM_ARRAY_TASK_ID}]}"
CKPT_NAME=$(basename "$CKPT")
mkdir -p "$OUT_DIR"

OUT_CSV="${OUT_DIR}/${CKPT_NAME}_filler_gap.csv"

CMD=(python scripts/eval/eval_filler_gap.py --checkpoint "$CKPT" --input-csv "$INPUT_CSV" --output-csv "$OUT_CSV" --batch-size "$BATCH_SIZE")
if [ -n "$TOKENIZER_DIR" ]; then
  CMD+=(--tokenizer-dir "$TOKENIZER_DIR")
fi

echo "Running checkpoint [$SLURM_ARRAY_TASK_ID]: $CKPT"
echo "${CMD[@]}"
"${CMD[@]}"