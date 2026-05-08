#!/bin/bash
#SBATCH --account=ywa@h100
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
# Set --array to 0-$((${#MODELS[@]} * 4 - 1)), e.g. 0-3 for 1 model, 0-7 for 2 models
#SBATCH --array=0-7
#SBATCH --error=error_eval_%a.log
#SBATCH --output=output_eval_%a.log

source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate litgpt_jz

# Add models here; update --array upper bound to (N_MODELS * 4 - 1)
MODELS=(out/pretrain/gpt_wh_2 out/pretrain/gpt_orc_2)

NUM_TASKS=4
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / NUM_TASKS))
TASK_IDX=$((SLURM_ARRAY_TASK_ID % NUM_TASKS))

CHECKPOINT_DIR=${MODELS[$MODEL_IDX]}
MODEL_NAME=$(basename $CHECKPOINT_DIR)

STRUCTURES=(orc wh orc wh)
SENT_FILES=(data/valid_orc7_not_in_train_orc_6_72.txt data/valid_wh5_not_in_wh_7_20.txt "" "")
FILLER_GAP_INPUTS=("" "" orc_fillergap_100.csv wh_fillergap_100.csv)

STRUCTURE=${STRUCTURES[$TASK_IDX]}
SENT_FILE=${SENT_FILES[$TASK_IDX]}
FILLER_GAP_INPUT=${FILLER_GAP_INPUTS[$TASK_IDX]}

if [[ $TASK_IDX -le 1 ]]; then
    python eval_test.py \
        --checkpoint-dir $CHECKPOINT_DIR \
        --sentences-file $SENT_FILE \
        --structure $STRUCTURE \
        --batch-size 512 \
        --result-name np_mass_${MODEL_NAME}_${STRUCTURE}.csv
else
    python eval_filler_gap.py \
        --checkpoint-dir $CHECKPOINT_DIR \
        --input-csv $FILLER_GAP_INPUT \
        --output-csv filler_gap_${MODEL_NAME}_${STRUCTURE}.csv \
        --batch-size 512
fi