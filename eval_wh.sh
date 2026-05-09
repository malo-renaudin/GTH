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
#SBATCH --array=0-2
#SBATCH --error=error_eval_wh_%a.log
#SBATCH --output=output_eval_wh_%a.log

source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate litgpt_jz

MODELS=(out/pretrain/gpt_wh_2 out/pretrain/gpt_orc_2 out/pretrain/gpt_baseline_2)

CHECKPOINT_DIR=${MODELS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME=$(basename $CHECKPOINT_DIR)

python eval_test.py \
    --checkpoint-dir $CHECKPOINT_DIR \
    --sentences-file data/english_data/wh.txt \
    --structure wh \
    --batch-size 512 \
    --result-name np_mass_${MODEL_NAME}_wh.csv
