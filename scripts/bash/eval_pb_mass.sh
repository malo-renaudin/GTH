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
#SBATCH --time=20:00:00

#SBATCH --array=0-3
#SBATCH --error=results/pb_mass/error_eval_%A_%a.log
#SBATCH --output=results/pb_mass/output_eval_%A_%a.log

source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate litgpt_jz

experiments=(
    "baseline_no_augmentation"
    "baseline_orc_augmented"
    "baseline_wh_augmented"
    "baseline_both_augmented"
)

EXP=${experiments[$SLURM_ARRAY_TASK_ID]}

echo "Running evaluation for: $EXP"

python scripts/eval/run_eval_many_checkpoints.py \
    --checkpoints results/${EXP} \
    --orc_test eval_data/orc_test.txt \
    --wh_test eval_data/wh_test.txt \
    --out-dir results/${EXP}/eval_summary/ \
    --device cuda