#!/bin/bash
#SBATCH --account=ywa@h100
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=1:00:00
#SBATCH --array=0-4
#SBATCH --error=results/logs/error_eval_fg_%A_%a.log
#SBATCH --output=results/logs/output_eval_fg_%A_%a.log


source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate litgpt_jz


TASKS=(
    "orc"
    "wh"
    "orc_tokens"
    "wh_tokens"
)

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

echo "Running task: $TASK"


case $TASK in

orc)
    python scripts/eval/acc_fg.py \
        --data_dir eval_data/relativizers2 \
        --checkpoint_dir babylm-baseline-100m-gpt2 \
        --output_file results/accuracy_fg_final/orc.csv
    ;;

wh)
    python scripts/eval/acc_fg.py \
        --data_dir eval_data/wh_fillers2 \
        --checkpoint_dir babylm-baseline-100m-gpt2 \
        --output_file results/accuracy_fg_final/wh.csv
    ;;

orc_tokens)
    python scripts/eval/acc_fg_all_tokens.py \
        --data_dir eval_data/relativizers2 \
        --checkpoint_dir babylm-baseline-100m-gpt2 \
        --output_file results/accuracy_fg_final/orc_tokens.csv
    ;;

wh_tokens)
    python scripts/eval/acc_fg_all_tokens.py \
        --data_dir eval_data/wh_fillers2 \
        --checkpoint_dir babylm-baseline-100m-gpt2 \
        --output_file results/accuracy_fg_final/wh_tokens.csv
    ;;

*)
    echo "Unknown task"
    exit 1
    ;;

esac