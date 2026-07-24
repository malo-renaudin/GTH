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
#SBATCH --error=results/logs/error_eval__nested_%A_%a.log
#SBATCH --output=results/logs/output_eval__nested_%A_%a.log

source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate litgpt_jz

for ckpt in babylm-baseline-100m-gpt2/chck_*; do
    echo "Running $ckpt"
    python scripts/eval/eval_nested.py \
        --ckpt "$ckpt" \
        --eval-dataset eval_data/short_nested_inner_english.json \
        --out-metrics "results/$(basename $ckpt)_metrics.json"
done