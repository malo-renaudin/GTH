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
#SBATCH --error=error_transitivity_%a.log
#SBATCH --output=output_transitivity_%a.log

source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate litgpt_jz

python eval_transitivity_orc.py --checkpoint-dir out/pretrain/gpt_baseline_2 --input-csv orc_transitivity.csv --result-name transitivity_baseline.csv --batch-size 512