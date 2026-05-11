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
#SBATCH --time=02:00:00
#SBATCH --error=error_train_%a.log
#SBATCH --output=output_train_%a.log

source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate litgpt_jz


python eval_semantic_distractors_orc.py --checkpoint-dir out/pretrain/gpt_baseline_2 --input-csv orc_semantic_distractors.csv --result-name semantic_distractor_baseline.csv --batch-size 512