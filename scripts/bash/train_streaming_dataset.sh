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
#SBATCH --error=error_train_streaming%a.log
#SBATCH --output=output_train_streaming_%a.log

source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate litgpt_jz

python scripts/train/streaming_dataset.py --output-dir results