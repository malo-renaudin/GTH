#!/bin/bash
#SBATCH --job-name=test_ppl         #
#SBATCH --array=0-1
#SBATCH --output=job_outputs/test_ppl/job_%A_%a.out
#SBATCH --error=job_outputs/test_ppl/job_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --constraint=h100
#SBATCH --account=ywa@h100
#SBATCH --hint=nomultithread
#SBATCH --partition=gpu_p6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=malorenaudin1@gmail.com
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --signal=SIGUSR1@90

MODELS=("train_RNNModel_freq_4" "train_RNNModel_freq_6")
TEST_FILES="base_data/test.txt test_orc.txt"

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

python test_ppl.py \
    --checkpoint_dir "checkpoints/$MODEL" \
    --test_files $TEST_FILES \
    --model_name "$MODEL"

