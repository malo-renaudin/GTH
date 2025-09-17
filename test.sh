#!/bin/bash
#SBATCH --job-name=test_transformer        
#SBATCH --output=job_outputs/job_%A_%a.out
#SBATCH --error=job_outputs/job_%A_%a.err
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

mkdir -p job_outputs

python src/language_models/main.py \
        --data english_data\
        --name test \
        --classmodel TransformerLM \
        --batch_size 512 \
        --emsize 650 \
        --nhid 650 \
        --nlayers 2 \
        --optimizer 'Adam' \
        --epochs 40 \
        --lr 0.001 \
        --cuda