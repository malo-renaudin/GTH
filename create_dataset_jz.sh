#!/bin/bash
#SBATCH --job-name=create_dataset
#SBATCH --output=logs/grid_%A_%a.out
#SBATCH --error=logs/grid_%A_%a.err
#SBATCH --time=20:00:00  
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=h100
#SBATCH --account=ywa@h100
#SBATCH --hint=nomultithread
#SBATCH --partition=gpu_p6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=malorenaudin1@gmail.com
#SBATCH --qos=qos_gpu_h100-t3

cd $WORK/GTH



# (1) Clean slate
module purge
module load cuda/12.2.0 nccl/2.19.3-1-cuda cudnn/8.9.5.30-cuda gcc/10.1.0 openmpi/4.1.5-cuda intel-mkl/2020.4 magma/2.7.2-cuda
module load sox/14.4.2 sparsehash/2.0.3 libjpeg-turbo/2.1.3
module load pytorch-gpu/py3/2.2.0

export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

python stimuli_generator/generate_obj_wh_q.py
python stimuli_generator/generate_orc.py
