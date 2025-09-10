#!/bin/bash
#SBATCH --job-name=create_datasets        # Job name
#SBATCH --partition=gpu 
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --cpus-per-task=6           # Number of CPU cores per task
#SBATCH --mem=32G                   # Memory per task
#SBATCH --time=2:00:00              # Adjust time as needed
#SBATCH --output=log/%x-%A_%a.log   # Log file per array task

# Load modules
module load miniconda3/24.3.0
module load python/3.9.18-lpwk
module load cuda/11.8.0-r465
source ~/.bashrc
conda activate leaps3

echo "Running job on $(hostname)"
which python
echo "Python version: $(python --version)"
echo "CUDA_DEVICE: $CUDA_VISIBLE_DEVICES"

python /scratch2/mrenaudin/GTH/stimuli_generator/generate_obj_wh_q.py
python /scratch2/mrenaudin/GTH/stimuli_generator/generate_orc.py