#!/bin/bash
#SBATCH --job-name=lm_train         # Job name
#SBATCH --partition=gpu 
#SBATCH --export=ALL 
#SBATCH --cpus-per-task=6           # Number of CPU cores per task
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --mem=32G                   # Memory per task
#SBATCH --time=2:00:00              # Adjust time as needed
#SBATCH --output=log/%x-%A_%a.log   # Log file per array task
#SBATCH --array=0-18                 # Number of tasks - adjust based on your dataset/model combinations

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

# Define datasets and model configs
DATASETS=(
    "/scratch2/mrenaudin/colorlessgreenRNNs/english_data"
    "/scratch2/mrenaudin/GTH/generated_train_sets/rc_0_08_q_0_50"
    "/scratch2/mrenaudin/GTH/generated_train_sets/rc_0_08_q_1_00"
    "/scratch2/mrenaudin/GTH/generated_train_sets/rc_0_16_q_0_50"
    "/scratch2/mrenaudin/GTH/generated_train_sets/rc_0_16_q_1_00"
    "/scratch2/mrenaudin/GTH/generated_train_sets/rc_0_32_q_2_00"
)

MODELS=(
    "RNNModel"
    "TransformerLM"
)

# Each array task runs a unique combination of dataset and model
# Compute indices
TOTAL_DATASETS=${#DATASETS[@]}
TOTAL_MODELS=${#MODELS[@]}
TASK_INDEX=$SLURM_ARRAY_TASK_ID

# Compute dataset and model indices
DATASET_INDEX=$(( TASK_INDEX % TOTAL_DATASETS ))
MODEL_INDEX=$(( TASK_INDEX / TOTAL_DATASETS ))

# Get dataset and model for this task
DATASET=${DATASETS[$DATASET_INDEX]}
MODEL=${MODELS[$MODEL_INDEX]}

# Set experiment name
BASENAME=$(basename $DATASET)
EXPERIMENT_NAME="train_${MODEL}_${BASENAME}"

# Run training
if [ "$MODEL" == "RNNModel" ]; then
    python src/language_models/main.py \
        --data $DATASET \
        --name $EXPERIMENT_NAME \
        --classmodel $MODEL \
        --batch_size 512 \
        --emsize 650 \
        --nhid 650 \
        --nlayers 2 \
        --optimizer 'Adam' \
        --epochs 40 \
        --cuda
else
    python src/language_models/main.py \
        --data $DATASET \
        --name $EXPERIMENT_NAME \
        --classmodel $MODEL \
        --batch_size 512 \
        --emsize 650 \
        --nheads 10 \
        --d_model 650 \
        --d_ff 2600 \
        --nlayers 2 \
        --optimizer 'Adam' \
        --epochs 40 \
        --cuda
fi
