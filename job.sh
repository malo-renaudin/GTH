#!/bin/bash
#SBATCH --job-name=lm_train         #
#SBATCH --array=0-15
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

# Create output directory
mkdir -p job_outputs

# Define datasets and model configs
DATASETS=(
    "modulated_sets/sRC0.00009_sQ0.00250_tRC0.00009_tQ0.00500"
    "modulated_sets/sRC0.00009_sQ0.00250_tRC0.00009_tQ0.02000"
    "modulated_sets/sRC0.00009_sQ0.00250_tRC0.00020_tQ0.00250"
    "modulated_sets/sRC0.00009_sQ0.00250_tRC0.00020_tQ0.00500"
    "modulated_sets/sRC0.00009_sQ0.00250_tRC0.00020_tQ0.02000"
    "modulated_sets/sRC0.00009_sQ0.00250_tRC0.00100_tQ0.00250"
    "modulated_sets/sRC0.00009_sQ0.00250_tRC0.00100_tQ0.00500"
    "modulated_sets/sRC0.00009_sQ0.00250_tRC0.00100_tQ0.02000"
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
        --lr 0.001 \
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
        --d_ff 1300 \
        --nlayers 2 \
        --optimizer 'Adam' \
        --epochs 40 \
        --lr 0.001 \
        --bptt 128 \
        --cuda
fi
