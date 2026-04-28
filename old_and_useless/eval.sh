#!/bin/bash
#SBATCH --job-name=evaluation  # Fixed spacing
#SBATCH --array=0-8
#SBATCH --partition=gpu
#SBATCH --export=ALL
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=log/%x-%A_%a.log  # %A=job_id, %a=array_index

# Load modules
module load miniconda3/24.3.0
module load python/3.9.18-lpwk
module load cuda/11.8.0-r465

source ~/.bashrc
conda activate leaps3

# Set PYTHONPATH for imports
export PYTHONPATH="/scratch2/mrenaudin/GTH:${PYTHONPATH}"

echo "Running job on $(hostname)"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
which python
echo "python-version $(python --version)"
echo "CUDA_DEVICE: $CUDA_VISIBLE_DEVICES"

# Define all configurations
DATA_PATHS=(
    "data/english_data"
    "data/modulated_sets/sRC0.00009_sQ0.00250_tRC0.00009_tQ0.00500"
    "data/modulated_sets/sRC0.00009_sQ0.00250_tRC0.00009_tQ0.02000"
    "data/modulated_sets/sRC0.00009_sQ0.00250_tRC0.00020_tQ0.00250"
    "data/modulated_sets/sRC0.00009_sQ0.00250_tRC0.00020_tQ0.00500"
    "data/modulated_sets/sRC0.00009_sQ0.00250_tRC0.00020_tQ0.02000"
    "data/modulated_sets/sRC0.00009_sQ0.00250_tRC0.00100_tQ0.00250"
    "data/modulated_sets/sRC0.00009_sQ0.00250_tRC0.00100_tQ0.00500"
    "data/modulated_sets/sRC0.00009_sQ0.00250_tRC0.00100_tQ0.02000"
)

CHECKPOINT_DIRS=(
    "checkpoints/train_RNNModel_english_data"
    "checkpoints/train_RNNModel_sRC0.00009_sQ0.00250_tRC0.00009_tQ0.00500"
    "checkpoints/train_RNNModel_sRC0.00009_sQ0.00250_tRC0.00009_tQ0.02000"
    "checkpoints/train_RNNModel_sRC0.00009_sQ0.00250_tRC0.00020_tQ0.00250"
    "checkpoints/train_RNNModel_sRC0.00009_sQ0.00250_tRC0.00020_tQ0.00500"
    "checkpoints/train_RNNModel_sRC0.00009_sQ0.00250_tRC0.00020_tQ0.02000"
    "checkpoints/train_RNNModel_sRC0.00009_sQ0.00250_tRC0.00100_tQ0.00250"
    "checkpoints/train_RNNModel_sRC0.00009_sQ0.00250_tRC0.00100_tQ0.00500"
    "checkpoints/train_RNNModel_sRC0.00009_sQ0.00250_tRC0.00100_tQ0.02000"
)

# Get current task's configuration
DATA_PATH=${DATA_PATHS[$SLURM_ARRAY_TASK_ID]}
CHECKPOINT_DIR=${CHECKPOINT_DIRS[$SLURM_ARRAY_TASK_ID]}

echo "Processing: $DATA_PATH"
echo "Checkpoint: $CHECKPOINT_DIR"

# Run only this task's job
python tests/scripts/analyze_minimal_pairs.py \
    --data_path "$DATA_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR"

echo "Job completed successfully"