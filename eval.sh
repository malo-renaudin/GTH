#!/bin/bash
#SBATCH --job-name=checkpoint_eval
#SBATCH --array=0-19  
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#SBATCH --constraint=h100
#SBATCH --account=ywa@h100
#SBATCH --hint=nomultithread
#SBATCH --partition=gpu_p6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=malorenaudin1@gmail.com
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --signal=SIGUSR1@90

# MODIFY THESE PATHS
PARENT_DIR="checkpoints"
PYTHON_SCRIPT="pm.py"
QUESTIONS_FILE="structure_test/questions_test.txt"
RELATIVE_FILE="structure_test/relative_clauses_test.txt"

# Dictionary mapping checkpoint names to data paths
# MODIFY THIS DICTIONARY
declare -A DATA_PATHS=(
    ["train_RNNModel_english_data"]="english_data"
    ["train_RNNModel_rc_0008_q_0050"]="modulated_sets/rc_0008_q_0050"
    ["train_RNNModel_rc_0008_q_0100"]="modulated_sets/rc_0008_q_0100"
    ["train_RNNModel_rc_0008_q_0200"]="modulated_sets/rc_0008_q_0200"
    ["train_RNNModel_rc_0016_q_0050"]="modulated_sets/rc_0016_q_0050"
    ["train_RNNModel_rc_0016_q_0100"]="modulated_sets/rc_0016_q_0100"
    ["train_RNNModel_rc_0016_q_0200"]="modulated_sets/rc_0016_q_0200"
    ["train_RNNModel_rc_0032_q_0050"]="modulated_sets/rc_0032_q_0050"
    ["train_RNNModel_rc_0032_q_0100"]="modulated_sets/rc_0032_q_0100"
    ["train_RNNModel_rc_0032_q_0200"]="modulated_sets/rc_0032_q_0200"
    ["train_TransformerLM_english_data"]="english_data"
    ["train_TransformerLM_rc_0008_q_0050"]="modulated_sets/rc_0008_q_0050"
    ["train_TransformerLM_rc_0008_q_0100"]="modulated_sets/rc_0008_q_0100"
    ["train_TransformerLM_rc_0008_q_0200"]="modulated_sets/rc_0008_q_0200"
    ["train_TransformerLM_rc_0016_q_0050"]="modulated_sets/rc_0016_q_0050"
    ["train_TransformerLM_rc_0016_q_0100"]="modulated_sets/rc_0016_q_0100"
    ["train_TransformerLM_rc_0016_q_0200"]="modulated_sets/rc_0016_q_0200"
    ["train_TransformerLM_rc_0032_q_0050"]="modulated_sets/rc_0032_q_0050"
    ["train_TransformerLM_rc_0032_q_0100"]="modulated_sets/rc_0032_q_0100"
    ["train_TransformerLM_rc_0032_q_0200"]="modulated_sets/rc_0032_q_0200"
)

# Load modules for Jean Zay
module purge
module load pytorch-gpu/py3/2.1.1

# Install NLTK if not already installed (using the Python from pytorch module)
python -m pip install --user nltk

# Download NLTK data
python -c "
import nltk
import os
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
try:
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir, quiet=True)
    print('NLTK data downloaded successfully')
except Exception as e:
    print(f'Error downloading NLTK data: {e}')
"

# Set NLTK data path
export NLTK_DATA="$HOME/nltk_data"

# Get checkpoint directories
CHECKPOINTS=($(find "$PARENT_DIR" -mindepth 1 -maxdepth 1 -type d | sort))
CHECKPOINT_DIR="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT_NAME=$(basename "$CHECKPOINT_DIR")
DATA_PATH="${DATA_PATHS[$CHECKPOINT_NAME]}"

mkdir -p logs results

echo "Processing: $CHECKPOINT_NAME"
echo "Data path: $DATA_PATH"

# Run both evaluations
python "$PYTHON_SCRIPT" --data_path "$DATA_PATH" --checkpoint_dir "$CHECKPOINT_DIR" \
    --test_file "$QUESTIONS_FILE" --output_dir "results/$CHECKPOINT_NAME" \
    --test_type questions --device auto

python "$PYTHON_SCRIPT" --data_path "$DATA_PATH" --checkpoint_dir "$CHECKPOINT_DIR" \
    --test_file "$RELATIVE_FILE" --output_dir "results/$CHECKPOINT_NAME" \
    --test_type relative_clauses --device auto

echo "Done: $CHECKPOINT_NAME"