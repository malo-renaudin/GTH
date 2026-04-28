#!/bin/bash
#SBATCH --job-name=ppl_dist_gpt2_wh
#SBATCH --output=log/ppl_distribution_%j.out
#SBATCH --error=log/ppl_distribution_%j.err
#SBATCH --time=16:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-p2


echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "=========================================="

# Load modules if needed (check with: module avail)
# module load cuda/11.8
# module load python/3.9

# Activate conda environment
source activate leaps3

# Or activate virtual environment
# source /scratch2/mrenaudin/GTH/stimuli_generator/english-stimuli-generator/.venv/bin/activate

# Check GPU availability
nvidia-smi
echo ""

# Configuration
# MODEL_PATH="${MODEL_PATH:-hidden650_batch128_dropout0.2_lr20.0.pt}"
DATA_DIR="${DATA_DIR:-data/english_data}"
SPLIT="${SPLIT:-wh.txt}"
BATCH_SIZE="${BATCH_SIZE:-256}"
OUTPUT_FILE="${OUTPUT_FILE:-}"

echo "Configuration:"
echo "  Model: GPT-2"
echo "  Data: $DATA_DIR"
echo "  Split: $SPLIT"
echo "  Batch size: $BATCH_SIZE"
echo "=========================================="
echo ""

# Run the script
python compute_ppl_per_sentence.py \
    --use-gpt2 \
    --data "$DATA_DIR" \
    --split "$SPLIT" \
    --batch-size "$BATCH_SIZE" \
    ${OUTPUT_FILE:+--output "$OUTPUT_FILE"} \
    --save-individual

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
