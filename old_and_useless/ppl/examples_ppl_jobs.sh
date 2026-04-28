#!/bin/bash
# Example usage scripts for computing perplexity distributions

# =============================================================================
# Basic usage with default settings
# =============================================================================
echo "Example 1: Submit job with default settings"
echo "sbatch job_compute_ppl.sh"
echo ""

# =============================================================================
# Specify custom model and data
# =============================================================================
echo "Example 2: Custom model and data"
echo "sbatch --export=MODEL_PATH=checkpoints/train_RNNModel_english_data/model_best.pt,DATA_DIR=data/modulated_sets/rc_0008_q_0050 job_compute_ppl.sh"
echo ""

# =============================================================================
# Run on test set instead of validation
# =============================================================================
echo "Example 3: Evaluate on test set"
echo "sbatch --export=SPLIT=test.txt job_compute_ppl.sh"
echo ""

# =============================================================================
# Run directly without job submission (on current node)
# =============================================================================
echo "Example 4: Run directly without SLURM"
echo "python compute_ppl_per_sentence.py --model hidden650_batch128_dropout0.2_lr20.0.pt --data data/english_data --split valid.txt"
echo ""

# =============================================================================
# Batch process multiple models
# =============================================================================
echo "Example 5: Process all models in a directory"
echo "for model in checkpoints/*/model_best.pt; do"
echo "    sbatch --export=MODEL_PATH=\$model job_compute_ppl.sh"
echo "done"
echo ""

# =============================================================================
# Interactive session with GPU
# =============================================================================
echo "Example 6: Run in interactive GPU session"
echo "srun --gres=gpu:1 --mem=32G --pty bash"
echo "# Then run directly:"
echo "python compute_ppl_per_sentence.py --model <model> --data <data> --batch-size 512"
echo ""

# =============================================================================
# Check job status and output
# =============================================================================
echo "Example 7: Monitor jobs"
echo "squeue -u \$USER                    # Check job queue"
echo "tail -f log/ppl_distribution_*.out # Watch output"
echo "scancel <job_id>                   # Cancel a job"
