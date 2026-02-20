#!/bin/bash
#SBATCH --job-name=test_ppl         #
#SBATCH --array=0-3
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

MODELS=("train_RNNModel_base_data" "train_RNNModel_freq_16" "train_RNNModel_freq_8" "train_RNNModel_freq_32")
TEST_FILES="base_data/test.txt test_orc.txt"
CSV_NAMES="results_base_data.csv results_test_orc.csv"

for MODEL in "${MODELS[@]}"; do
    python test_models.py \
        --checkpoint_dir "checkpoint/$MODEL" \
        --test_files $TEST_FILES \
        --csv_names $CSV_NAMES
done

# Concatenate CSVs for each test set
for TEST in "${TEST_FILES[@]}"; do
    TEST_BASE=$(basename "$TEST" .txt)
    csvs=$(ls results_${TEST_BASE}_*.csv)
    paste -d',' $csvs > results_${TEST_BASE}_all.csv
done