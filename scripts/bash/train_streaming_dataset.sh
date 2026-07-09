#!/bin/bash
#SBATCH --account=ywa@h100
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --array=0-3
#SBATCH --error=results/logs/error_train_streaming_%A_%a.log
#SBATCH --output=results/logs/output_train_streaming_%A_%a.log

source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate litgpt_jz

case ${SLURM_ARRAY_TASK_ID} in
    0)
        OUTPUT_DIR="results/orc_augmented_full_datasets"
        C4=0.9
        ORC=0.1
        WH=0.0
        ;;
    1)
        OUTPUT_DIR="results/wh_augmented_full_datasets"
        C4=0.9
        ORC=0.0
        WH=0.1
        ;;
    2)
        OUTPUT_DIR="results/both_augmented_full_datasets"
        C4=0.9
        ORC=0.05
        WH=0.05
        ;;
    3)
        OUTPUT_DIR="results/baseline_full_datasets"
        C4=1.0
        ORC=0.0
        WH=0.0
        ;;
esac

echo "Running configuration ${SLURM_ARRAY_TASK_ID}"
echo "Output: ${OUTPUT_DIR}"
echo "C4=${C4} ORC=${ORC} WH=${WH}"

python scripts/train/streaming_dataset.py \
    --output-dir "${OUTPUT_DIR}" \
    --blimp-dir eval_data/blimp_data \
    --nested-inner eval_data/short_nested_inner_english.json \
    --nested-outer eval_data/short_nested_outer_english.json \
    --filler-gap-orc eval_data/filler_gap_orc.csv \
    --filler-gap-wh eval_data/filler_gap_wh.csv \
    --c4 "${C4}" \
    --orc "${ORC}" \
    --wh "${WH}"