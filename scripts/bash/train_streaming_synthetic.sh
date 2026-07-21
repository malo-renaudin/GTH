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
#SBATCH --array=0-4
#SBATCH --error=results/logs/error_train_streaming_%A_%a.log
#SBATCH --output=results/logs/output_train_streaming_%A_%a.log

source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate litgpt_jz

case ${SLURM_ARRAY_TASK_ID} in
    0)
        OUTPUT_DIR="results/childes_5_5"
        C4=0
        ORC=0.05
        WH=0.05
        SVO=0.9
        ;;
    1)
        OUTPUT_DIR="results/childes_4_6"
        C4=0
        ORC=0.04
        SVO=0.06
        WH=0.9
        ;;
    2)
        OUTPUT_DIR="results/childes_6_4"
        C4=0
        ORC=0.06
        WH=0.04
        SVO=0.0
        ;;
    3)
        OUTPUT_DIR="results/childes_wh"
        C4=0
        ORC=0.0
        WH=0.1
        SVO=0.9
        ;;

    4)
        OUTPUT_DIR="results/childes_orc"
        C4=0
        ORC=0.1
        WH=0
        SVO=0.9
        ;;
    
esac

echo "Running configuration ${SLURM_ARRAY_TASK_ID}"
echo "Output: ${OUTPUT_DIR}"
echo "C4=${C4} ORC=${ORC} WH=${WH} SVO=${SVO}"

python scripts/train/streaming_dataset.py \
    --output-dir "${OUTPUT_DIR}" \
    --blimp-dir eval_data/blimp_data \
    --nested-inner eval_data/short_nested_inner_english.json \
    --nested-outer eval_data/short_nested_outer_english.json \
    --filler-gap-orc-iv eval_data/orc_factorial_iv.csv \
    --filler-gap-orc-oov eval_data/orc_factorial_oov.csv \
    --filler-gap-wh-iv eval_data/wh_factorial_iv.csv \
    --filler-gap-wh-oov eval_data/wh_factorial_oov.csv \
    --c4 "${C4}" \
    --orc "${ORC}" \
    --wh "${WH}" \
    --svo "${SVO}"
