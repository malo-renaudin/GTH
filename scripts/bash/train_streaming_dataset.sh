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
#SBATCH --array=0-8
#SBATCH --error=results/logs/error_train_streaming_%A_%a.log
#SBATCH --output=results/logs/output_train_streaming_%A_%a.log
# export PATH=/lustre/fswork/projects/rech/ldh/una68ug/malo/envs/litgpt_jz/bin:$PATH

# export HF_HOME=/lustre/fswork/projects/rech/ldh/una68ug/malo/GTH/scripts/train/.cache
# export HF_HUB_CACHE=$HF_HOME/hub
# export HF_HUB_DISABLE_XET=1

source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate litgpt_jz

case ${SLURM_ARRAY_TASK_ID} in
    0)
        OUTPUT_DIR="results/orc_05"
        C4=0.97
        ORC=0.005
        WH=0.0
        SVO=0.025
        ;;
    1)
        OUTPUT_DIR="results/orc_1"
        C4=0.97
        ORC=0.01
        SVO=0.02
        WH=0.0
        ;;
    2)
        OUTPUT_DIR="results/orc_3"
        C4=0.97
        ORC=0.03
        WH=0.0
        SVO=0.0
        ;;
    3)
        OUTPUT_DIR="results/wh_05"
        C4=0.97
        ORC=0.0
        WH=0.005
        SVO=0.025
        ;;

    4)
        OUTPUT_DIR="results/wh_1"
        C4=0.97
        ORC=0.0
        WH=0.01
        SVO=0.02
        ;;
    5)
        OUTPUT_DIR="results/wh_3"
        C4=0.97
        ORC=0.0
        WH=0.03
        SVO=0.0
        ;;
    
    6)
        OUTPUT_DIR="results/svo_3"
        C4=0.97
        ORC=0.0
        WH=0.0
        SVO=0.03
        ;;
    7)
        OUTPUT_DIR="results/both_1_5_each"
        C4=0.97
        ORC=0.015
        WH=0.015
        SVO=0.0
        ;;
    8)
        OUTPUT_DIR="results/both_1_each"
        C4=0.97
        ORC=0.01
        WH=0.01
        SVO=0.01
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
    --filler-gap-orc-iv eval_data/filler_gap_factorial_iv.csv \
    --filler-gap-orc-oov eval_data/filler_gap_factorial_oov.csv \
    --filler-gap-wh-iv eval_data/wh_movement_factorial_iv.csv \
    --filler-gap-wh-oov eval_data/wh_movement_factorial_oov.csv \
    --probability-masses-orc-iv eval_data/orc_test_1.txt \
    --probability-masses-orc-oov eval_data/orc_test_1.txt \
    --probability-masses-wh-iv eval_data/wh_test_1.txt \
    --probability-masses-wh-oov eval_data/wh_test_1.txt \
    --c4 "${C4}" \
    --orc "${ORC}" \
    --wh "${WH}" \
    --svo "${SVO}"