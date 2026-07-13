#!/bin/bash
#SBATCH --account=ldh@h100
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --error=results/logs/error_missing_one_train_streaming_%A_%a.log
#SBATCH --output=results/logs/output_missing_one_train_streaming_%A_%a.log
export PATH=/lustre/fswork/projects/rech/ldh/una68ug/malo/envs/litgpt_jz/bin:$PATH

export HF_HOME=/lustre/fswork/projects/rech/ldh/una68ug/malo/GTH/scripts/train/.cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_DISABLE_XET=1
python scripts/train/streaming_dataset.py \
    --output-dir "results/wh_augmented_good_datasets"" \
    --blimp-dir eval_data/blimp_data \
    --nested-inner eval_data/short_nested_inner_english.json \
    --nested-outer eval_data/short_nested_outer_english.json \
    --filler-gap-orc eval_data/filler_gap_orc.csv \
    --filler-gap-wh eval_data/filler_gap_wh.csv \
    --probability-masses-orc eval_data/orc_test.txt \
    --probability-masses-wh eval_data/wh_test.txt \
    --c4 "0.9" \
    --orc "0.0" \
    --wh "0.1" \
    --svo "0.0"