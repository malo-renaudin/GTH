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
#SBATCH --error=results/baseline_no_augmentation/error_train_streaming%a.log
#SBATCH --output=results/baseline_no_augmentation/output_train_streaming_%a.log

source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate litgpt_jz

python scripts/train/streaming_dataset.py \
   --output-dir results/baseline_no_augmentation\
   --probability-masses-orc eval_data/orc_test.txt \
   --probability-masses-wh  eval_data/wh_test.txt \
   --blimp-dir eval_data/blimp_data \
   --nested-inner eval_data/short_nested_inner_english.json \
   --nested-outer eval_data/short_nested_outer_english.json \
   --filler-gap-orc eval_data/filler_gap_orc.csv \
   --filler-gap-wh eval_data/filler_gap_wh.csv \
   --c4 1 \
   --orc 0 \
   --wh 0 \
   --svo_wh 0 \
   --svo_orc 0 \