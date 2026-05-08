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
#SBATCH --time=10:00:00
#SBATCH --array=0-1

STRUCTURES=(orc wh)
SENT_FILES=(data/valid_orc7_not_in_train_orc_6_72.txt data/valid_wh5_not_in_wh_7_20.txt)
RESULT_FILES=(np_mass_baseline_2_orc.csv np_mass_baseline_2_wh.csv)

STRUCTURE=${STRUCTURES[$SLURM_ARRAY_TASK_ID]}
SENT_FILE=${SENT_FILES[$SLURM_ARRAY_TASK_ID]}
RESULT_FILE=${RESULT_FILES[$SLURM_ARRAY_TASK_ID]}

python eval_test.py \
    --checkpoint-dir out/pretrain/gpt_baseline_2 \
    --sentences-file $SENT_FILE \
    --structure $STRUCTURE \
    --batch-size 512 \
    --result-name $RESULT_FILE