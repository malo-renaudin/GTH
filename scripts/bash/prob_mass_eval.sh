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
#SBATCH --error=results/logs/error_prob_mass_eval_%A_%a.log
#SBATCH --output=results/logs/output_prob_mass_eval_%A_%a.log

source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate litgpt_jz

case ${SLURM_ARRAY_TASK_ID} in
    0)
        CKPT_DIR="results/orc_augmented_full_datasets"
        ;;
    1)
        CKPT_DIR="results/wh_augmented_full_datasets"
        ;;
    2)
        CKPT_DIR="results/both_augmented_full_datasets"
        ;;
    3)
        CKPT_DIR="results/baseline_full_datasets"
        ;;
esac

echo "Running configuration ${SLURM_ARRAY_TASK_ID}"
echo "Output: ${CKPT_DIR}"

python scripts/eval/prob_mass_over_many_ckpts.py\
  --checkpoint_dir "${CKPT_DIR}"\
  --orc_test datasets/orc_test.txt \
  --wh_test datasets/wh_test.txt \
  --output_csv "${CKPT_DIR}"/prob_mass.csv \
  --subset_size 1 \
  --subset_repeats 3 \
  --amp
