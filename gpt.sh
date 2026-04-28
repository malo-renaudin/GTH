#!/bin/bash
#SBATCH --job-name=gpt_baseline
#SBATCH --output=outputs/gpt/%A_%a.out
#SBATCH --error=outputs/gpt/%A_%a.err
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

set -euo pipefail

# Ensure log directory exists.
mkdir -p outputs/gpt

# Jean Zay environment setup.
module purge
module load cuda/11.8.0-r465

MINICONDA_ROOT="${MINICONDA_ROOT:-$WORK/miniconda3}"
if [[ ! -f "${MINICONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
	echo "Miniconda not found at ${MINICONDA_ROOT}. Install it first on Jean Zay." >&2
	exit 1
fi

source "${MINICONDA_ROOT}/etc/profile.d/conda.sh"

# Allow overriding the environment at submission time:
# sbatch --export=CONDA_ENV_NAME=myenv,MINICONDA_ROOT=/path/to/miniconda3 gpt.sh
CONDA_ENV_NAME="${CONDA_ENV_NAME:-litgpt_jz}"
conda activate "${CONDA_ENV_NAME}"

# Use the submission directory by default instead of a hardcoded path.
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "Host: $(hostname)"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "Python: $(which python)"
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
python -c "import litgpt; print('litgpt import ok')"

litgpt pretrain --config gpt.yaml
