#!/bin/bash
# Stellar GPU job: train a fresh static Transolver on the isolated
# Pool-2-augmented split and write it to a new run namespace.
#SBATCH --account=pppl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=train-transolver-p2-10
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail

# Record the exact staged script in the job output.
cat "$0"

# Slurm executes a staged copy of the script, so locate the repository from
# the original submission directory. Submission is supported from either the
# workspace root or the repository itself.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
if [[ -f "${SUBMIT_DIR}/alpha_analysis/alpha_analysis/ai/train_transolver.py" ]]; then
  REPO_ROOT="${SUBMIT_DIR}/alpha_analysis"
elif [[ -f "${SUBMIT_DIR}/alpha_analysis/ai/train_transolver.py" ]]; then
  REPO_ROOT="${SUBMIT_DIR}"
else
  echo "Could not locate the alpha_analysis repository under ${SUBMIT_DIR}" >&2
  exit 1
fi

RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/../results/G1600}"
SPLIT_ROOT="${SPLIT_ROOT:-${REPO_ROOT}/runs/frame_autoencoder/56011733_pool2_10}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/runs/transolver_alpha_pool2_10/${SLURM_JOB_ID}}"

if [[ ! -d "${RESULTS_ROOT}" ]]; then
  echo "Results directory does not exist: ${RESULTS_ROOT}" >&2
  exit 1
fi
for manifest in train_folders.txt val_folders.txt; do
  if [[ ! -f "${SPLIT_ROOT}/${manifest}" ]]; then
    echo "Split manifest does not exist: ${SPLIT_ROOT}/${manifest}" >&2
    exit 1
  fi
done

mkdir -p "${REPO_ROOT}/workflow/logs" "${SAVE_DIR}"
cd "${REPO_ROOT}"

ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda3/2025.6}"
ALPHA_ENV="${ALPHA_ENV:-torch-stellar}"
if [[ -n "${ANACONDA_MODULE}" && "${ANACONDA_MODULE}" != "none" ]]; then
  module load "${ANACONDA_MODULE}"
fi
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ALPHA_ENV}"
else
  ENV_PREFIX="/scratch/gpfs/rmc2/stellar-conda-envs/${ALPHA_ENV}"
  if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    echo "Could not find Python environment ${ENV_PREFIX}" >&2
    exit 1
  fi
  export PATH="${ENV_PREFIX}/bin:${PATH}"
fi
set -u

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

python - <<'PY'
import torch
from models.Transolver_plus import Model

print(f"torch {torch.__version__}")
print(f"cuda_available {torch.cuda.is_available()}")
print(f"cuda_device_count {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"cuda_device_0 {torch.cuda.get_device_name(0)}")
print(f"transolver_model {Model.__name__}")
PY

srun --ntasks=1 python -u -m alpha_analysis.ai.train_transolver \
  --results-root "${RESULTS_ROOT}" \
  --train-folders "${SPLIT_ROOT}/train_folders.txt" \
  --val-folders "${SPLIT_ROOT}/val_folders.txt" \
  --save-dir "${SAVE_DIR}" \
  --device cuda \
  --epochs 400 \
  --batch-size 1 \
  --num-workers 0 \
  --max-nodes 16384 \
  --target-reduction mean \
  "$@"
