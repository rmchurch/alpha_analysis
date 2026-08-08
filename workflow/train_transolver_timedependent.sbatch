#!/bin/bash
# Stellar GPU job: one process on one A100.
# Change the account if `sshare` shows a different PPPL project for you.
##SBATCH --account=m5300
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=transolver-time
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail

#write out script contents to slurm file
cat "$0"

# Slurm may execute a staged copy of this script from /var/spool/slurmd.
# Use the original submission directory so outputs stay in the project tree.
# Accept submission from either the workspace root or the repository itself.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
if [[ -f "${SUBMIT_DIR}/alpha_analysis/alpha_analysis/ai/train_transolver_timedependent.py" ]]; then
  REPO_ROOT="${SUBMIT_DIR}/alpha_analysis"
elif [[ -f "${SUBMIT_DIR}/alpha_analysis/ai/train_transolver_timedependent.py" ]]; then
  REPO_ROOT="${SUBMIT_DIR}"
else
  echo "Could not locate the alpha_analysis repository under ${SUBMIT_DIR}" >&2
  exit 1
fi
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/../results/G1600}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/runs/transolver_alpha_timedependent/${SLURM_JOB_ID}}"
ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda3/2025.6}"
ALPHA_ENV="${ALPHA_ENV:-torch-stellar}"
if [[ ! -d "${RESULTS_ROOT}" ]]; then
  echo "Results directory does not exist: ${RESULTS_ROOT}" >&2
  exit 1
fi

mkdir -p "${REPO_ROOT}/workflow/logs" "${SAVE_DIR}"
cd "${REPO_ROOT}"

# The cluster module and Conda initialization read optional interactive-shell
# variables. Keep nounset disabled until that initialization is complete.
# Set ANACONDA_MODULE=none when the module tree is unavailable and use the
# pre-existing environment directly.
if [[ -n "${ANACONDA_MODULE}" && "${ANACONDA_MODULE}" != "none" ]]; then
  module load "${ANACONDA_MODULE}"
fi

# Batch shells do not always initialize `conda activate` automatically. If
# Conda is unavailable, the shared environment's bin directory is sufficient.
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

python - <<'PY'
import torch
import alpha_analysis.ai.train_transolver_timedependent
from models.Transolver_plus import Model

print(f"torch {torch.__version__}")
print(f"cuda_available {torch.cuda.is_available()}")
print(f"cuda_device_count {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"cuda_device_0 {torch.cuda.get_device_name(0)}")
print(f"transolver_model {Model.__name__}")
PY

srun python -u -m alpha_analysis.ai.train_transolver_timedependent \
  --results-root "${RESULTS_ROOT}" \
  --save-dir "${SAVE_DIR}" \
  --device cuda \
  --input-frames 1 \
  --output-frames 1 \
  --batch-size 4 \
  --num-workers 4 \
  --max-nodes 171462 \
  --epochs 400 \
  "$@"
