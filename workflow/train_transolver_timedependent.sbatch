#!/bin/bash
# Stellar single-node DDP job: one process per A100.
# Change the account if `sshare` shows a different PPPL project for you.
##SBATCH --account=m5300
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --job-name=transolver-time
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-user=rchurchi@pppl.gov
#SBATCH --mail-type=BEGIN,END,FAIL

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

# Routine experiment settings live in this file rather than being scattered
# through the launch command below.
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/workflow/train_transolver_timedependent.conf}"
if [[ ! -f "${TRAIN_CONFIG}" ]]; then
  echo "Training config does not exist: ${TRAIN_CONFIG}" >&2
  exit 1
fi
source "${TRAIN_CONFIG}"

# torchrun starts one process per GPU allocated by Slurm.
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-1}"

if (( GPUS_PER_NODE <= 0 )); then
  echo "GPUS_PER_NODE must be positive: ${GPUS_PER_NODE}" >&2
  exit 1
fi
if (( BATCH_SIZE_PER_GPU <= 0 )); then
  echo "BATCH_SIZE_PER_GPU must be positive: ${BATCH_SIZE_PER_GPU}" >&2
  exit 1
fi
if (( NUM_WORKERS_PER_GPU < 0 )); then
  echo "NUM_WORKERS_PER_GPU must be nonnegative: ${NUM_WORKERS_PER_GPU}" >&2
  exit 1
fi

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

echo "Training config: ${TRAIN_CONFIG}"
echo "Results root: ${RESULTS_ROOT}"
echo "Save directory: ${SAVE_DIR}"
echo "Launching DDP with ${GPUS_PER_NODE} GPUs, batch_size_per_gpu=${BATCH_SIZE_PER_GPU}, global_batch_size=$((GPUS_PER_NODE * BATCH_SIZE_PER_GPU)), num_workers_per_gpu=${NUM_WORKERS_PER_GPU}"

FOLDER_ARGS=(--train-folders "${TRAIN_FOLDERS}" --val-folders "${VAL_FOLDERS}")

srun --ntasks=1 torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${GPUS_PER_NODE}" \
  -m alpha_analysis.ai.train_transolver_timedependent \
  --results-root "${RESULTS_ROOT}" \
  --save-dir "${SAVE_DIR}" \
  --device cuda \
  --input-frames "${INPUT_FRAMES}" \
  --output-frames "${OUTPUT_FRAMES}" \
  "${FOLDER_ARGS[@]}" \
  --batch-size "${BATCH_SIZE_PER_GPU}" \
  --num-workers "${NUM_WORKERS_PER_GPU}" \
  --max-nodes "${MAX_NODES}" \
  --epochs "${EPOCHS}" \
  "${EXTRA_TRAIN_ARGS[@]}" \
  "$@"
