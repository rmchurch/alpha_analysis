#!/bin/bash
# Export static-Transolver tokens for the all-random train/val split.
#SBATCH -A pppl
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 01:00:00
#SBATCH -J export-transolver-all-random
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -eo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
if [[ -f "${SUBMIT_DIR}/alpha_analysis/alpha_analysis/ai/export_transolver_slice_tokens.py" ]]; then
  REPO_ROOT="${SUBMIT_DIR}/alpha_analysis"
elif [[ -f "${SUBMIT_DIR}/alpha_analysis/ai/export_transolver_slice_tokens.py" ]]; then
  REPO_ROOT="${SUBMIT_DIR}"
else
  echo "Could not locate the alpha_analysis repository under ${SUBMIT_DIR}" >&2
  exit 1
fi

STATIC_JOB_ID="${STATIC_JOB_ID:?STATIC_JOB_ID is required}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/transolver_alpha_all_random_seed0/${STATIC_JOB_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/best_slice_tokens}"
SPLIT_ROOT="${REPO_ROOT}/runs/frame_autoencoder/56011733_all_random_seed0"
RESULTS_ROOT="${REPO_ROOT}/../results/G1600"

mkdir -p "${REPO_ROOT}/workflow/logs" "${OUTPUT_DIR}"
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

srun python -u -m alpha_analysis.ai.export_transolver_slice_tokens \
  --run-dir "${RUN_DIR}" \
  --checkpoint best.pt \
  --results-root "${RESULTS_ROOT}" \
  --train-folders "${SPLIT_ROOT}/train_folders.txt" \
  --val-folders "${SPLIT_ROOT}/val_folders.txt" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --token-dtype float32 \
  --log-every 25 \
  "$@"
