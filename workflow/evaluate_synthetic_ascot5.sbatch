#!/bin/bash
#SBATCH --account=pppl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=synthetic-ascot5
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-user=rchurchi@pppl.gov
#SBATCH --mail-type=BEGIN,END,FAIL

set -eo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
if [[ -f "${SUBMIT_DIR}/alpha_analysis/workflow/evaluate_synthetic_ascot5.py" ]]; then
  REPO_ROOT="${SUBMIT_DIR}/alpha_analysis"
elif [[ -f "${SUBMIT_DIR}/workflow/evaluate_synthetic_ascot5.py" ]]; then
  REPO_ROOT="${SUBMIT_DIR}"
else
  echo "Could not locate the alpha_analysis repository under ${SUBMIT_DIR}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/workflow/logs"

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

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/../results/G1600}"
srun python -u workflow/evaluate_synthetic_ascot5.py \
  --device cuda \
  --results-root "${RESULTS_ROOT}" \
  "$@"
