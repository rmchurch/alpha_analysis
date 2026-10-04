#!/bin/bash
# Train the latent scalar head on tokens from the isolated split.
#SBATCH -A pppl
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 06:00:00
#SBATCH -J train-latent-p2-10
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -eo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
if [[ -f "${SUBMIT_DIR}/alpha_analysis/alpha_analysis/ai/train_latent_generator.py" ]]; then
  REPO_ROOT="${SUBMIT_DIR}/alpha_analysis"
elif [[ -f "${SUBMIT_DIR}/alpha_analysis/ai/train_latent_generator.py" ]]; then
  REPO_ROOT="${SUBMIT_DIR}"
else
  echo "Could not locate the alpha_analysis repository under ${SUBMIT_DIR}" >&2
  exit 1
fi

TOKEN_DIR="${TOKEN_DIR:?TOKEN_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"

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

srun python -u -m alpha_analysis.ai.train_latent_generator \
  --token-dir "${TOKEN_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --token-kind out_slice_tokens \
  --layers last \
  --scalar-target target \
  --batch-size 16 \
  --scalar-epochs 200 \
  --generator-epochs 300 \
  --log-every 10 \
  "$@"
