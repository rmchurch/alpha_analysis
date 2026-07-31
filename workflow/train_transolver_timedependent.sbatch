#!/bin/bash
#SBATCH -A m4505
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 04:00:00
#SBATCH -J transolver-time
#SBATCH -o /global/cfs/projectdirs/m5300/rchurchi/alpha_analysis/workflow/logs/train_transolver_timedependent_%j.log
#SBATCH -e /global/cfs/projectdirs/m5300/rchurchi/alpha_analysis/workflow/logs/train_transolver_timedependent_%j.log

set -euo pipefail

REPO_ROOT=/global/cfs/projectdirs/m5300/rchurchi/alpha_analysis
RESULTS_ROOT=/global/cfs/cdirs/m5300/results/G1600
SAVE_DIR="${REPO_ROOT}/runs/transolver_alpha_timedependent/${SLURM_JOB_ID}"

mkdir -p "${REPO_ROOT}/workflow/logs" "${SAVE_DIR}"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/modules"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

srun python -u -m alpha_analysis.ai.train_transolver_timedependent \
  --results-root "${RESULTS_ROOT}" \
  --save-dir "${SAVE_DIR}" \
  --device cuda \
  --input-frames 1 \
  --output-frames 1 \
  --epochs 400 \
  "$@"
