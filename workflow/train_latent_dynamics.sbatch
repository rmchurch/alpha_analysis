#!/bin/bash
#SBATCH -A m4505
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -J latent-dynamics
#SBATCH -o workflow/logs/latent_dynamics_%j.log
#SBATCH -e workflow/logs/latent_dynamics_%j.log
set -euo pipefail
: "${LATENT_DIR:?Set LATENT_DIR to an export directory}"
REPO_ROOT=/global/cfs/projectdirs/m5300/rchurchi/alpha_analysis
cd "${REPO_ROOT}"
source "${REPO_ROOT}/modules"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
srun python -u -m alpha_analysis.ai.train_latent_dynamics \
  --latent-dir "${LATENT_DIR}" \
  --save-dir "${SAVE_DIR:-${LATENT_DIR}/dynamics_${SLURM_JOB_ID}}" \
  --stage "${STAGE:-one-step}" --device cuda "$@"
