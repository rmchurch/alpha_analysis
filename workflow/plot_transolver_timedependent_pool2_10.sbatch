#!/bin/bash
# Generate the four-column rho pressure rollout plot for the Pool-2-10 run.
#SBATCH --account=pppl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=plot-pressure-p2
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

REPO_ROOT=/scratch/gpfs/rmc2/m5300/alpha_analysis
PYTHON=/scratch/gpfs/rmc2/stellar-conda-envs/torch-stellar/bin/python
RUN_DIR=${REPO_ROOT}/runs/transolver_alpha_timedependent_pool2_10/2919732

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" workflow/plot_transolver_timedependent_rollouts.py \
  --run-dir "${RUN_DIR}" \
  --checkpoint best.pt \
  --results-root "${REPO_ROOT}/results/G1600" \
  --indices 0,1,2,3 \
  --total-frames 10 \
  --representation rho \
  --device cuda \
  --output "${RUN_DIR}/pool2_10_best_timedependent_rollouts_rho.png"
