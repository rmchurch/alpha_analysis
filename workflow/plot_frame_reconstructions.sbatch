#!/bin/bash
#SBATCH -A m4505
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 00:30:00
#SBATCH -J plot-frame-reconstruction
#SBATCH -o workflow/logs/plot_frame_reconstruction_%j.log
#SBATCH -e workflow/logs/plot_frame_reconstruction_%j.log
set -euo pipefail

REPO_ROOT=/global/cfs/projectdirs/m5300/rchurchi/alpha_analysis
RESULTS_ROOT=/global/cfs/cdirs/m5300/results/G1600
cd "${REPO_ROOT}"
source "${REPO_ROOT}/modules"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg

srun --ntasks=1 python workflow/plot_frame_reconstructions.py \
  --checkpoint "${REPO_ROOT}/runs/frame_autoencoder/56011733/best_field.pt" \
  --results-root "${RESULTS_ROOT}" \
  --simulation-index 406 \
  --frame-index 5 \
  --device cuda \
  --output "${REPO_ROOT}/runs/frame_autoencoder/56011733/validation_reconstruction.png"
