#!/bin/bash
#SBATCH -A m4505
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
#SBATCH -J frame-latents
#SBATCH -o workflow/logs/frame_latents_%j.log
#SBATCH -e workflow/logs/frame_latents_%j.log
set -euo pipefail
: "${CHECKPOINT:?Set CHECKPOINT to a frame-autoencoder checkpoint}"
REPO_ROOT=/global/cfs/projectdirs/m5300/rchurchi/alpha_analysis
cd "${REPO_ROOT}"
source "${REPO_ROOT}/modules"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
srun python -u -m alpha_analysis.ai.export_frame_latents \
  --checkpoint "${CHECKPOINT}" \
  --results-root /global/cfs/cdirs/m5300/results/G1600 \
  --output-dir "${OUTPUT_DIR:-${CHECKPOINT%/*}/frame_latents}" --device cuda "$@"
