#!/bin/bash
#SBATCH -A m4505
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH -t 04:00:00
#SBATCH -J frame-autoencoder
#SBATCH -o workflow/logs/frame_autoencoder_%j.log
#SBATCH -e workflow/logs/frame_autoencoder_%j.log
set -euo pipefail
REPO_ROOT=/global/cfs/projectdirs/m5300/rchurchi/alpha_analysis
RESULTS_ROOT=/global/cfs/cdirs/m5300/results/G1600
cd "${REPO_ROOT}"
source "${REPO_ROOT}/modules"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
NUM_WORKERS="${NUM_WORKERS:-32}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=none \
  torchrun --standalone --nnodes=1 --nproc-per-node="${NPROC_PER_NODE}" \
  -m alpha_analysis.ai.train_frame_autoencoder \
  --results-root "${RESULTS_ROOT}" \
  --save-dir "${REPO_ROOT}/runs/frame_autoencoder/${SLURM_JOB_ID}" \
  --training-sample-count 578 --device cuda \
  --num-workers "${NUM_WORKERS}" "$@"
