#!/bin/bash
#SBATCH --account=pppl
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-549%20
#SBATCH --job-name=afsi-initial
#SBATCH --output=workflow/logs/afsi_initial_%A_%a.out
#SBATCH --error=workflow/logs/afsi_initial_%A_%a.err
#SBATCH --requeue

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/gpfs/rmc2/m5300/alpha_analysis}"
RESULTS_ROOT="${RESULTS_ROOT:-/scratch/gpfs/rmc2/m5300/results}"
ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda3/2026.7}"
ALPHA_ENV="${ALPHA_ENV:-alpha_analysis}"

if [[ ! -f "${REPO_ROOT}/workflow/export_afsi_initial_pressure.py" ]]; then
    echo "Could not find the alpha_analysis repository at ${REPO_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${RESULTS_ROOT}" ]]; then
    echo "Could not find the results tree at ${RESULTS_ROOT}" >&2
    exit 1
fi

cd "${REPO_ROOT}"
mkdir -p workflow/logs

module load "${ANACONDA_MODULE}"
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ALPHA_ENV}"

# Keep BLAS/OpenMP libraries inside the CPUs assigned by Slurm. Without these
# limits, DESC/JAX/SciPy can create hundreds of runnable threads per process.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export VECLIB_MAXIMUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MALLOC_ARENA_MAX=4
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
echo "Host: $(hostname); CPUs: ${SLURM_CPUS_PER_TASK}; results: ${RESULTS_ROOT}"
python --version

# The Python driver reads SLURM_ARRAY_TASK_ID/COUNT automatically. With the
# current 1,100 equilibria and 550 shards, every task processes two files.
# Existing afsi_initial.h5 outputs are skipped, so resubmission is safe.
srun --cpu-bind=cores --kill-on-bad-exit=1 \
    python -u workflow/export_afsi_initial_pressure.py \
    --results-root "${RESULTS_ROOT}" \
    --show-field-progress \
    "$@"
