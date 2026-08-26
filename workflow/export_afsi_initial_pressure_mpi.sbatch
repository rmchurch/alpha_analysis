#!/bin/bash
#SBATCH --account=pppl
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=4
#SBATCH --mem=700G
#SBATCH --time=08:00:00
#SBATCH --job-name=afsi-initial-mpi
#SBATCH --output=workflow/logs/afsi_initial_mpi_%j.out
#SBATCH --error=workflow/logs/afsi_initial_mpi_%j.err
#SBATCH --requeue

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/gpfs/rmc2/m5300/alpha_analysis}"
RESULTS_ROOT="${RESULTS_ROOT:-/scratch/gpfs/rmc2/m5300/results}"
MPI_MODULE="${MPI_MODULE:-openmpi/gcc/4.1.6}"
ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda3/2026.7}"
ALPHA_ENV="${ALPHA_ENV:-alpha_analysis}"

cd "${REPO_ROOT}"
mkdir -p workflow/logs

# mpi4py in alpha_analysis was compiled against this Open MPI module.
module load "${MPI_MODULE}"
module load "${ANACONDA_MODULE}"
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ALPHA_ENV}"

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

echo "Job ${SLURM_JOB_ID}: ${SLURM_NNODES} nodes, ${SLURM_NTASKS} MPI ranks"
echo "Rank 0 is the dispatcher; $((SLURM_NTASKS - 1)) ranks run AFSI work."
python - <<'PY'
from mpi4py import MPI
print(MPI.Get_library_version().splitlines()[0])
PY

srun --mpi=pmix_v3 --cpu-bind=cores --kill-on-bad-exit=1 \
    python -u workflow/export_afsi_initial_pressure.py \
    --mpi \
    --results-root "${RESULTS_ROOT}" \
    --nrho-bins 100 \
    --nenergy-bins 49 \
    --npitch-bins 1 \
    --nmc 100000 \
    --field-nr 100 \
    --field-nz 100 \
    --field-nphi 100 \
    --l-radial 4 \
    --m-poloidal 4 \
    "$@"
