# Perlmutter handoff

This inventory describes the Stellar workspace at `/scratch/gpfs/rmc2/m5300`.
The code branch is `ai_phase1_timedependent`. Perlmutter's current files and
environments have not been inspected; compare them against this inventory.

## Code and exact experiment splits

Pull the branch in your Perlmutter checkout:

```bash
git fetch origin
git switch ai_phase1_timedependent
git pull --ff-only origin ai_phase1_timedependent
mkdir -p runs/frame_autoencoder workflow/logs
cp -R workflow/splits/* runs/frame_autoencoder/
```

The saved split snapshots restore the paths expected by the experiment launchers.
Manifest entries use `/global/cfs/cdirs/m5300/results/G1600`; readers can resolve
the `projectdirs` alias or sample names beneath an explicit results root.

| Split directory | Train | Validation | Later/unseen |
| --- | ---: | ---: | ---: |
| `56011733` | 462 | 116 | 522 |
| `56011733_pool2_10` | 472 | 116 | 512 |
| `56011733_pool2_50` | 512 | 116 | 472 |
| `56011733_all_random_seed0` | 880 | 220 | — |

Preserve these exact splits. The all-random experiment evaluates the validation
manifest using `--evaluation-folders`; it does not have a separate unseen set.

## Data and artifacts outside Git

The local `results/G1600` has 1,506 sample directories and AFSI files, but only
1,100 folders have each of `analysis_results.h5`, `desc_equilibrium.h5`, and
`bfield.h5`. AFSI presence alone does not make a sample usable for training.
Compare sample names on Perlmutter and transfer any missing inputs. Preserve
`G1600_end_database.json` (about 708 MiB), which supplies static loss targets.

Copy the ignored `alpha_analysis/runs/` tree to the Perlmutter checkout to keep
checkpoints, configs, normalization statistics, static slice tokens, scalar
heads, evaluation outputs, and plots. It is about 39 GiB locally, including
about 34 GiB of synthetic evaluation outputs. Copying the model run directories
and split manifests alone is sufficient if evaluation outputs will be rerun.

| Experiment | Static checkpoint run | AFSI temporal checkpoint run |
| --- | --- | --- |
| Baseline | `transolver_alpha/53562942` | `transolver_alpha_timedependent/2916088` |
| Pool 2 +10 | `transolver_alpha_pool2_10/2920488` | `transolver_alpha_timedependent_pool2_10/2919732` |
| Pool 2 +50 | `transolver_alpha_pool2_50/2921133` | `transolver_alpha_timedependent_pool2_50/2921134` |
| All random | `transolver_alpha_all_random_seed0/2920880` | `transolver_alpha_timedependent_all_random_seed0/2920881` |

These paths are relative to `runs/`. Scalar heads are inside the static run
directories (`scalar_head_best.pt`); preserve their parent directories too.

Saved configs contain Stellar absolute paths. Use explicit `--results-root`,
run/checkpoint paths, and manifest paths on Perlmutter. The rollout plotter
reads `train_folders`/`val_folders` from the saved config: update those copied
config fields to the restored Perlmutter manifests before plotting. Token
exports and other metadata may also retain old paths; check before reuse.
Current temporal checkpoints require `afsi_initial.h5` conditioning.

## AFSI: transfer or recompute

An existing `/scratch/gpfs/rmc2/m5300/afsi_initial_G1600.tar` is about 434 MiB and
contains `<sample>/afsi_initial.h5`. Copy it to Perlmutter and inspect its file
list/coverage before extracting beneath the target `results/G1600` directory;
its completeness and freshness have not been established. Alternatively copy
the current individual AFSI files directly. This avoids rerunning AFSI.

If recomputation is needed, match the configuration recorded in the local
`G1600_00000/afsi_initial.h5` (the Python defaults are different):

```bash
python workflow/export_afsi_initial_pressure.py \
  --results-root /global/cfs/cdirs/m5300/results/G1600 \
  --replace-incompatible \
  --nrho-bins 100 --nenergy-bins 49 --npitch-bins 1 \
  --nmc 100000 --nthermal-vel 10 \
  --field-nr 100 --field-nz 100 --field-nphi 100 \
  --profile-nrho 1024 --l-radial 4 --m-poloidal 4 \
  --fraction-tritium 0.5 --zeff 1.0
```

Run in a CPU allocation with the ASCOT/DESC environment. Set `JAX_PLATFORMS=cpu`
and bound BLAS/OpenMP threads to allocated CPUs, as in the existing wrappers.
Use `--limit 1` for a first sample, then array sharding or `--mpi` for throughput.
AFSI is Monte Carlo, so recalculation need not reproduce identical numbers.
The stored moments are source rates (particles/s, Pa/s), with no accumulation
duration; keep this interpretation when comparing outputs.

## Environments and dependencies

Rebuild environments on Perlmutter rather than copying Stellar environment
directories or shared libraries. The repository requires Python >=3.12. Local
training uses `torch-stellar`; AFSI uses a separate `alpha_analysis` environment.

| Local environment | Observed package versions |
| --- | --- |
| `torch-stellar` | torch 2.4.1, torchvision 0.19.1, numpy 1.26.4, h5py 3.12.1, einops 0.8.2, timm 1.0.28 |
| `alpha_analysis` | jax/jaxlib 0.9.2, desc-opt 0.17.3, numpy 2.4.6, h5py 3.16.0, mpi4py 4.1.1 |

`pip install -e .` does not provide PyTorch, Transolver++, or compiled ASCOT.
Use the repository installation helpers and a compatible CUDA PyTorch build.
Local external-source revisions are:

- ASCOT5: `d884bac99b4d3a6e0fd18c9c74b600a8a9553f72`.
- Transolver++: `d5a23bc734a0ebac56384cf72049a26af9673452`.

Both source trees live in ignored `no_sync/`. Pin matching revisions if exact
behavior matters. The Transolver installer follows upstream `main` by default
and defaults to torchvision 0.25.0 if it needs to install torchvision; use a
compatible torch/torchvision pair instead of mixing it with local torch 2.4.1.

ASCOT has a local NumPy scalar conversion fix, preserved in
`tools/patches/ascot5-plasma-numpy-scalar.patch`. Apply it to the matching source:

```bash
git -C no_sync/ascot5-src apply --unidiff-zero --check ../../tools/patches/ascot5-plasma-numpy-scalar.patch
git -C no_sync/ascot5-src apply --unidiff-zero ../../tools/patches/ascot5-plasma-numpy-scalar.patch
```

The local `a5py/ascotpy/ascot2py.py` also differs because bindings were generated
against the local library; regenerate bindings after compiling libascot on
Perlmutter. The ASCOT helper skips installation if `a5py` already imports, so
an existing installation still needs an explicit library/bindings check.
Adapt `modules` and `LD_LIBRARY_PATH` to the actual checkout/library location.

## Batch launcher changes

The temporal wrapper and AFSI wrappers currently use Stellar partition/GRES,
memory, module, and environment settings. The new static/token/scalar wrappers
mix NERSC-style constraints/QOS with a PPPL account and Stellar environment
fallback paths. Review all headers before submitting.

On Perlmutter choose your valid NERSC account (the existing project is m5300),
`--constraint=gpu` for training or `--constraint=cpu` for AFSI, an appropriate
QOS/time limit, and the GPU count. Keep four GPUs for the temporal run's current
batch size of 2 per GPU (global batch size 8). Replace Stellar's
`anaconda3/2025.6` or `anaconda3/2026.7` module setup with Perlmutter's Python
module/environment initialization. `RESULTS_ROOT=${REPO_ROOT}/../results/G1600`
is incorrect if the checkout is under `m5300/rchurchi/alpha_analysis` and data
remain under `m5300/results/G1600`; set the data root explicitly in the scripts
or temporal config.

For MPI AFSI, replace `openmpi/gcc/4.1.6` and the hard-coded `--mpi=pmix_v3`
launch settings with the supported Perlmutter MPI setup and build/install
mpi4py against that runtime. Retune rank count/memory on a CPU node; the
Stellar wrapper's 700 GiB memory request is not a portable resource choice.

Before long jobs, verify CUDA and the Transolver import in a GPU allocation,
run a small static `--dry-run`, calculate/check one AFSI sample, then exercise
temporal training and evaluation with the restored manifests. If models and
tokens are not copied, the dependency order is static training → token export
→ scalar-head training, alongside AFSI → temporal training → evaluation.

NERSC references:
[job configuration](https://docs.nersc.gov/systems/perlmutter/running-jobs/),
[Python and MPI](https://docs.nersc.gov/development/languages/python/using-python-perlmutter/).
