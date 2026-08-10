# AI for Ascot5

This repository is the starting point for the AI for Ascot5 (https://github.com/ascot4fusion/ascot5) project. This will serve as the central place for project code, notes, and supporting documentation as the work takes shape.

# Current status
- Data loaders for generated dataset
- Helper for sampling `br`, `bphi`, and `bz` from `ascot_results.h5` onto the
  `analysis_results.h5/profiles` grid
- Temporal Transolver training for profile-field forecasting. The default is
  one frame in and the next frame out:

  ```bash
  python -m alpha_analysis.ai.train_transolver_timedependent \
    --results-root /path/to/G1600 \
    --input-frames 1 \
    --output-frames 1
  ```

  Wider input/output windows use the same path (for example,
  `--input-frames 3 --output-frames 2`). Training/validation are split by
  simulation before temporal windows are constructed. Checkpoints can be used
  for chunked autoregressive rollout with
  `workflow/predict_transolver_timedependent.py`.

  Single-node multi-GPU training uses PyTorch DDP with one process per GPU:

  ```bash
  torchrun --standalone --nproc_per_node=2 \
    -m alpha_analysis.ai.train_transolver_timedependent \
    --results-root /path/to/G1600 \
    --device cuda \
    --batch-size 2
  ```

  `--batch-size` is per GPU, so the global batch size in this example is four.
  For Slurm jobs, edit
  `workflow/train_transolver_timedependent.conf` to set the batch size, worker
  count, save directory, data path, frame counts, and other common training
  options. Then submit `workflow/train_transolver_timedependent.sbatch`; the
  wrapper automatically starts one process for every GPU in its Slurm request.
