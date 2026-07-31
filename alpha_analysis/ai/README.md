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
