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

## Reconstruction-trained grid latents

`train_grid_autoencoder.py` is an experiment aimed at preventing the latent
collapse observed when Transolver slice tokens are supervised only by the
global fast-ion-loss scalar. It is a separate training path and does not alter
`train_transolver.py` or its checkpoints.

The model uses all profile time channels plus `br`, `bphi`, and `bz` as its
physical fields. A learned-query encoder pools the node set into a fixed
`[num_latents, latent_dim]` representation. A coordinate-query decoder must
reconstruct the normalized physical channels at every sampled node. Attention
therefore costs `O(nodes * num_latents)`, making full or large grids practical.
Normalization statistics are fit on the training split and stored in every
checkpoint. Omitting `--max-nodes` uses the complete grid; setting it randomly
samples the same node count as a memory/performance tradeoff.

Train the default continuous bottleneck:

```bash
python -m alpha_analysis.ai.train_grid_autoencoder \
  --results-root /path/to/G1600 \
  --save-dir runs/grid_autoencoder/continuous \
  --max-nodes 16384 \
  --num-latents 32 \
  --latent-dim 64
```

Use a VQ-VAE-style discrete bottleneck by adding:

```bash
  --latent-mode vq --codebook-size 256 --commitment-cost 0.25
```

The reconstruction objective is the default. To jointly test whether the
representation retains fast-ion-loss information, add (for example)
`--scalar-weight 0.1`. This enables a scalar head and target loading; leaving
the weight at zero permits fully self-supervised training without loss labels.
`best_reconstruction.pt` is selected only by validation reconstruction MSE,
while `last.pt`, periodic checkpoints, `config.json`, and `metrics.jsonl` are
also written.
Metrics include `latent_token_std`, which should be monitored alongside
reconstruction error: a value approaching zero is an early warning that the
learned tokens are collapsing across their token dimension.

Run the NERSC job wrapper with optional CLI overrides:

```bash
sbatch workflow/train_grid_autoencoder.sbatch --latent-mode vq --scalar-weight 0.1
```

Export the fixed-size latent tokens for downstream analysis:

```bash
python -m alpha_analysis.ai.export_grid_latents \
  --checkpoint runs/grid_autoencoder/continuous/best_reconstruction.pt \
  --split val
```

Each output `.pt` contains `latents` with shape
`[num_latents, latent_dim]`, `continuous_latents`, the source folder and grid
metadata. VQ runs additionally contain `code_indices` and codebook perplexity.
The output directory includes `manifest.json`. In Python, the same stable API
is `model.encode(coordinates, normalized_physical, mask)`; reconstruct at any
coordinate set with `model.decode(coordinates, latents, mask)`. Load the model
configuration and physical-channel normalizer from the checkpoint before
calling those methods.
