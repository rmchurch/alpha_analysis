import torch
from torch.utils.data import Dataset

from alpha_analysis.ai.grid_autoencoder import (
    GridLatentAutoencoder,
    masked_reconstruction_metrics,
)
from alpha_analysis.ai.train_grid_autoencoder import (
    fit_feature_normalizer,
    sample_to_grid_tensors,
)


def _sample(offset: float = 0.0):
    grid = (2, 3, 4)
    values = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).reshape(2, *grid)
    coordinates = torch.meshgrid(
        torch.linspace(0.0, 1.0, grid[0]),
        torch.linspace(-2.0, 2.0, grid[1]),
        torch.linspace(4.0, 8.0, grid[2]),
        indexing="ij",
    )
    return {
        "folder": f"sample-{offset}",
        "prs_para": values + offset,
        "prs_perp": values + 100.0 + offset,
        "target": torch.tensor([0.25 + offset]),
        "bfield": {
            "br": torch.ones(grid) + offset,
            "bphi": torch.full(grid, 2.0 + offset),
            "bz": torch.full(grid, 3.0 + offset),
            "rho": coordinates[0],
            "theta": coordinates[1],
            "phi": coordinates[2],
        },
    }


class _Samples(Dataset):
    def __init__(self):
        self.samples = [_sample(0.0), _sample(1.0)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_sample_conversion_preserves_full_grid_and_channel_layout():
    coordinates, physical, target, names = sample_to_grid_tensors(
        _sample(),
        max_nodes=None,
        profile_log1p=False,
        target_reduction="mean",
        generator=torch.Generator().manual_seed(0),
    )
    assert coordinates.shape == (24, 3)
    assert physical.shape == (24, 7)  # two frames per profile plus B_r/B_phi/B_z
    assert names == [
        "prs_para_0",
        "prs_para_1",
        "prs_perp_0",
        "prs_perp_1",
        "br",
        "bphi",
        "bz",
    ]
    assert torch.all(coordinates >= -1.0) and torch.all(coordinates <= 1.0)
    torch.testing.assert_close(target, torch.tensor(0.25))


def test_training_normalizer_produces_finite_unit_scale_channels():
    normalizer, names = fit_feature_normalizer(
        _Samples(),
        max_nodes=None,
        profile_log1p=False,
        target_reduction="mean",
        seed=0,
    )
    _, physical, _, _ = sample_to_grid_tensors(
        _sample(),
        max_nodes=None,
        profile_log1p=False,
        target_reduction="mean",
        generator=torch.Generator().manual_seed(0),
    )
    normalized = normalizer.normalize(physical)
    assert len(names) == 7
    assert torch.isfinite(normalized).all()
    torch.testing.assert_close(normalizer.denormalize(normalized), physical)


def test_continuous_autoencoder_exposes_fixed_size_latents_and_scalar_head():
    model = GridLatentAutoencoder(
        physical_dim=7,
        hidden_dim=16,
        latent_dim=6,
        num_latents=4,
        heads=4,
        encoder_layers=1,
        decoder_layers=1,
        dropout=0.0,
        predict_scalar=True,
    )
    coordinates = torch.randn(2, 9, 3)
    physical = torch.randn(2, 9, 7)
    mask = torch.tensor([[True] * 9, [True] * 6 + [False] * 3])
    output = model(coordinates, physical, mask)
    assert output["latents"].shape == (2, 4, 6)
    assert output["reconstruction"].shape == physical.shape
    assert output["scalar_prediction"].shape == (2,)
    assert torch.count_nonzero(output["reconstruction"][1, 6:]) == 0


def test_vq_autoencoder_returns_codes_and_differentiable_loss():
    model = GridLatentAutoencoder(
        physical_dim=5,
        hidden_dim=12,
        latent_dim=4,
        num_latents=3,
        heads=3,
        encoder_layers=1,
        decoder_layers=1,
        dropout=0.0,
        latent_mode="vq",
        codebook_size=8,
    )
    coordinates = torch.randn(2, 7, 3)
    physical = torch.randn(2, 7, 5)
    mask = torch.ones(2, 7, dtype=torch.bool)
    output = model(coordinates, physical, mask)
    reconstruction_mse, _ = masked_reconstruction_metrics(
        output["reconstruction"], physical, mask
    )
    loss = reconstruction_mse + output["vq_loss"]
    loss.backward()
    assert output["code_indices"].shape == (2, 3)
    assert output["code_indices"].min() >= 0
    assert output["code_indices"].max() < 8
    assert output["codebook_perplexity"] >= 1.0
    assert model.input_projection.weight.grad is not None
