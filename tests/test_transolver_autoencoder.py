import torch

from alpha_analysis.ai.transolver_autoencoder import (
    AutoencoderConfig,
    SliceBottleneck,
    TransolverFrameAutoencoder,
    autoencoder_loss,
)


def test_bottleneck_mask_shape_and_no_deslice():
    module = SliceBottleneck(16, 4, 8, 4)
    nodes = torch.randn(2, 7, 16, requires_grad=True)
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool
    )
    latent, _, weights, norms = module(nodes, mask)
    assert latent.shape == (2, 4, 8)
    assert weights[0, :, 5:].count_nonzero() == 0
    assert torch.allclose(weights.sum(-1)[0, :, :5], torch.ones(4, 5))
    latent.sum().backward()
    assert nodes.grad is not None


def test_autoencoder_gradients_and_external_latent_decoder():
    config = AutoencoderConfig(
        hidden_dim=16,
        encoder_layers=1,
        encoder_heads=4,
        encoder_slice_num=4,
        latent_tokens=4,
        latent_dim=8,
        decoder_hidden_dim=16,
        decoder_heads=4,
        decoder_layers=1,
    )
    model = TransolverFrameAutoencoder(config)
    profile = torch.randn(2, 9, 2)
    coordinates = torch.randn(2, 9, 3)
    field = torch.randn(2, 9, 3)
    time = torch.tensor([0.0, 0.5])
    mask = torch.ones(2, 9, dtype=torch.bool)
    encoded, reconstructed = model(profile, coordinates, field, time, node_mask=mask)
    loss, _ = autoencoder_loss(reconstructed, profile, mask)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())
    external = model.decode(
        torch.randn_like(encoded.latent), coordinates, field, time, node_mask=mask
    )
    assert external.shape == profile.shape


def test_evaluation_repeatable():
    model = TransolverFrameAutoencoder(
        AutoencoderConfig(
            hidden_dim=16,
            encoder_layers=0,
            encoder_heads=4,
            encoder_slice_num=4,
            latent_tokens=4,
            latent_dim=8,
            decoder_hidden_dim=16,
            decoder_heads=4,
            decoder_layers=0,
        )
    ).eval()
    args = (
        torch.randn(1, 6, 2),
        torch.randn(1, 6, 3),
        torch.randn(1, 6, 3),
        torch.tensor([0.2]),
    )
    with torch.no_grad():
        first = model(*args)[1]
        second = model(*args)[1]
    assert torch.equal(first, second)
