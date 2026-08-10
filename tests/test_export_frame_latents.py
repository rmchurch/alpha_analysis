import torch

from alpha_analysis.ai.export_frame_latents import _encode_sequence
from alpha_analysis.ai.frame_dataset import FrameNormalization
from alpha_analysis.ai.transolver_autoencoder import (
    AutoencoderConfig,
    TransolverFrameAutoencoder,
)


def test_export_matches_direct_encoder_and_is_repeatable():
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
    sequence = {
        "folder": "synthetic",
        "profiles": torch.randn(3, 7, 2),
        "coordinates": torch.randn(7, 3),
        "bfield": torch.randn(7, 3),
        "times": torch.linspace(0, 1, 3),
        "context": {"R_lmn": torch.randn(5), "Z_lmn": torch.randn(5)},
        "target": torch.tensor(0.2),
        "grid_shape": (7,),
    }
    normalization = FrameNormalization(
        torch.zeros(2),
        torch.ones(2),
        torch.zeros(3),
        torch.ones(3),
        torch.zeros(3),
        torch.ones(3),
    )
    with torch.inference_mode():
        first = _encode_sequence(
            model, sequence, normalization, torch.device("cpu"), None, 0
        )
        second = _encode_sequence(
            model, sequence, normalization, torch.device("cpu"), None, 0
        )
        direct = model.encode(
            normalization.transform_profile(sequence["profiles"][0]).unsqueeze(0),
            sequence["coordinates"].unsqueeze(0),
            sequence["bfield"].unsqueeze(0),
            sequence["times"][0].reshape(1),
            {key: value.unsqueeze(0) for key, value in sequence["context"].items()},
        ).latent[0]
    assert torch.equal(first["latents"], second["latents"])
    assert torch.equal(first["latents"][0], direct)
