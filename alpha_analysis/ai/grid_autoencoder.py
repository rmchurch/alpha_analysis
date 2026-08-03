"""Latent-token autoencoder for high-dimensional ASCOT5 grid fields.

The encoder pools an arbitrary number of grid nodes into a fixed number of
tokens.  The decoder uses the node coordinates as queries, so reconstruction
cost scales as ``number of nodes * number of latent tokens`` rather than
quadratically in the number of nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ResidualMLP(nn.Module):
    """Pre-normalized residual MLP used for per-node feature processing."""

    def __init__(self, width: int, expansion: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * expansion, width),
            nn.Dropout(dropout),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.block(value)


@dataclass
class QuantizerOutput:
    values: Tensor
    indices: Tensor
    loss: Tensor
    perplexity: Tensor


class VectorQuantizer(nn.Module):
    """Straight-through vector quantizer with codebook and commitment losses."""

    def __init__(self, codebook_size: int, latent_dim: int, commitment_cost: float) -> None:
        super().__init__()
        if codebook_size <= 1:
            raise ValueError("codebook_size must be greater than one.")
        if commitment_cost < 0.0:
            raise ValueError("commitment_cost must be non-negative.")
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(codebook_size, latent_dim)
        nn.init.uniform_(
            self.embedding.weight,
            -1.0 / codebook_size,
            1.0 / codebook_size,
        )

    def forward(self, values: Tensor) -> QuantizerOutput:
        flat = values.reshape(-1, values.shape[-1])
        codebook = self.embedding.weight
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            + codebook.square().sum(dim=1).unsqueeze(0)
            - 2.0 * flat @ codebook.transpose(0, 1)
        )
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).reshape_as(values)
        codebook_loss = F.mse_loss(quantized, values.detach())
        commitment_loss = F.mse_loss(values, quantized.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss
        straight_through = values + (quantized - values).detach()

        frequencies = F.one_hot(indices, self.codebook_size).float().mean(dim=0)
        perplexity = torch.exp(-torch.sum(frequencies * torch.log(frequencies + 1.0e-10)))
        return QuantizerOutput(
            values=straight_through,
            indices=indices.reshape(*values.shape[:-1]),
            loss=loss,
            perplexity=perplexity,
        )


class GridLatentAutoencoder(nn.Module):
    """Compress node fields to latent tokens and reconstruct them by coordinate.

    Inputs are normalized physical channels and three normalized coordinates.
    ``encode`` is the stable API for retrieving continuous or quantized latent
    states for downstream representation experiments.
    """

    def __init__(
        self,
        *,
        physical_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_latents: int = 32,
        heads: int = 8,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        latent_mode: str = "continuous",
        codebook_size: int = 256,
        commitment_cost: float = 0.25,
        predict_scalar: bool = False,
    ) -> None:
        super().__init__()
        if physical_dim <= 0 or hidden_dim <= 0 or latent_dim <= 0 or num_latents <= 0:
            raise ValueError("Model dimensions must be positive.")
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads.")
        if latent_mode not in {"continuous", "vq"}:
            raise ValueError("latent_mode must be 'continuous' or 'vq'.")

        self.physical_dim = physical_dim
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.latent_mode = latent_mode
        self.predict_scalar = predict_scalar

        self.input_projection = nn.Linear(3 + physical_dim, hidden_dim)
        self.encoder_blocks = nn.ModuleList(
            ResidualMLP(hidden_dim, mlp_ratio, dropout) for _ in range(encoder_layers)
        )
        self.latent_queries = nn.Parameter(torch.empty(1, num_latents, hidden_dim))
        nn.init.trunc_normal_(self.latent_queries, std=0.02)
        self.pool_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.latent_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim)
        )

        self.quantizer: Optional[VectorQuantizer]
        if latent_mode == "vq":
            self.quantizer = VectorQuantizer(codebook_size, latent_dim, commitment_cost)
        else:
            self.quantizer = None

        self.coordinate_projection = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decode_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.decoder_blocks = nn.ModuleList(
            ResidualMLP(hidden_dim, mlp_ratio, dropout) for _ in range(decoder_layers)
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, physical_dim)
        )
        self.scalar_head = (
            nn.Sequential(
                nn.LayerNorm(num_latents * latent_dim),
                nn.Linear(num_latents * latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            if predict_scalar
            else None
        )

    def encode(
        self,
        coordinates: Tensor,
        physical: Tensor,
        mask: Tensor,
        *,
        quantized: bool = True,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Return ``[batch, num_latents, latent_dim]`` latent tokens."""

        nodes = self.input_projection(torch.cat((coordinates, physical), dim=-1))
        for block in self.encoder_blocks:
            nodes = block(nodes)
        queries = self.latent_queries.expand(nodes.shape[0], -1, -1)
        pooled, _ = self.pool_attention(
            queries,
            nodes,
            nodes,
            key_padding_mask=~mask,
            need_weights=False,
        )
        continuous = self.latent_projection(pooled)
        auxiliary: Dict[str, Tensor] = {"continuous_latents": continuous}
        if self.quantizer is None:
            zero = continuous.new_zeros(())
            auxiliary.update(
                {
                    "vq_loss": zero,
                    "codebook_perplexity": zero,
                    "code_indices": torch.full(
                        continuous.shape[:-1],
                        fill_value=-1,
                        dtype=torch.long,
                        device=continuous.device,
                    ),
                }
            )
            return continuous, auxiliary

        result = self.quantizer(continuous)
        auxiliary.update(
            {
                "vq_loss": result.loss,
                "codebook_perplexity": result.perplexity,
                "code_indices": result.indices,
                "quantized_latents": result.values,
            }
        )
        return (result.values if quantized else continuous), auxiliary

    def decode(self, coordinates: Tensor, latents: Tensor, mask: Tensor) -> Tensor:
        """Reconstruct normalized physical channels at the supplied coordinates."""

        queries = self.coordinate_projection(coordinates)
        latent_hidden = self.latent_to_hidden(latents)
        decoded, _ = self.decode_attention(
            queries, latent_hidden, latent_hidden, need_weights=False
        )
        decoded = decoded + queries
        for block in self.decoder_blocks:
            decoded = block(decoded)
        reconstructed = self.output_projection(decoded)
        return reconstructed * mask.unsqueeze(-1).to(reconstructed.dtype)

    def forward(self, coordinates: Tensor, physical: Tensor, mask: Tensor) -> Dict[str, Tensor]:
        latents, auxiliary = self.encode(coordinates, physical, mask, quantized=True)
        output = dict(auxiliary)
        output["latents"] = latents
        output["latent_token_std"] = auxiliary["continuous_latents"].std(
            dim=1, unbiased=False
        ).mean()
        output["reconstruction"] = self.decode(coordinates, latents, mask)
        if self.scalar_head is not None:
            output["scalar_prediction"] = self.scalar_head(
                latents.reshape(latents.shape[0], -1)
            ).squeeze(-1)
        return output


def masked_reconstruction_metrics(
    prediction: Tensor, target: Tensor, mask: Tensor
) -> Tuple[Tensor, Tensor]:
    """Return MSE and MAE over valid nodes and all physical channels."""

    weights = mask.unsqueeze(-1).to(prediction.dtype)
    count = (weights.sum() * prediction.shape[-1]).clamp_min(1.0)
    error = prediction - target
    return (
        (error.square() * weights).sum() / count,
        (error.abs() * weights).sum() / count,
    )


__all__ = [
    "GridLatentAutoencoder",
    "QuantizerOutput",
    "VectorQuantizer",
    "masked_reconstruction_metrics",
]
