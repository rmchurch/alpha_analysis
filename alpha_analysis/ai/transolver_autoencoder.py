"""Tracked deterministic Transolver-style per-frame autoencoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _masked_softmax(logits: Tensor, mask: Tensor | None) -> Tensor:
    if mask is not None:
        logits = logits.masked_fill(
            ~mask[:, None, :, None], torch.finfo(logits.dtype).min
        )
    weights = logits.softmax(dim=-1)
    if mask is not None:
        weights = weights * mask[:, None, :, None]
    return weights


class SliceBottleneck(nn.Module):
    """Aggregate nodes into stable latent slots without a deslice operation."""

    def __init__(
        self,
        hidden_dim: int = 128,
        latent_tokens: int = 16,
        latent_dim: int = 64,
        heads: int = 4,
        token_blocks: int = 1,
        dropout: float = 0.0,
        temperature_min: float = 0.01,
    ) -> None:
        super().__init__()
        if hidden_dim % heads or latent_dim % heads:
            raise ValueError("hidden_dim and latent_dim must be divisible by heads")
        self.heads, self.latent_tokens = heads, latent_tokens
        self.head_dim = hidden_dim // heads
        self.temperature_min = temperature_min
        self.node_projection = nn.Linear(hidden_dim, hidden_dim)
        self.slice_projection = nn.Linear(self.head_dim, latent_tokens)
        nn.init.orthogonal_(self.slice_projection.weight)
        self.raw_temperature = nn.Parameter(torch.zeros(1, heads, 1, 1))
        self.slot_embeddings = nn.Parameter(
            torch.randn(1, heads, latent_tokens, self.head_dim) * 0.02
        )
        merged_dim = heads * self.head_dim
        self.to_latent = nn.Linear(merged_dim, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.token_blocks = nn.TransformerEncoder(encoder_layer, token_blocks)

    def forward(
        self, nodes: Tensor, node_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, num_nodes, _ = nodes.shape
        if node_mask is not None and (
            node_mask.shape != (batch, num_nodes) or not node_mask.any(1).all()
        ):
            raise ValueError(
                "node_mask must be [B,N] with at least one valid node per sample"
            )
        projected = (
            self.node_projection(nodes)
            .reshape(batch, num_nodes, self.heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        temperature = F.softplus(self.raw_temperature) + self.temperature_min
        weights = _masked_softmax(
            self.slice_projection(projected) / temperature, node_mask
        )
        occupancy = weights.sum(dim=2)
        tokens = torch.einsum("bhng,bhnd->bhgd", weights, projected)
        tokens = tokens / occupancy.clamp_min(1e-6).unsqueeze(-1)
        pre_attention = tokens + self.slot_embeddings
        merged = pre_attention.permute(0, 2, 1, 3).reshape(
            batch, self.latent_tokens, -1
        )
        latent = self.token_blocks(self.to_latent(merged))
        return latent, merged, weights, occupancy


class MeshTransolverBlock(nn.Module):
    """Deterministic mesh-to-mesh slicing block with mask support."""

    def __init__(
        self, hidden_dim: int, heads: int, slice_num: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.slicer = SliceBottleneck(
            hidden_dim, slice_num, hidden_dim, heads, 1, dropout
        )
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, nodes: Tensor, node_mask: Tensor | None = None) -> Tensor:
        latent, _, weights, _ = self.slicer(self.norm1(nodes), node_mask)
        # Average head assignments for a stable latent-to-node deslice.
        assignment = weights.mean(1)
        update = torch.einsum("bng,bgd->bnd", assignment, latent)
        nodes = nodes + self.out(update)
        nodes = nodes + self.mlp(self.norm2(nodes))
        return nodes if node_mask is None else nodes * node_mask.unsqueeze(-1)


def _context_summary(
    context: Mapping[str, Tensor] | None, batch: int, device: torch.device
) -> Tensor:
    if not context:
        return torch.zeros(batch, 4, device=device)
    pieces = []
    for key in ("R_lmn", "Z_lmn"):
        if key not in context:
            pieces.extend(
                [torch.zeros(batch, device=device), torch.zeros(batch, device=device)]
            )
            continue
        value = context[key].to(device).float()
        if value.ndim == 1:
            value = value.unsqueeze(0)
        flat = value.reshape(value.shape[0], -1)
        pieces.extend((flat.mean(1), flat.std(1, unbiased=False)))
    return torch.stack(pieces, dim=-1)


@dataclass
class EncoderOutput:
    latent: Tensor
    pre_attention_tokens: Tensor
    slice_weights: Tensor
    slice_norms: Tensor


@dataclass
class AutoencoderConfig:
    hidden_dim: int = 128
    encoder_layers: int = 2
    encoder_heads: int = 4
    encoder_slice_num: int = 32
    latent_tokens: int = 16
    latent_dim: int = 64
    bottleneck_blocks: int = 1
    decoder_hidden_dim: int = 128
    decoder_heads: int = 4
    decoder_layers: int = 1
    dropout: float = 0.0


class FrameEncoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.context = nn.Sequential(
            nn.Linear(4, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.input = nn.Sequential(
            nn.Linear(9, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                MeshTransolverBlock(
                    config.hidden_dim,
                    config.encoder_heads,
                    config.encoder_slice_num,
                    config.dropout,
                )
                for _ in range(config.encoder_layers)
            ]
        )
        self.bottleneck = SliceBottleneck(
            config.hidden_dim,
            config.latent_tokens,
            config.latent_dim,
            config.encoder_heads,
            config.bottleneck_blocks,
            config.dropout,
        )

    def forward(
        self,
        profile: Tensor,
        coordinates: Tensor,
        bfield: Tensor,
        time: Tensor,
        context: Mapping[str, Tensor] | None = None,
        node_mask: Tensor | None = None,
    ) -> EncoderOutput:
        batch, nodes, _ = profile.shape
        time = time.reshape(batch, 1, 1).expand(batch, nodes, 1)
        features = torch.cat((coordinates, bfield, profile, time), dim=-1)
        hidden = self.input(features) + self.context(
            _context_summary(context, batch, profile.device)
        ).unsqueeze(1)
        for block in self.blocks:
            hidden = block(hidden, node_mask)
        latent, pre, weights, norms = self.bottleneck(hidden, node_mask)
        return EncoderOutput(latent, pre, weights, norms)


class LatentToMeshDecoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        hidden = config.decoder_hidden_dim
        self.query = nn.Sequential(
            nn.Linear(7, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.context = nn.Linear(4, hidden)
        self.latent = nn.Linear(config.latent_dim, hidden)
        self.cross = nn.MultiheadAttention(
            hidden, config.decoder_heads, config.dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Linear(hidden * 4, hidden)
        )
        self.blocks = nn.ModuleList(
            [
                MeshTransolverBlock(
                    hidden,
                    config.decoder_heads,
                    config.encoder_slice_num,
                    config.dropout,
                )
                for _ in range(config.decoder_layers)
            ]
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))

    def forward(
        self,
        latent: Tensor,
        coordinates: Tensor,
        bfield: Tensor,
        time: Tensor,
        context: Mapping[str, Tensor] | None = None,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        batch, nodes, _ = coordinates.shape
        time_feature = time.reshape(batch, 1, 1).expand(batch, nodes, 1)
        query = self.query(torch.cat((coordinates, bfield, time_feature), -1))
        query = query + self.context(
            _context_summary(context, batch, coordinates.device)
        ).unsqueeze(1)
        attended = self.cross(
            query, self.latent(latent), self.latent(latent), need_weights=False
        )[0]
        hidden = query + attended
        hidden = hidden + self.mlp(self.norm(hidden))
        for block in self.blocks:
            hidden = block(hidden, node_mask)
        result = self.output(hidden)
        return result if node_mask is None else result * node_mask.unsqueeze(-1)


class TransolverFrameAutoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or AutoencoderConfig()
        self.encoder, self.decoder = (
            FrameEncoder(self.config),
            LatentToMeshDecoder(self.config),
        )

    def encode(
        self,
        profile: Tensor,
        coordinates: Tensor,
        bfield: Tensor,
        time: Tensor,
        context: Mapping[str, Tensor] | None = None,
        node_mask: Tensor | None = None,
    ) -> EncoderOutput:
        return self.encoder(profile, coordinates, bfield, time, context, node_mask)

    def decode(
        self,
        latent: Tensor,
        coordinates: Tensor,
        bfield: Tensor,
        time: Tensor,
        context: Mapping[str, Tensor] | None = None,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        return self.decoder(latent, coordinates, bfield, time, context, node_mask)

    def forward(
        self,
        profile: Tensor,
        coordinates: Tensor,
        bfield: Tensor,
        time: Tensor,
        context: Mapping[str, Tensor] | None = None,
        node_mask: Tensor | None = None,
    ) -> tuple[EncoderOutput, Tensor]:
        encoded = self.encode(profile, coordinates, bfield, time, context, node_mask)
        return encoded, self.decode(
            encoded.latent, coordinates, bfield, time, context, node_mask
        )

    def model_config(self) -> dict[str, Any]:
        return asdict(self.config)


def autoencoder_loss(
    reconstruction: Tensor,
    target: Tensor,
    node_mask: Tensor | None = None,
    integral_weight: float = 0.01,
) -> tuple[Tensor, dict[str, Tensor]]:
    point = F.huber_loss(reconstruction, target, reduction="none")
    if node_mask is not None:
        point = point * node_mask.unsqueeze(-1)
        denominator = node_mask.sum().clamp_min(1) * target.shape[-1]
    else:
        denominator = point.numel()
    field = point.sum() / denominator
    mask = (
        node_mask.unsqueeze(-1)
        if node_mask is not None
        else torch.ones_like(target[..., :1])
    )
    truth_integral = (target * mask).sum(1) / mask.sum(1).clamp_min(1)
    predicted_integral = (reconstruction * mask).sum(1) / mask.sum(1).clamp_min(1)
    integral = (
        ((predicted_integral - truth_integral) / truth_integral.abs().clamp_min(1e-3))
        .square()
        .mean()
    )
    total = field + integral_weight * integral
    return total, {
        "field": field,
        "integral": integral,
        "parallel": point[..., 0].sum() / denominator * 2,
        "perpendicular": point[..., 1].sum() / denominator * 2,
    }


def anti_collapse_losses(latent: Tensor, occupancy: Tensor) -> dict[str, Tensor]:
    normalized = occupancy / occupancy.sum(-1, keepdim=True).clamp_min(1e-6)
    balance = ((normalized - 1.0 / normalized.shape[-1]) ** 2).mean()
    variance = F.relu(0.01 - latent.var(dim=(0, 1), unbiased=False)).mean()
    centered = F.normalize(latent - latent.mean(1, keepdim=True), dim=-1)
    gram = centered @ centered.transpose(-1, -2)
    eye = torch.eye(gram.shape[-1], device=gram.device)
    decorrelation = ((gram - eye) ** 2).mean()
    return {
        "balance": balance,
        "variance_floor": variance,
        "decorrelation": decorrelation,
    }


__all__ = [
    "AutoencoderConfig",
    "EncoderOutput",
    "SliceBottleneck",
    "FrameEncoder",
    "LatentToMeshDecoder",
    "TransolverFrameAutoencoder",
    "autoencoder_loss",
    "anti_collapse_losses",
]
