"""Deterministic residual dynamics over exported per-frame latents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class DynamicsConfig:
    latent_tokens: int = 16
    latent_dim: int = 64
    hidden_dim: int = 128
    heads: int = 4
    layers: int = 3
    dropout: float = 0.0


def _context_features(
    context: Mapping[str, Tensor] | None, batch: int, device: torch.device
) -> Tensor:
    if not context:
        return torch.zeros(batch, 4, device=device)
    values = []
    for key in ("R_lmn", "Z_lmn"):
        value = context.get(key)
        if value is None:
            values.extend(
                (torch.zeros(batch, device=device), torch.zeros(batch, device=device))
            )
        else:
            value = value.to(device).float()
            if value.ndim == 1:
                value = value.unsqueeze(0)
            flat = value.reshape(value.shape[0], -1)
            values.extend((flat.mean(1), flat.std(1, unbiased=False)))
    return torch.stack(values, -1)


class ResidualLatentDynamics(nn.Module):
    def __init__(self, config: DynamicsConfig | None = None) -> None:
        super().__init__()
        self.config = config or DynamicsConfig()
        c = self.config
        self.input = nn.Linear(c.latent_dim, c.hidden_dim)
        self.time = nn.Sequential(
            nn.Linear(2, c.hidden_dim), nn.SiLU(), nn.Linear(c.hidden_dim, c.hidden_dim)
        )
        self.context = nn.Sequential(
            nn.Linear(4, c.hidden_dim), nn.SiLU(), nn.Linear(c.hidden_dim, c.hidden_dim)
        )
        self.slots = nn.Parameter(torch.randn(1, c.latent_tokens, c.hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            c.hidden_dim,
            c.heads,
            c.hidden_dim * 4,
            c.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, c.layers)
        self.output = nn.Sequential(
            nn.LayerNorm(c.hidden_dim), nn.Linear(c.hidden_dim, c.latent_dim)
        )

    def forward(
        self,
        latent: Tensor,
        time: Tensor,
        dt: Tensor,
        context: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        batch, tokens, _ = latent.shape
        if tokens != self.config.latent_tokens:
            raise ValueError(
                f"Expected {self.config.latent_tokens} latent tokens, got {tokens}"
            )
        temporal = torch.stack((time.reshape(batch), dt.reshape(batch)), -1)
        hidden = self.input(latent) + self.slots
        hidden = hidden + self.time(temporal).unsqueeze(1)
        hidden = hidden + self.context(
            _context_features(context, batch, latent.device)
        ).unsqueeze(1)
        return latent + self.output(self.blocks(hidden))

    def rollout(
        self,
        initial: Tensor,
        times: Tensor,
        context: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        if times.ndim == 1:
            times = times.unsqueeze(0).expand(initial.shape[0], -1)
        states = [initial]
        current = initial
        for step in range(times.shape[1] - 1):
            current = self(
                current, times[:, step], times[:, step + 1] - times[:, step], context
            )
            states.append(current)
        return torch.stack(states, dim=1)


class LatentSequenceHead(nn.Module):
    def __init__(
        self, latent_dim: int, hidden_dim: int = 128, heads: int = 4, layers: int = 2
    ) -> None:
        super().__init__()
        self.input = nn.Linear(latent_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim, heads, hidden_dim * 4, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.pool = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, sequence: Tensor) -> Tensor:
        batch, frames, tokens, dims = sequence.shape
        hidden = self.blocks(self.input(sequence.reshape(batch, frames * tokens, dims)))
        return self.pool(hidden.mean(1)).squeeze(-1)


class ExportedLatentDataset(Dataset):
    def __init__(self, export_dir: Path, split: str = "train") -> None:
        self.export_dir = export_dir
        self.records = [
            json.loads(line)
            for line in (export_dir / "manifest.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.records = [item for item in self.records if item["split"] == split]
        if not self.records:
            raise ValueError(f"No '{split}' records in {export_dir / 'manifest.jsonl'}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        payload = torch.load(
            self.export_dir / record["path"], map_location="cpu", weights_only=False
        )
        return payload


def persistence_rollout(initial: Tensor, steps: int) -> Tensor:
    return initial.unsqueeze(1).expand(-1, steps, -1, -1).clone()


def rollout_mse(predicted: Tensor, target: Tensor) -> Tensor:
    if predicted.shape != target.shape:
        raise ValueError(
            f"Rollout shapes differ: {predicted.shape} versus {target.shape}"
        )
    return (predicted - target).square().mean(dim=(-1, -2))


__all__ = [
    "DynamicsConfig",
    "ResidualLatentDynamics",
    "LatentSequenceHead",
    "ExportedLatentDataset",
    "persistence_rollout",
    "rollout_mse",
]
