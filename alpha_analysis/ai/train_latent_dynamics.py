"""Train residual dynamics on training-cohort frame latents only."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .frame_dataset import AscotSequenceDataset, FrameNormalization
from .latent_dynamics import (
    DynamicsConfig,
    ExportedLatentDataset,
    ResidualLatentDynamics,
    persistence_rollout,
)
from .transolver_autoencoder import AutoencoderConfig, TransolverFrameAutoencoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--autoencoder-checkpoint", type=Path)
    parser.add_argument(
        "--stage", choices=("one-step", "rollout3", "full"), default="one-step"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--decode-weight", type=float, default=0.1)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _context_to_device(
    context: Mapping[str, Tensor] | None, device: torch.device
) -> dict[str, Tensor] | None:
    if context is None:
        return None
    return {
        key: value.unsqueeze(0).to(device) if value.ndim == 1 else value.to(device)
        for key, value in context.items()
    }


def _field_tensors(
    payload: dict[str, Any], normalization: FrameNormalization, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
    sequence = AscotSequenceDataset([payload["folder"]])[0]
    indices = payload.get("node_indices")
    coordinates, bfield, profiles = (
        sequence["coordinates"],
        sequence["bfield"],
        sequence["profiles"],
    )
    if indices is not None:
        indices = indices.long()
        coordinates, bfield, profiles = (
            coordinates[indices],
            bfield[indices],
            profiles[:, indices],
        )
    return (
        normalization.transform_profile(profiles).unsqueeze(0).to(device),
        normalization.transform_coordinates(coordinates).unsqueeze(0).to(device),
        normalization.transform_bfield(bfield).unsqueeze(0).to(device),
        {
            key: value.unsqueeze(0).to(device)
            for key, value in sequence["context"].items()
        },
    )


def _sequence_loss(
    model: ResidualLatentDynamics,
    payload: dict[str, Any],
    stage: str,
    device: torch.device,
    decoder: TransolverFrameAutoencoder | None,
    normalization: FrameNormalization | None,
    decode_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    target = payload["latents"].unsqueeze(0).to(device)
    times = payload["times"].unsqueeze(0).to(device)
    context = _context_to_device(payload.get("context"), device)
    if target.shape[1] < 2:
        raise ValueError("Dynamics requires at least two frames")
    if stage == "one-step":
        predictions = torch.stack(
            [
                model(
                    target[:, index],
                    times[:, index],
                    times[:, index + 1] - times[:, index],
                    context,
                )
                for index in range(target.shape[1] - 1)
            ],
            dim=1,
        )
        truth = target[:, 1:]
    else:
        horizon = min(target.shape[1], 4) if stage == "rollout3" else target.shape[1]
        predictions = model.rollout(target[:, 0], times[:, :horizon], context)[:, 1:]
        truth = target[:, 1:horizon]
    latent_loss = F.mse_loss(predictions, truth)
    persistence = F.mse_loss(
        persistence_rollout(target[:, 0], truth.shape[1] + 1)[:, 1:], truth
    )
    field_loss = latent_loss.new_zeros(())
    if stage != "one-step" and decode_weight > 0:
        if decoder is None or normalization is None:
            raise ValueError(
                "--autoencoder-checkpoint is required when --decode-weight is positive"
            )
        profiles, coordinates, bfield, field_context = _field_tensors(
            payload, normalization, device
        )
        decoded = torch.stack(
            [
                decoder.decode(
                    predictions[:, index],
                    coordinates,
                    bfield,
                    times[:, index + 1],
                    field_context,
                )
                for index in range(predictions.shape[1])
            ],
            dim=1,
        )
        field_loss = F.huber_loss(decoded, profiles[:, 1 : predictions.shape[1] + 1])
    total = latent_loss + decode_weight * field_loss
    return total, {
        "total": float(total.detach()),
        "latent_mse": float(latent_loss.detach()),
        "field_huber": float(field_loss.detach()),
        "persistence_mse": float(persistence.detach()),
    }


def _epoch(
    model: ResidualLatentDynamics,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    stage: str,
    grad_clip: float,
    dry_run: bool,
    decoder: TransolverFrameAutoencoder | None,
    normalization: FrameNormalization | None,
    decode_weight: float,
) -> dict[str, float]:
    model.train(optimizer is not None)
    values = {
        key: 0.0 for key in ("total", "latent_mse", "field_huber", "persistence_mse")
    }
    count = 0
    for payload in loader:
        with torch.set_grad_enabled(optimizer is not None):
            loss, metrics = _sequence_loss(
                model, payload, stage, device, decoder, normalization, decode_weight
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        for key in values:
            values[key] += metrics[key]
        count += 1
        if dry_run:
            break
    return {key: value / max(count, 1) for key, value in values.items()}


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    if (
        args.stage != "one-step"
        and args.decode_weight > 0
        and args.autoencoder_checkpoint is None
    ):
        raise ValueError(
            "--autoencoder-checkpoint is required for decoded rollout loss"
        )
    decoder = None
    normalization = None
    if args.autoencoder_checkpoint is not None:
        checkpoint = torch.load(
            args.autoencoder_checkpoint, map_location="cpu", weights_only=False
        )
        decoder = TransolverFrameAutoencoder(
            AutoencoderConfig(**checkpoint["model_config"])
        )
        decoder.load_state_dict(checkpoint["model_state_dict"])
        decoder.to(args.device).eval()
        decoder.requires_grad_(False)
        normalization = FrameNormalization.from_state_dict(checkpoint["normalization"])
    train = ExportedLatentDataset(args.latent_dir, "train")
    validation = ExportedLatentDataset(args.latent_dir, "val")
    first = train[0]["latents"]
    config = DynamicsConfig(
        first.shape[1], first.shape[2], args.hidden_dim, args.heads, args.layers
    )
    device = torch.device(args.device)
    model = ResidualLatentDynamics(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), args.lr, weight_decay=args.weight_decay
    )

    def collate(items):
        return items[0]

    train_loader = DataLoader(train, batch_size=1, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(validation, batch_size=1, shuffle=False, collate_fn=collate)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model_config": asdict(config),
        "training_split": "train",
        "later_simulations_are_evaluation_only": True,
    }
    (args.save_dir / "config.json").write_text(json.dumps(config_payload, indent=2))
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        common = (
            args.stage,
            args.grad_clip,
            args.dry_run,
            decoder,
            normalization,
            args.decode_weight,
        )
        train_metrics = _epoch(model, train_loader, device, optimizer, *common)
        val_metrics = _epoch(model, val_loader, device, None, *common)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with (args.save_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(record) + "\n")
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(config),
            "epoch": epoch,
            "metrics": record,
            "stage": args.stage,
            "autoencoder_checkpoint": str(args.autoencoder_checkpoint)
            if args.autoencoder_checkpoint
            else None,
        }
        torch.save(checkpoint, args.save_dir / "last.pt")
        if val_metrics["total"] < best:
            best = val_metrics["total"]
            torch.save(checkpoint, args.save_dir / "best.pt")
        print(json.dumps(record), flush=True)
        if args.dry_run:
            break


if __name__ == "__main__":
    main()
