"""Evaluate free latent and decoded-field rollout against persistence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .frame_dataset import AscotSequenceDataset, FrameNormalization
from .latent_dynamics import (
    DynamicsConfig,
    ExportedLatentDataset,
    ResidualLatentDynamics,
    persistence_rollout,
)
from .transolver_autoencoder import AutoencoderConfig, TransolverFrameAutoencoder


def _mean(values: dict[int, list[float]]) -> dict[str, float]:
    return {str(key): sum(items) / len(items) for key, items in values.items()}


def _mean_channels(values: dict[int, list[Tensor]]) -> dict[str, list[float]]:
    return {
        str(key): torch.stack(items).mean(0).tolist() for key, items in values.items()
    }


def _decode_inputs(
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
    context = {
        key: value.unsqueeze(0).to(device) for key, value in sequence["context"].items()
    }
    return (
        profiles.to(device),
        normalization.transform_coordinates(coordinates).unsqueeze(0).to(device),
        normalization.transform_bfield(bfield).unsqueeze(0).to(device),
        context,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--autoencoder-checkpoint", type=Path)
    parser.add_argument("--split", choices=("val", "later"), default="val")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ResidualLatentDynamics(DynamicsConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    decoder = None
    normalization = None
    if args.autoencoder_checkpoint:
        ae_checkpoint = torch.load(
            args.autoencoder_checkpoint, map_location="cpu", weights_only=False
        )
        decoder = TransolverFrameAutoencoder(
            AutoencoderConfig(**ae_checkpoint["model_config"])
        )
        decoder.load_state_dict(ae_checkpoint["model_state_dict"])
        decoder.to(device).eval()
        normalization = FrameNormalization.from_state_dict(
            ae_checkpoint["normalization"]
        )
    dataset = ExportedLatentDataset(args.latent_dir, args.split)
    errors: dict[int, list[float]] = {}
    persistence: dict[int, list[float]] = {}
    norm_drift: dict[int, list[float]] = {}
    variance_drift: dict[int, list[float]] = {}
    field_mae: dict[int, list[Tensor]] = {}
    field_rmse: dict[int, list[Tensor]] = {}
    field_persistence_rmse: dict[int, list[Tensor]] = {}
    integral_error: dict[int, list[Tensor]] = {}
    with torch.inference_mode():
        for payload in dataset:
            target = payload["latents"].unsqueeze(0).to(device)
            times = payload["times"].to(device)
            context = {
                key: value.unsqueeze(0).to(device)
                if value.ndim == 1
                else value.to(device)
                for key, value in payload.get("context", {}).items()
            } or None
            predicted = model.rollout(target[:, 0], times, context)
            baseline = persistence_rollout(target[:, 0], target.shape[1])
            decode_data = (
                _decode_inputs(payload, normalization, device)
                if decoder and normalization
                else None
            )
            for horizon in range(1, target.shape[1]):
                errors.setdefault(horizon, []).append(
                    float((predicted[:, horizon] - target[:, horizon]).square().mean())
                )
                persistence.setdefault(horizon, []).append(
                    float((baseline[:, horizon] - target[:, horizon]).square().mean())
                )
                norm_drift.setdefault(horizon, []).append(
                    float(
                        predicted[:, horizon].norm(dim=-1).mean()
                        - target[:, horizon].norm(dim=-1).mean()
                    )
                )
                variance_drift.setdefault(horizon, []).append(
                    float(
                        predicted[:, horizon].var(unbiased=False)
                        - target[:, horizon].var(unbiased=False)
                    )
                )
                if decode_data is not None:
                    profiles, coordinates, bfield, field_context = decode_data
                    predicted_field = normalization.inverse_profile(
                        decoder.decode(
                            predicted[:, horizon],
                            coordinates,
                            bfield,
                            times[horizon].reshape(1),
                            field_context,
                        )
                    )[0]
                    persistent_field = normalization.inverse_profile(
                        decoder.decode(
                            baseline[:, horizon],
                            coordinates,
                            bfield,
                            times[horizon].reshape(1),
                            field_context,
                        )
                    )[0]
                    truth = profiles[horizon]
                    error = predicted_field - truth
                    field_mae.setdefault(horizon, []).append(error.abs().mean(0).cpu())
                    field_rmse.setdefault(horizon, []).append(
                        error.square().mean(0).sqrt().cpu()
                    )
                    field_persistence_rmse.setdefault(horizon, []).append(
                        (persistent_field - truth).square().mean(0).sqrt().cpu()
                    )
                    integral_error.setdefault(horizon, []).append(
                        (
                            (predicted_field.mean(0) - truth.mean(0))
                            / truth.mean(0).abs().clamp_min(1e-6)
                        )
                        .abs()
                        .cpu()
                    )
    metrics: dict[str, Any] = {
        "split": args.split,
        "samples": len(dataset),
        "latent_mse_by_horizon": _mean(errors),
        "persistence_mse_by_horizon": _mean(persistence),
        "latent_norm_drift": _mean(norm_drift),
        "token_variance_drift": _mean(variance_drift),
        "beats_persistence_horizons": [
            key
            for key in errors
            if sum(errors[key]) / len(errors[key])
            < sum(persistence[key]) / len(persistence[key])
        ],
    }
    if field_rmse:
        metrics.update(
            {
                "physical_field_mae_by_horizon": _mean_channels(field_mae),
                "physical_field_rmse_by_horizon": _mean_channels(field_rmse),
                "physical_persistence_rmse_by_horizon": _mean_channels(
                    field_persistence_rmse
                ),
                "integrated_pressure_relative_error_by_horizon": _mean_channels(
                    integral_error
                ),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
