#!/usr/bin/env python3
"""Plot a geometry-aware truth/reconstruction comparison for one profile frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from alpha_analysis.ai.frame_dataset import (
    AscotSequenceDataset,
    FrameNormalization,
    discover_simulation_folders,
)
from alpha_analysis.ai.transolver_autoencoder import (
    AutoencoderConfig,
    TransolverFrameAutoencoder,
)


def signed_log10(value: np.ndarray) -> np.ndarray:
    """Compress physical dynamic range while preserving sign and exact zeros."""
    return np.sign(value) * np.log10(1.0 + np.abs(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--simulation-index", type=int, required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument(
        "--phi-index",
        type=int,
        help="Toroidal slice index (default: middle of the phi grid).",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    normalization = FrameNormalization.from_state_dict(checkpoint["normalization"])
    model = TransolverFrameAutoencoder(AutoencoderConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device).eval()

    folders = discover_simulation_folders(args.results_root)
    split_name = next(
        (
            name
            for name, indices in checkpoint["split_indices"].items()
            if args.simulation_index in indices
        ),
        "later"
        if args.simulation_index >= checkpoint["training_sample_count"]
        else "unknown",
    )
    sequences = AscotSequenceDataset(folders, include_target=False)
    sample = sequences.load_frame(args.simulation_index, args.frame_index)
    grid_shape = tuple(sample["grid_shape"])
    if len(grid_shape) != 3:
        raise ValueError(f"Expected a 3-D profile grid, got {grid_shape}")
    phi_index = grid_shape[2] // 2 if args.phi_index is None else args.phi_index
    if not 0 <= phi_index < grid_shape[2]:
        raise ValueError(f"phi-index must be in [0, {grid_shape[2]})")

    frame = sample["profile"]
    context = {
        key: value.unsqueeze(0).to(args.device)
        for key, value in sample["context"].items()
    }
    profile = normalization.transform_profile(frame).unsqueeze(0).to(args.device)
    coordinates = (
        normalization.transform_coordinates(sample["coordinates"])
        .unsqueeze(0)
        .to(args.device)
    )
    bfield = (
        normalization.transform_bfield(sample["bfield"])
        .unsqueeze(0)
        .to(args.device)
    )
    time = sample["time"].reshape(1).to(args.device)
    with torch.inference_mode():
        _, predicted = model(profile, coordinates, bfield, time, context)
        reconstruction = normalization.inverse_profile(predicted.cpu())[0]

    truth = frame.reshape(*grid_shape, 2).numpy()
    reconstruction_np = reconstruction.reshape(*grid_shape, 2).numpy()
    coordinates_np = sample["coordinates"].reshape(*grid_shape, 3).numpy()
    rho = coordinates_np[:, 0, 0, 0]
    theta = coordinates_np[0, :, 0, 1]

    figure, axes = plt.subplots(2, 4, figsize=(18, 8.5), constrained_layout=True)
    channel_names = ("Parallel pressure", "Perpendicular pressure")
    metric_lines: list[str] = []
    for channel, name in enumerate(channel_names):
        true_channel = truth[..., channel]
        predicted_channel = reconstruction_np[..., channel]
        true_slice = signed_log10(true_channel[:, :, phi_index])
        predicted_slice = signed_log10(predicted_channel[:, :, phi_index])
        residual_slice = predicted_slice - true_slice
        color_limit = max(
            float(np.quantile(np.abs(np.concatenate((true_slice.ravel(), predicted_slice.ravel()))), 0.995)),
            1e-8,
        )
        residual_limit = max(float(np.quantile(np.abs(residual_slice), 0.995)), 1e-8)

        image = axes[channel, 0].pcolormesh(
            theta, rho, true_slice, shading="auto", cmap="RdBu_r",
            vmin=-color_limit, vmax=color_limit,
        )
        axes[channel, 0].set_title(f"{name}: ground truth")
        figure.colorbar(image, ax=axes[channel, 0], label="signed log10(1 + |pressure|)")

        image = axes[channel, 1].pcolormesh(
            theta, rho, predicted_slice, shading="auto", cmap="RdBu_r",
            vmin=-color_limit, vmax=color_limit,
        )
        axes[channel, 1].set_title(f"{name}: reconstruction")
        figure.colorbar(image, ax=axes[channel, 1], label="signed log10(1 + |pressure|)")

        image = axes[channel, 2].pcolormesh(
            theta, rho, residual_slice, shading="auto", cmap="coolwarm",
            vmin=-residual_limit, vmax=residual_limit,
        )
        axes[channel, 2].set_title("Reconstruction - truth")
        figure.colorbar(image, ax=axes[channel, 2], label="difference in signed-log view")

        true_radial = true_channel.mean(axis=(1, 2))
        predicted_radial = predicted_channel.mean(axis=(1, 2))
        axes[channel, 3].plot(rho, true_radial, color="black", lw=2.2, label="Truth")
        axes[channel, 3].plot(
            rho, predicted_radial, color="#d95f02", lw=1.8, ls="--", label="Reconstruction"
        )
        axes[channel, 3].set_title("Angularly averaged radial profile")
        axes[channel, 3].set_xlabel(r"$\rho$")
        axes[channel, 3].set_ylabel("pressure (physical units)")
        axes[channel, 3].legend(frameon=False)

        error = predicted_channel - true_channel
        relative_l2 = np.linalg.norm(error.ravel()) / max(
            np.linalg.norm(true_channel.ravel()), 1e-12
        )
        correlation = np.corrcoef(true_channel.ravel(), predicted_channel.ravel())[0, 1]
        metric_lines.append(
            f"{name}: relative L2={relative_l2:.3f}, correlation={correlation:.3f}"
        )
        for column in range(3):
            axes[channel, column].set_xlabel(r"$\theta$")
            axes[channel, column].set_ylabel(r"$\rho$")

    folder_name = Path(sample["folder"]).name
    figure.suptitle(
        f"Held-out frame reconstruction ({split_name} split)\n"
        f"{folder_name}, simulation index {args.simulation_index}, frame {args.frame_index}, "
        f"phi index {phi_index}; checkpoint epoch {checkpoint.get('epoch', '?')}\n"
        + " | ".join(metric_lines),
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(args.output)
    print("\n".join(metric_lines))


if __name__ == "__main__":
    main()
