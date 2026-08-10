"""Evaluate frame reconstruction, latent use, occupancy, and rank diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .frame_dataset import (
    AscotSequenceDataset,
    FrameNormalization,
    discover_simulation_folders,
)
from .transolver_autoencoder import AutoencoderConfig, TransolverFrameAutoencoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "later"), default="val")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-nodes", type=int)
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
    indices = list(checkpoint["split_indices"].get(args.split, []))
    if args.split == "later":
        indices = list(range(checkpoint["training_sample_count"], len(folders)))
    if args.max_samples is not None:
        indices = indices[: args.max_samples]
    sequences = AscotSequenceDataset(folders)
    absolute, squared, physical_absolute, physical_squared = [], [], [], []
    baseline_squared, zero_latent_squared, integrations, latents, occupancies = (
        [],
        [],
        [],
        [],
        [],
    )
    by_frame: dict[int, list[float]] = {}
    with torch.inference_mode():
        for simulation_index in indices:
            sample = sequences[simulation_index]
            node_indices = None
            if args.max_nodes and sample["coordinates"].shape[0] > args.max_nodes:
                node_indices = torch.randperm(
                    sample["coordinates"].shape[0],
                    generator=torch.Generator().manual_seed(simulation_index),
                )[: args.max_nodes]
            coords = (
                sample["coordinates"]
                if node_indices is None
                else sample["coordinates"][node_indices]
            )
            bfield = (
                sample["bfield"]
                if node_indices is None
                else sample["bfield"][node_indices]
            )
            coords = (
                normalization.transform_coordinates(coords).unsqueeze(0).to(args.device)
            )
            bfield = normalization.transform_bfield(bfield).unsqueeze(0).to(args.device)
            context = {
                key: value.unsqueeze(0).to(args.device)
                for key, value in sample["context"].items()
            }
            for frame_index, (frame, time) in enumerate(
                zip(sample["profiles"], sample["times"])
            ):
                frame = frame if node_indices is None else frame[node_indices]
                target = (
                    normalization.transform_profile(frame).unsqueeze(0).to(args.device)
                )
                encoded, reconstruction = model(
                    target, coords, bfield, time.reshape(1).to(args.device), context
                )
                zero = model.decode(
                    torch.zeros_like(encoded.latent),
                    coords,
                    bfield,
                    time.reshape(1).to(args.device),
                    context,
                )
                error = reconstruction - target
                absolute.append(error.abs().cpu())
                squared.append(error.square().cpu())
                baseline_squared.append(target.square().cpu())
                zero_latent_squared.append((zero - target).square().cpu())
                physical_error = normalization.inverse_profile(
                    reconstruction.cpu()
                ) - frame.unsqueeze(0)
                physical_absolute.append(physical_error.abs())
                physical_squared.append(physical_error.square())
                true_integral = frame.mean(0)
                predicted_integral = normalization.inverse_profile(
                    reconstruction.cpu()
                ).mean(1)[0]
                integrations.append(
                    (
                        (predicted_integral - true_integral)
                        / true_integral.abs().clamp_min(1e-6)
                    ).abs()
                )
                by_frame.setdefault(frame_index, []).append(float(error.abs().mean()))
                latents.append(encoded.latent.cpu())
                occupancies.append(encoded.slice_norms.cpu())
    if not absolute:
        raise ValueError(f"No samples in split {args.split}")
    latent_matrix = torch.cat(latents).reshape(-1, latents[0].shape[-1])
    singular = torch.linalg.svdvals(latent_matrix - latent_matrix.mean(0))
    spectrum = singular.square() / singular.square().sum().clamp_min(1e-12)
    effective_rank = float(
        torch.exp(-(spectrum * spectrum.clamp_min(1e-12).log()).sum())
    )

    def channel(values):
        joined = torch.cat([item.reshape(-1, 2) for item in values])
        return joined.mean(0).tolist()

    latent = torch.cat(latents)
    normalized_latent = torch.nn.functional.normalize(latent, dim=-1)
    cosine = normalized_latent @ normalized_latent.transpose(-1, -2)
    metrics = {
        "split": args.split,
        "samples": len(indices),
        "transformed_mae": channel(absolute),
        "transformed_rmse": torch.tensor(channel(squared)).sqrt().tolist(),
        "physical_mae": channel(physical_absolute),
        "physical_rmse": torch.tensor(channel(physical_squared)).sqrt().tolist(),
        "integrated_relative_error": torch.stack(integrations).mean(0).tolist(),
        "mean_field_baseline_rmse": torch.cat(
            [x.reshape(-1, 2) for x in baseline_squared]
        )
        .mean(0)
        .sqrt()
        .tolist(),
        "zero_latent_rmse": torch.cat([x.reshape(-1, 2) for x in zero_latent_squared])
        .mean(0)
        .sqrt()
        .tolist(),
        "latent_variance": float(latent.var(unbiased=False)),
        "pairwise_token_cosine": float(cosine.mean()),
        "effective_rank": effective_rank,
        "pca_spectrum": spectrum[:20].tolist(),
        "dead_token_count": int((torch.cat(occupancies).mean(dim=(0, 1)) < 1e-3).sum()),
        "error_by_frame": {
            str(key): sum(value) / len(value) for key, value in by_frame.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
