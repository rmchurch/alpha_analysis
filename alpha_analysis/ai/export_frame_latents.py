"""Export deterministic ``[T,G,D]`` frame latents without changing splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .frame_dataset import (
    AscotSequenceDataset,
    FrameNormalization,
    discover_simulation_folders,
)
from .transolver_autoencoder import AutoencoderConfig, TransolverFrameAutoencoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Optional deterministic node limit; full mesh is the default.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int)
    return parser


def _encode_sequence(
    model: TransolverFrameAutoencoder,
    sequence: dict[str, Any],
    normalization: FrameNormalization,
    device: torch.device,
    max_nodes: int | None,
    seed: int,
) -> dict[str, Any]:
    coordinates = sequence["coordinates"]
    bfield = sequence["bfield"]
    node_indices = None
    if max_nodes is not None and coordinates.shape[0] > max_nodes:
        generator = torch.Generator().manual_seed(seed)
        node_indices = (
            torch.randperm(coordinates.shape[0], generator=generator)[:max_nodes]
            .sort()
            .values
        )
        coordinates, bfield = coordinates[node_indices], bfield[node_indices]
    coordinates = (
        normalization.transform_coordinates(coordinates).unsqueeze(0).to(device)
    )
    bfield = normalization.transform_bfield(bfield).unsqueeze(0).to(device)
    context = {
        key: value.unsqueeze(0).to(device) for key, value in sequence["context"].items()
    }
    latents, norms = [], []
    for frame, time in zip(sequence["profiles"], sequence["times"]):
        if node_indices is not None:
            frame = frame[node_indices]
        profile = normalization.transform_profile(frame).unsqueeze(0).to(device)
        encoded = model.encode(
            profile, coordinates, bfield, time.reshape(1).to(device), context
        )
        latents.append(encoded.latent[0].cpu())
        norms.append(encoded.slice_norms[0].cpu())
    return {
        "folder": sequence["folder"],
        "times": sequence["times"],
        "latents": torch.stack(latents),
        "slice_norms": torch.stack(norms),
        "target": sequence["target"],
        "grid_shape": sequence["grid_shape"],
        "node_count": sequence["coordinates"].shape[0],
        "node_indices": node_indices,
        "context": sequence["context"],
    }


def main() -> None:
    args = build_parser().parse_args()
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.overwrite
    ):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TransolverFrameAutoencoder(AutoencoderConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device).eval()
    normalization = FrameNormalization.from_state_dict(checkpoint["normalization"])
    folders = discover_simulation_folders(args.results_root)
    if args.max_samples is not None:
        folders = folders[: args.max_samples]
    sequences = AscotSequenceDataset(folders)
    saved_splits = checkpoint["split_indices"]
    split_lookup = {
        index: split for split, indices in saved_splits.items() for index in indices
    }
    cutoff = int(
        checkpoint.get(
            "training_sample_count",
            max(saved_splits["train"] + saved_splits["val"]) + 1,
        )
    )
    records = []
    with torch.inference_mode():
        for index in range(len(sequences)):
            split = split_lookup.get(
                index, "later" if index >= cutoff else "unassigned"
            )
            if split == "unassigned":
                raise ValueError(
                    f"Historical simulation index {index} is absent from checkpoint splits"
                )
            payload = _encode_sequence(
                model,
                sequences[index],
                normalization,
                torch.device(args.device),
                args.max_nodes,
                seed=index,
            )
            payload.update({"dataset_index": index, "split": split})
            directory = args.output_dir / split
            directory.mkdir(exist_ok=True)
            path = directory / f"{index:06d}_{Path(payload['folder']).name}.pt"
            torch.save(payload, path)
            records.append(
                {
                    "folder": payload["folder"],
                    "dataset_index": index,
                    "split": split,
                    "path": str(path.relative_to(args.output_dir)),
                    "frames": len(payload["times"]),
                }
            )
            print(
                f"{index + 1}/{len(sequences)} {split} {payload['folder']}", flush=True
            )
    (args.output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in records)
    )
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "results_root": str(args.results_root.resolve()),
        "normalization": normalization.state_dict(),
        "time_convention": normalization.time_convention,
        "max_nodes": args.max_nodes,
        "full_mesh": args.max_nodes is None,
        "training_sample_count": cutoff,
        "later_simulations_are_prediction_only": True,
        "split_sizes": {
            name: sum(item["split"] == name for item in records)
            for name in ("train", "val", "later")
        },
    }
    # JSON cannot serialize tensors.
    metadata["normalization"] = {
        key: value.tolist() if isinstance(value, torch.Tensor) else value
        for key, value in metadata["normalization"].items()
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
