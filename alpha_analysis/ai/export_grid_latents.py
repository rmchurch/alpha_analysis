"""Export latent tokens from a trained ASCOT5 grid autoencoder checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from .dataloader import Ascot5Dataset
from .grid_autoencoder import GridLatentAutoencoder
from .train_grid_autoencoder import (
    FeatureNormalizer,
    _discover_sample_folders,
    sample_to_grid_tensors,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, help="Override the checkpoint results root.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--max-nodes", type=int, help="Override node sampling used for encoding.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _safe_key(folder: str) -> str:
    path = Path(folder)
    return path.name or path.parent.name


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = args.checkpoint.expanduser()
    checkpoint: Dict[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("format") != "ascot5_grid_latent_autoencoder_v1":
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}.")
    saved_args = checkpoint["args"]
    results_root = (
        args.results_root.expanduser()
        if args.results_root is not None
        else Path(saved_args["results_root"]).expanduser()
    )
    folders = _discover_sample_folders(
        results_root,
        saved_args["analysis_filename"],
        saved_args["equilibrium_filename"],
        saved_args["bfield_filename"],
    )
    if saved_args.get("max_samples") is not None:
        folders = folders[: saved_args["max_samples"]]
    config_path = checkpoint_path.parent / "config.json"
    with config_path.open() as file:
        run_config = json.load(file)
    if args.split != "all":
        indices = run_config[f"{args.split}_indices"]
        folders = [folders[index] for index in indices]

    dataset = Ascot5Dataset(
        folders,
        analysis_filename=saved_args["analysis_filename"],
        equilibrium_filename=saved_args["equilibrium_filename"],
        bfield_filename=saved_args["bfield_filename"],
        include_bfield=True,
        include_target=False,
        strict=True,
    )
    device = torch.device(args.device)
    model = GridLatentAutoencoder(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalizer = FeatureNormalizer.from_state_dict(checkpoint["normalizer"])
    output_dir = args.output_dir or checkpoint_path.parent / f"{checkpoint_path.stem}_latents"
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)
    max_nodes = args.max_nodes if args.max_nodes is not None else saved_args["max_nodes"]
    manifest: List[Dict[str, Any]] = []

    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            coordinates, physical, _, feature_names = sample_to_grid_tensors(
                sample,
                max_nodes=max_nodes,
                profile_log1p=not saved_args["no_profile_log1p"],
                target_reduction=saved_args["target_reduction"],
                generator=generator,
            )
            if feature_names != checkpoint["feature_names"]:
                raise ValueError(f"Feature layout changed for {sample['folder']}.")
            coordinates = coordinates.unsqueeze(0).to(device)
            physical = normalizer.normalize(physical).unsqueeze(0).to(device)
            mask = torch.ones(
                (1, coordinates.shape[1]), dtype=torch.bool, device=device
            )
            latents, auxiliary = model.encode(
                coordinates, physical, mask, quantized=True
            )
            payload: Dict[str, Any] = {
                "folder": sample["folder"],
                "latents": latents.squeeze(0).cpu(),
                "continuous_latents": auxiliary["continuous_latents"].squeeze(0).cpu(),
                "grid_shape": list(sample["bfield"]["br"].shape),
                "encoded_node_count": int(coordinates.shape[1]),
                "feature_names": feature_names,
                "checkpoint": str(checkpoint_path),
            }
            if model.latent_mode == "vq":
                payload["code_indices"] = auxiliary["code_indices"].squeeze(0).cpu()
                payload["codebook_perplexity"] = float(
                    auxiliary["codebook_perplexity"].cpu()
                )
            if model.scalar_head is not None:
                payload["scalar_prediction"] = float(
                    model.scalar_head(latents.reshape(1, -1)).squeeze().cpu()
                )
            output_path = output_dir / f"{_safe_key(sample['folder'])}.pt"
            torch.save(payload, output_path)
            manifest.append(
                {
                    "folder": sample["folder"],
                    "path": str(output_path),
                    "latent_shape": list(payload["latents"].shape),
                    "encoded_node_count": payload["encoded_node_count"],
                }
            )
    with (output_dir / "manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)
    print(f"Exported {len(manifest)} latent records to {output_dir}")


if __name__ == "__main__":
    main()
