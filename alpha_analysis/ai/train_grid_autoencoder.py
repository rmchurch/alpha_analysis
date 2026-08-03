"""Train a reconstruction-supervised latent representation of ASCOT5 grids."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .dataloader import (
    DEFAULT_ANALYSIS_FILENAME,
    DEFAULT_BFIELD_FILENAME,
    DEFAULT_EQUILIBRIUM_FILENAME,
    DEFAULT_TARGET_DATABASE_KEY,
    Ascot5Dataset,
)
from .grid_autoencoder import GridLatentAutoencoder, masked_reconstruction_metrics


def _identity_collate(batch: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(batch)


def _discover_sample_folders(
    root: Path,
    analysis_filename: str,
    equilibrium_filename: str,
    bfield_filename: str,
) -> List[Path]:
    folders = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir()
        and (path / analysis_filename).is_file()
        and (path / equilibrium_filename).is_file()
        and (path / bfield_filename).is_file()
    ]
    if not folders:
        raise ValueError(f"No sample folders with required HDF5 files found under {root}")
    return folders


def _default_target_database_path(results_root: Path) -> Path | None:
    path = results_root / "G1600_end_database.json"
    return path if path.is_file() else None


def _make_index_coordinates(grid_shape: Sequence[int]) -> Tensor:
    axes = [torch.linspace(-1.0, 1.0, steps=size) for size in grid_shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)


def _normalize_coordinate_columns(coordinates: Tensor) -> Tensor:
    coordinates = torch.nan_to_num(coordinates.float())
    minimum = coordinates.amin(dim=0, keepdim=True)
    span = (coordinates.amax(dim=0, keepdim=True) - minimum).clamp_min(1.0e-12)
    return 2.0 * (coordinates - minimum) / span - 1.0


def _coordinate_features(sample: Dict[str, Any], grid_shape: Sequence[int]) -> Tensor:
    bfield = sample["bfield"]
    if all(
        name in bfield and tuple(bfield[name].shape) == tuple(grid_shape)
        for name in ("rho", "theta", "phi")
    ):
        values = torch.stack(
            [bfield[name].reshape(-1) for name in ("rho", "theta", "phi")], dim=-1
        )
        return _normalize_coordinate_columns(values)
    return _make_index_coordinates(grid_shape)


def _profile_channels(
    tensor: Tensor, grid_shape: Sequence[int], *, log1p: bool
) -> Tensor:
    if tuple(tensor.shape[-len(grid_shape) :]) != tuple(grid_shape):
        raise ValueError(
            f"Expected trailing profile shape {tuple(grid_shape)}, got {tuple(tensor.shape)}."
        )
    channels = tensor.reshape(-1, *grid_shape).float()
    if log1p:
        channels = torch.sign(channels) * torch.log1p(torch.abs(channels))
    return channels.reshape(channels.shape[0], -1).transpose(0, 1).contiguous()


def _physical_features(
    sample: Dict[str, Any], grid_shape: Sequence[int], *, profile_log1p: bool
) -> Tuple[Tensor, List[str]]:
    para = _profile_channels(sample["prs_para"], grid_shape, log1p=profile_log1p)
    perp = _profile_channels(sample["prs_perp"], grid_shape, log1p=profile_log1p)
    bfield = []
    for name in ("br", "bphi", "bz"):
        value = sample["bfield"][name]
        if tuple(value.shape) != tuple(grid_shape):
            raise ValueError(
                f"Expected bfield/{name} shape {tuple(grid_shape)}, got {tuple(value.shape)}."
            )
        bfield.append(value.reshape(-1, 1).float())
    names = (
        [f"prs_para_{index}" for index in range(para.shape[1])]
        + [f"prs_perp_{index}" for index in range(perp.shape[1])]
        + ["br", "bphi", "bz"]
    )
    physical = torch.cat((para, perp, *bfield), dim=-1)
    return torch.nan_to_num(physical), names


def _reduce_target(target: Tensor, reduction: str) -> Tensor:
    finite = target[torch.isfinite(target)]
    if finite.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    if reduction == "mean":
        return finite.mean().float()
    if reduction == "sum":
        return finite.sum().float()
    if reduction == "max":
        return finite.max().float()
    raise ValueError(f"Unsupported target reduction: {reduction}")


def sample_to_grid_tensors(
    sample: Dict[str, Any],
    *,
    max_nodes: int | None,
    profile_log1p: bool,
    target_reduction: str,
    generator: torch.Generator,
) -> Tuple[Tensor, Tensor, Tensor | None, List[str]]:
    """Convert one complete simulation to coordinates and physical node channels."""

    grid_shape = tuple(sample["bfield"]["br"].shape)
    coordinates = _coordinate_features(sample, grid_shape)
    physical, feature_names = _physical_features(
        sample, grid_shape, profile_log1p=profile_log1p
    )
    if coordinates.shape[0] != physical.shape[0]:
        raise ValueError("Coordinate and physical node counts do not match.")
    if max_nodes is not None and coordinates.shape[0] > max_nodes:
        indices = torch.randperm(coordinates.shape[0], generator=generator)[:max_nodes]
        coordinates = coordinates[indices]
        physical = physical[indices]
    target = (
        _reduce_target(sample["target"], target_reduction)
        if "target" in sample
        else None
    )
    return coordinates, physical, target, feature_names


@dataclass
class FeatureNormalizer:
    """Training-split statistics for physical reconstruction channels."""

    mean: Tensor
    std: Tensor

    def normalize(self, value: Tensor) -> Tensor:
        return (value - self.mean.to(value.device)) / self.std.to(value.device)

    def denormalize(self, value: Tensor) -> Tensor:
        return value * self.std.to(value.device) + self.mean.to(value.device)

    def state_dict(self) -> Dict[str, Tensor]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: Dict[str, Tensor]) -> "FeatureNormalizer":
        return cls(mean=state["mean"].float(), std=state["std"].float())


def fit_feature_normalizer(
    dataset: Dataset,
    *,
    max_nodes: int | None,
    profile_log1p: bool,
    target_reduction: str,
    seed: int,
) -> Tuple[FeatureNormalizer, List[str]]:
    """Compute channel statistics from the training split only."""

    total: Tensor | None = None
    square_total: Tensor | None = None
    count = 0
    feature_names: List[str] | None = None
    generator = torch.Generator().manual_seed(seed)
    for index in range(len(dataset)):
        _, physical, _, names = sample_to_grid_tensors(
            dataset[index],
            max_nodes=max_nodes,
            profile_log1p=profile_log1p,
            target_reduction=target_reduction,
            generator=generator,
        )
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("All simulations must have the same physical channel layout.")
        values = physical.double()
        batch_total = values.sum(dim=0)
        batch_square_total = values.square().sum(dim=0)
        total = batch_total if total is None else total + batch_total
        square_total = (
            batch_square_total if square_total is None else square_total + batch_square_total
        )
        count += values.shape[0]
    if total is None or square_total is None or feature_names is None or count == 0:
        raise ValueError("Cannot fit normalization on an empty dataset.")
    mean = total / count
    variance = (square_total / count - mean.square()).clamp_min(1.0e-12)
    return FeatureNormalizer(mean.float(), variance.sqrt().float()), feature_names


def _pad_nodes(values: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
    max_nodes = max(value.shape[0] for value in values)
    width = values[0].shape[1]
    output = values[0].new_zeros((len(values), max_nodes, width))
    mask = torch.zeros((len(values), max_nodes), dtype=torch.bool)
    for index, value in enumerate(values):
        if value.shape[1] != width:
            raise ValueError("All samples in a batch must use the same channel count.")
        output[index, : value.shape[0]] = value
        mask[index, : value.shape[0]] = True
    return output, mask


def make_grid_batch(
    samples: Sequence[Dict[str, Any]],
    *,
    normalizer: FeatureNormalizer,
    max_nodes: int | None,
    profile_log1p: bool,
    target_reduction: str,
    generator: torch.Generator,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor | None]:
    coordinates, physical, targets = [], [], []
    for sample in samples:
        coords, fields, target, _ = sample_to_grid_tensors(
            sample,
            max_nodes=max_nodes,
            profile_log1p=profile_log1p,
            target_reduction=target_reduction,
            generator=generator,
        )
        coordinates.append(coords)
        physical.append(normalizer.normalize(fields))
        if target is not None:
            targets.append(target)
    coordinate_batch, mask = _pad_nodes(coordinates)
    physical_batch, _ = _pad_nodes(physical)
    target_batch = torch.stack(targets) if targets else None
    return (
        coordinate_batch.to(device),
        physical_batch.to(device),
        mask.to(device),
        target_batch.to(device) if target_batch is not None else None,
    )


def run_epoch(
    model: GridLatentAutoencoder,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    normalizer: FeatureNormalizer,
    args: argparse.Namespace,
    generator: torch.Generator,
    device: torch.device,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = {}
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for samples in loader:
            coordinates, physical, mask, target = make_grid_batch(
                samples,
                normalizer=normalizer,
                max_nodes=args.max_nodes,
                profile_log1p=not args.no_profile_log1p,
                target_reduction=args.target_reduction,
                generator=generator,
                device=device,
            )
            output = model(coordinates, physical, mask)
            reconstruction_mse, reconstruction_mae = masked_reconstruction_metrics(
                output["reconstruction"], physical, mask
            )
            scalar_mse = physical.new_zeros(())
            scalar_mae = physical.new_zeros(())
            if args.scalar_weight > 0.0:
                if target is None or "scalar_prediction" not in output:
                    raise RuntimeError(
                        "Scalar supervision was requested but target/head is absent."
                    )
                scalar_mse = F.mse_loss(output["scalar_prediction"], target)
                scalar_mae = F.l1_loss(output["scalar_prediction"], target)
            loss = (
                args.reconstruction_weight * reconstruction_mse
                + args.scalar_weight * scalar_mse
                + args.vq_weight * output["vq_loss"]
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            values = {
                "loss": loss,
                "reconstruction_mse": reconstruction_mse,
                "reconstruction_mae": reconstruction_mae,
                "scalar_mse": scalar_mse,
                "scalar_mae": scalar_mae,
                "vq_loss": output["vq_loss"],
                "codebook_perplexity": output["codebook_perplexity"],
                "latent_token_std": output["latent_token_std"],
            }
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
            batches += 1
    return {name: value / max(batches, 1) for name, value in totals.items()}


def _split_indices(count: int, fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    if fraction >= 1.0 or count == 1:
        return indices, []
    split = max(1, min(count - 1, int(count * fraction)))
    return indices[:split], indices[split:]


def _save_checkpoint(
    path: Path,
    *,
    model: GridLatentAutoencoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_config: Dict[str, Any],
    normalizer: FeatureNormalizer,
    feature_names: Sequence[str],
    args: argparse.Namespace,
    metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "ascot5_grid_latent_autoencoder_v1",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config,
            "normalizer": normalizer.state_dict(),
            "feature_names": list(feature_names),
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=Path("/global/cfs/cdirs/m5300/results/G1600")
    )
    parser.add_argument("--save-dir", type=Path, default=Path("runs/grid_autoencoder"))
    parser.add_argument("--analysis-filename", default=DEFAULT_ANALYSIS_FILENAME)
    parser.add_argument("--equilibrium-filename", default=DEFAULT_EQUILIBRIUM_FILENAME)
    parser.add_argument("--bfield-filename", default=DEFAULT_BFIELD_FILENAME)
    parser.add_argument("--target-database", type=Path)
    parser.add_argument("--target-database-key", default=DEFAULT_TARGET_DATABASE_KEY)
    parser.add_argument("--target-reduction", choices=("mean", "sum", "max"), default="mean")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Randomly sample this many grid nodes per simulation; omit for the full grid.",
    )
    parser.add_argument(
        "--normalization-max-nodes",
        type=int,
        default=None,
        help="Cap nodes per training sample while fitting statistics; omit for the full grid.",
    )
    parser.add_argument("--no-profile-log1p", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--num-latents", type=int, default=32)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--latent-mode", choices=("continuous", "vq"), default="continuous")
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--scalar-weight", type=float, default=0.0)
    parser.add_argument("--vq-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not (0.0 < args.train_fraction <= 1.0):
        raise ValueError("--train-fraction must be in (0, 1].")
    for name in ("max_nodes", "normalization_max_nodes"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if min(args.reconstruction_weight, args.scalar_weight, args.vq_weight) < 0.0:
        raise ValueError("Loss weights must be non-negative.")
    if args.reconstruction_weight == 0.0 and args.scalar_weight == 0.0:
        raise ValueError("At least one of reconstruction or scalar supervision must be enabled.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    results_root = args.results_root.expanduser()
    target_database = args.target_database
    if args.scalar_weight > 0.0 and target_database is None:
        target_database = _default_target_database_path(results_root)
    args.target_database = target_database

    folders = _discover_sample_folders(
        results_root,
        args.analysis_filename,
        args.equilibrium_filename,
        args.bfield_filename,
    )
    if args.max_samples is not None:
        folders = folders[: args.max_samples]
    dataset = Ascot5Dataset(
        folders,
        analysis_filename=args.analysis_filename,
        equilibrium_filename=args.equilibrium_filename,
        bfield_filename=args.bfield_filename,
        include_bfield=True,
        include_target=args.scalar_weight > 0.0,
        target_database_path=target_database,
        target_database_key=args.target_database_key,
        strict=True,
    )
    train_indices, val_indices = _split_indices(len(dataset), args.train_fraction, args.seed)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_indices else None
    print("Fitting physical-channel normalization from the training split...")
    normalizer, feature_names = fit_feature_normalizer(
        train_dataset,
        max_nodes=args.normalization_max_nodes,
        profile_log1p=not args.no_profile_log1p,
        target_reduction=args.target_reduction,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_identity_collate,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=_identity_collate,
        )
        if val_dataset is not None
        else None
    )
    model_config = {
        "physical_dim": len(feature_names),
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "num_latents": args.num_latents,
        "heads": args.heads,
        "encoder_layers": args.encoder_layers,
        "decoder_layers": args.decoder_layers,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
        "latent_mode": args.latent_mode,
        "codebook_size": args.codebook_size,
        "commitment_cost": args.commitment_cost,
        "predict_scalar": args.scalar_weight > 0.0,
    }
    model = GridLatentAutoencoder(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    print(
        f"Training grid autoencoder on {len(train_dataset)} simulations"
        + (f"; validating on {len(val_dataset)}" if val_dataset is not None else "")
        + f"; physical_dim={len(feature_names)}, latent=[{args.num_latents}, "
        f"{args.latent_dim}], mode={args.latent_mode}, device={device}."
    )

    if args.dry_run:
        coordinates, physical, mask, target = make_grid_batch(
            next(iter(train_loader)),
            normalizer=normalizer,
            max_nodes=args.max_nodes,
            profile_log1p=not args.no_profile_log1p,
            target_reduction=args.target_reduction,
            generator=generator,
            device=device,
        )
        model.eval()
        with torch.no_grad():
            output = model(coordinates, physical, mask)
        print(
            f"dry_run coordinates={tuple(coordinates.shape)} physical={tuple(physical.shape)} "
            f"latents={tuple(output['latents'].shape)} "
            f"reconstruction={tuple(output['reconstruction'].shape)} "
            f"target={None if target is None else tuple(target.shape)}"
        )
        return

    args.save_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "args": vars(args),
        "model_config": model_config,
        "feature_names": feature_names,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }
    with (args.save_dir / "config.json").open("w") as file:
        json.dump(config, file, indent=2, default=str)

    best_reconstruction = float("inf")
    history: List[Dict[str, float]] = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            normalizer=normalizer,
            args=args,
            generator=generator,
            device=device,
        )
        metrics: Dict[str, float] = {"epoch": float(epoch)}
        metrics.update({f"train_{name}": value for name, value in train_metrics.items()})
        selection_value = train_metrics["reconstruction_mse"]
        if val_loader is not None:
            val_metrics = run_epoch(
                model,
                val_loader,
                optimizer=None,
                normalizer=normalizer,
                args=args,
                generator=torch.Generator().manual_seed(args.seed + 1),
                device=device,
            )
            metrics.update({f"val_{name}": value for name, value in val_metrics.items()})
            selection_value = val_metrics["reconstruction_mse"]
        if selection_value < best_reconstruction:
            best_reconstruction = selection_value
            _save_checkpoint(
                args.save_dir / "best_reconstruction.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                model_config=model_config,
                normalizer=normalizer,
                feature_names=feature_names,
                args=args,
                metrics=metrics,
            )
        history.append(metrics)
        with (args.save_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(metrics) + "\n")
        print(
            " ".join(
                f"{name}={value:.6g}" if isinstance(value, float) else f"{name}={value}"
                for name, value in metrics.items()
            )
        )
        if args.save_every > 0 and epoch % args.save_every == 0:
            _save_checkpoint(
                args.save_dir / f"epoch_{epoch}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                model_config=model_config,
                normalizer=normalizer,
                feature_names=feature_names,
                args=args,
                metrics=metrics,
            )

    _save_checkpoint(
        args.save_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        model_config=model_config,
        normalizer=normalizer,
        feature_names=feature_names,
        args=args,
        metrics=history[-1],
    )
    print(f"Finished in {time.time() - start:.1f}s. Outputs written to {args.save_dir}")


if __name__ == "__main__":
    main()
