"""Train Transolver++ to forecast future ASCOT profile fields."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .dataloader import (
    DEFAULT_ANALYSIS_FILENAME,
    DEFAULT_BFIELD_FILENAME,
    DEFAULT_EQUILIBRIUM_FILENAME,
    Ascot5Dataset,
)
from .time_dependent import TemporalWindowDataset, profiles_to_node_channels
from .train_transolver import (
    TransolverPlusModel,
    _bfield_channels,
    _cleanup_distributed,
    _coordinate_features,
    _discover_sample_folders,
    _ensure_distributed,
    _identity_collate,
    _pad_node_tensors,
    patch_transolver_attention_for_cuda,
)


def sample_to_temporal_tensors(
    sample: Dict[str, Any],
    *,
    max_nodes: int | None,
    profile_log1p: bool,
    generator: torch.Generator,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Build node inputs, positions, and future profile targets for one window."""

    grid_shape = tuple(sample["bfield"]["br"].shape)
    coords = _coordinate_features(sample, grid_shape)
    input_profiles = profiles_to_node_channels(
        sample["input_prs_para"],
        sample["input_prs_perp"],
        transform=profile_log1p,
    )
    target = profiles_to_node_channels(
        sample["target_prs_para"],
        sample["target_prs_perp"],
        transform=profile_log1p,
    )
    expected_nodes = int(torch.tensor(grid_shape).prod())
    if input_profiles.shape[0] != expected_nodes:
        raise ValueError(
            f"Profile grid {grid_shape} has {expected_nodes} nodes, but profile data "
            f"produced {input_profiles.shape[0]} nodes."
        )

    x = torch.cat(
        (coords, input_profiles, _bfield_channels(sample, grid_shape)),
        dim=-1,
    )
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

    if max_nodes is not None and x.shape[0] > max_nodes:
        indices = torch.randperm(x.shape[0], generator=generator)[:max_nodes]
        x = x[indices]
        coords = coords[indices]
        target = target[indices]
    return x, coords, target


def make_temporal_batch(
    samples: Sequence[Dict[str, Any]],
    *,
    max_nodes: int | None,
    profile_log1p: bool,
    generator: torch.Generator,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    xs, positions, targets = [], [], []
    for sample in samples:
        x, pos, target = sample_to_temporal_tensors(
            sample,
            max_nodes=max_nodes,
            profile_log1p=profile_log1p,
            generator=generator,
        )
        xs.append(x)
        positions.append(pos)
        targets.append(target)

    x_batch, mask = _pad_node_tensors(xs)
    pos_batch, _ = _pad_node_tensors(positions)
    target_batch, _ = _pad_node_tensors(targets)
    return (
        x_batch.to(device),
        pos_batch.to(device),
        mask.to(device),
        target_batch.to(device),
    )


def predict_node_profiles(
    model: torch.nn.Module,
    x: Tensor,
    pos: Tensor,
    mask: Tensor,
) -> Tensor:
    """Predict per-node future profile channels and zero padded nodes."""

    prediction = model((x, pos, None))
    return prediction * mask.unsqueeze(-1).to(prediction.dtype)


def masked_field_metrics(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tuple[Tensor, Tensor]:
    weights = mask.unsqueeze(-1).to(prediction.dtype)
    count = (weights.sum() * prediction.shape[-1]).clamp_min(1.0)
    error = prediction - target
    mse = (error.square() * weights).sum() / count
    mae = (error.abs() * weights).sum() / count
    return mse, mae


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: List[float] = []
    maes: List[float] = []
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for samples in loader:
            x, pos, mask, target = make_temporal_batch(
                samples,
                max_nodes=args.max_nodes,
                profile_log1p=not args.no_profile_log1p,
                generator=generator,
                device=device,
            )
            prediction = predict_node_profiles(model, x, pos, mask)
            loss, mae = masked_field_metrics(prediction, target, mask)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            maes.append(float(mae.detach().cpu()))

    return {
        "mse": sum(losses) / max(len(losses), 1),
        "mae": sum(maes) / max(len(maes), 1),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/global/cfs/cdirs/m5300/results/G1600"),
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("runs/transolver_alpha_timedependent"),
    )
    parser.add_argument("--analysis-filename", default=DEFAULT_ANALYSIS_FILENAME)
    parser.add_argument("--equilibrium-filename", default=DEFAULT_EQUILIBRIUM_FILENAME)
    parser.add_argument("--bfield-filename", default=DEFAULT_BFIELD_FILENAME)
    parser.add_argument("--input-frames", type=int, default=1)
    parser.add_argument("--output-frames", type=int, default=1)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Spacing in stored time indices between consecutive window frames.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=1,
        help="Spacing in stored time indices between training-window starts.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-nodes", type=int, default=16384)
    parser.add_argument(
        "--no-profile-log1p",
        action="store_true",
        help="Disable signed log1p on both input and target profile fields.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--slice-num", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp-ratio", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _split_folders(
    folders: Sequence[Path], train_fraction: float, seed: int
) -> Tuple[List[Path], List[Path]]:
    shuffled = list(folders)
    random.Random(seed).shuffle(shuffled)
    if train_fraction >= 1.0 or len(shuffled) == 1:
        return shuffled, []
    split = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_fraction)))
    return shuffled[:split], shuffled[split:]


def _build_window_dataset(
    folders: Sequence[Path], args: argparse.Namespace
) -> TemporalWindowDataset:
    simulations = Ascot5Dataset(
        folders,
        analysis_filename=args.analysis_filename,
        equilibrium_filename=args.equilibrium_filename,
        bfield_filename=args.bfield_filename,
        include_bfield=True,
        include_target=False,
        strict=True,
    )
    return TemporalWindowDataset(
        simulations,
        input_frames=args.input_frames,
        output_frames=args.output_frames,
        frame_stride=args.frame_stride,
        window_stride=args.window_stride,
    )


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_config: Dict[str, Any],
    args: argparse.Namespace,
    metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config,
            "args": vars(args),
            "metrics": metrics,
            "feature_layout": {
                "input": [
                    "rho",
                    "theta",
                    "phi",
                    *[
                        (
                            f"{field}_t"
                            if frame == args.input_frames - 1
                            else (
                                f"{field}_t-"
                                f"{(args.input_frames - frame - 1) * args.frame_stride}"
                            )
                        )
                        for frame in range(args.input_frames)
                        for field in ("prs_para", "prs_perp")
                    ],
                    "br",
                    "bphi",
                    "bz",
                ],
                "output": [
                    f"{field}_t+{(frame + 1) * args.frame_stride}"
                    for frame in range(args.output_frames)
                    for field in ("prs_para", "prs_perp")
                ],
            },
        },
        path,
    )


def main() -> None:
    args = build_parser().parse_args()
    if not (0.0 < args.train_fraction <= 1.0):
        raise ValueError("--train-fraction must be in (0, 1].")
    if args.max_nodes is not None and args.max_nodes <= 0:
        raise ValueError("--max-nodes must be positive when provided.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    _ensure_distributed(device)
    patch_transolver_attention_for_cuda()

    folders = _discover_sample_folders(
        args.results_root.expanduser(),
        args.analysis_filename,
        args.equilibrium_filename,
        args.bfield_filename,
    )
    if args.max_samples is not None:
        folders = folders[: args.max_samples]
    train_folders, val_folders = _split_folders(folders, args.train_fraction, args.seed)
    train_dataset = _build_window_dataset(train_folders, args)
    val_dataset = _build_window_dataset(val_folders, args) if val_folders else None
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

    batch_generator = torch.Generator().manual_seed(args.seed)
    first_x, _, first_target = sample_to_temporal_tensors(
        train_dataset[0],
        max_nodes=min(args.max_nodes or 1024, 1024),
        profile_log1p=not args.no_profile_log1p,
        generator=batch_generator,
    )
    input_dim = int(first_x.shape[-1])
    output_dim = int(first_target.shape[-1])
    model_config = {
        "space_dim": input_dim,
        "fun_dim": 0,
        "out_dim": output_dim,
        "n_hidden": args.hidden_dim,
        "n_layers": args.layers,
        "n_head": args.heads,
        "slice_num": args.slice_num,
        "dropout": args.dropout,
        "mlp_ratio": args.mlp_ratio,
        "unified_pos": False,
    }
    model = TransolverPlusModel(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print(
        f"Training on {len(train_folders)} simulations / {len(train_dataset)} windows"
        + (
            f"; validating on {len(val_folders)} simulations / {len(val_dataset)} windows"
            if val_dataset is not None
            else ""
        )
        + f"; input_frames={args.input_frames}, output_frames={args.output_frames}, "
        f"input_dim={input_dim}, output_dim={output_dim}, device={device}."
    )

    if args.dry_run:
        x, pos, mask, target = make_temporal_batch(
            next(iter(train_loader)),
            max_nodes=args.max_nodes,
            profile_log1p=not args.no_profile_log1p,
            generator=batch_generator,
            device=device,
        )
        model.eval()
        with torch.no_grad():
            prediction = predict_node_profiles(model, x, pos, mask)
        print(
            f"dry_run x={tuple(x.shape)} pos={tuple(pos.shape)} "
            f"target={tuple(target.shape)} prediction={tuple(prediction.shape)}"
        )
        _cleanup_distributed()
        return

    args.save_dir.mkdir(parents=True, exist_ok=True)
    with (args.save_dir / "config.json").open("w") as file:
        json.dump({"args": vars(args), "model_config": model_config}, file, indent=2, default=str)

    best_val = float("inf")
    history: List[Dict[str, float]] = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            args=args,
            generator=batch_generator,
        )
        metrics = {
            "epoch": epoch,
            "train_mse": train_metrics["mse"],
            "train_mae": train_metrics["mae"],
        }
        if val_loader is not None:
            val_metrics = run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                args=args,
                generator=batch_generator,
            )
            metrics.update({"val_mse": val_metrics["mse"], "val_mae": val_metrics["mae"]})
            if val_metrics["mse"] < best_val:
                best_val = val_metrics["mse"]
                _save_checkpoint(
                    args.save_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    model_config,
                    args,
                    metrics,
                )

        history.append(metrics)
        with (args.save_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(metrics) + "\n")
        print(
            " ".join(
                f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
                for key, value in metrics.items()
            )
        )
        if args.save_every > 0 and epoch % args.save_every == 0:
            _save_checkpoint(
                args.save_dir / f"epoch_{epoch}.pt",
                model,
                optimizer,
                epoch,
                model_config,
                args,
                metrics,
            )

    _save_checkpoint(
        args.save_dir / "last.pt",
        model,
        optimizer,
        args.epochs,
        model_config,
        args,
        history[-1],
    )
    print(f"Finished in {time.time() - start:.1f}s. Checkpoints written to {args.save_dir}")
    _cleanup_distributed()


if __name__ == "__main__":
    main()
