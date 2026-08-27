"""Train Transolver++ on alpha_analysis profile-grid samples."""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from .dataloader import (
    DEFAULT_ANALYSIS_FILENAME,
    DEFAULT_BFIELD_FILENAME,
    DEFAULT_EQUILIBRIUM_FILENAME,
    DEFAULT_TARGET_DATABASE_KEY,
    Ascot5Dataset,
)

try:
    from models.Transolver_plus import (
        Physics_Attention_1D_Eidetic,
        Model as TransolverPlusModel,
        gumbel_softmax,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local install
    raise ModuleNotFoundError(
        "Could not import Transolver++. Run "
        "`PYTHON_BIN=/path/to/alpha_analysis/bin/python "
        "bash tools/install_transolver_plus.sh` first."
    ) from exc


_REDUCE_ATTENTION_ACROSS_RANKS = True


def _node_to_slice_tokens(x_mid: Tensor, slice_weights: Tensor) -> Tensor:
    if not x_mid.is_cuda:
        return torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights).contiguous()
    batch_size, num_heads, _, dim_head = x_mid.shape
    slice_num = slice_weights.shape[-1]
    out = x_mid.new_empty((batch_size, num_heads, slice_num, dim_head))
    for batch_index in range(batch_size):
        for head_index in range(num_heads):
            out[batch_index, head_index] = (
                slice_weights[batch_index, head_index].transpose(0, 1)
                @ x_mid[batch_index, head_index]
            )
    return out.contiguous()


def _slice_to_node_tokens(out_slice_token: Tensor, slice_weights: Tensor) -> Tensor:
    if not out_slice_token.is_cuda:
        return torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
    batch_size, num_heads, slice_num, dim_head = out_slice_token.shape
    num_nodes = slice_weights.shape[-2]
    out = out_slice_token.new_empty((batch_size, num_heads, num_nodes, dim_head))
    for batch_index in range(batch_size):
        for head_index in range(num_heads):
            out[batch_index, head_index] = (
                slice_weights[batch_index, head_index]
                @ out_slice_token[batch_index, head_index]
            )
    return out


def _patched_attention_forward(self: Physics_Attention_1D_Eidetic, x: Tensor) -> Tensor:
    batch_size, num_nodes, _ = x.shape

    x_mid = (
        self.in_project_x(x)
        .reshape(batch_size, num_nodes, self.heads, self.dim_head)
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    temperature = self.proj_temperature(x_mid) + self.bias
    temperature = torch.clamp(temperature, min=0.01)
    slice_weights = gumbel_softmax(self.in_project_slice(x_mid), temperature)
    slice_norm = slice_weights.sum(2)
    slice_token = _node_to_slice_tokens(x_mid, slice_weights)
    if _REDUCE_ATTENTION_ACROSS_RANKS:
        dist_nn.all_reduce(slice_norm, op=dist.ReduceOp.SUM)
        dist_nn.all_reduce(slice_token, op=dist.ReduceOp.SUM)
    slice_token = slice_token / (
        (slice_norm + 1.0e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
    )

    q_slice_token = self.to_q(slice_token)
    k_slice_token = self.to_k(slice_token)
    v_slice_token = self.to_v(slice_token)
    out_slice_token = F.scaled_dot_product_attention(
        q_slice_token,
        k_slice_token,
        v_slice_token,
    )

    context_tokens = getattr(self, "_context_tokens", None)
    if context_tokens is not None:
        # Reassemble the per-head physics slices so they can query the AFSI
        # context tokens with ordinary d_model multi-head cross-attention.
        physics_slices = out_slice_token.permute(0, 2, 1, 3).reshape(
            batch_size, out_slice_token.shape[2], -1
        )
        cross_output, _ = self.context_cross_attention(
            physics_slices,
            context_tokens,
            context_tokens,
            need_weights=False,
        )
        physics_slices = self.context_cross_norm(physics_slices + cross_output)
        out_slice_token = physics_slices.reshape(
            batch_size, out_slice_token.shape[2], self.heads, self.dim_head
        ).permute(0, 2, 1, 3).contiguous()

    out_x = _slice_to_node_tokens(out_slice_token, slice_weights)
    out_x = out_x.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, -1)
    return self.to_out(out_x)


def patch_transolver_attention_for_cuda(*, data_parallel: bool = False) -> None:
    """Patch attention and configure its process-group semantics.

    The upstream attention reduces slice tokens across the default process
    group to support node/domain decomposition. Standard DDP gives every rank
    different samples, so those activation reductions would incorrectly mix
    independent batches. In data-parallel mode DDP synchronizes gradients and
    attention remains local to each rank.
    """

    global _REDUCE_ATTENTION_ACROSS_RANKS
    _REDUCE_ATTENTION_ACROSS_RANKS = not data_parallel
    Physics_Attention_1D_Eidetic.forward = _patched_attention_forward


def _identity_collate(batch: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(batch)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ensure_distributed(device: torch.device) -> None:
    if not dist.is_available() or dist.is_initialized():
        return
    backend = "nccl" if device.type == "cuda" else "gloo"
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend, init_method="env://")
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=0,
        world_size=1,
    )


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _discover_sample_folders(
    root: Path,
    analysis_filename: str,
    equilibrium_filename: str,
    bfield_filename: str,
) -> List[Path]:
    folders = []
    for path in sorted(root.iterdir()):
        try:
            if (
                path.is_dir()
                and (path / analysis_filename).is_file()
                and (path / equilibrium_filename).is_file()
                and (path / bfield_filename).is_file()
            ):
                folders.append(path)
        except PermissionError:
            continue
    if not folders:
        raise ValueError(f"No sample folders with required HDF5 files found under {root}")
    return folders


def _default_target_database_path(results_root: Path) -> Path | None:
    database_path = results_root / "G1600_end_database.json"
    return database_path if database_path.is_file() else None


def _make_index_coordinates(grid_shape: Sequence[int]) -> Tensor:
    axes = [torch.linspace(-1.0, 1.0, steps=size) for size in grid_shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)


def _normalize_coordinate_columns(coords: Tensor) -> Tensor:
    coords = torch.nan_to_num(coords.float(), nan=0.0, posinf=0.0, neginf=0.0)
    min_values = coords.amin(dim=0, keepdim=True)
    max_values = coords.amax(dim=0, keepdim=True)
    span = (max_values - min_values).clamp_min(1.0e-12)
    return 2.0 * (coords - min_values) / span - 1.0


def _coordinate_features(sample: Dict[str, Any], grid_shape: Sequence[int]) -> Tensor:
    bfield = sample["bfield"]
    has_coordinates = all(
        name in bfield and tuple(bfield[name].shape) == tuple(grid_shape)
        for name in ("rho", "theta", "phi")
    )
    if has_coordinates:
        coords = torch.stack(
            [bfield[name].reshape(-1) for name in ("rho", "theta", "phi")],
            dim=-1,
        )
        return _normalize_coordinate_columns(coords)
    return _make_index_coordinates(grid_shape)


def _profile_channels(tensor: Tensor, grid_shape: Sequence[int], *, log1p: bool) -> Tensor:
    if tuple(tensor.shape[-len(grid_shape) :]) != tuple(grid_shape):
        raise ValueError(
            f"Expected trailing profile shape {tuple(grid_shape)}, received {tuple(tensor.shape)}"
        )
    channels = tensor.reshape(-1, *grid_shape).float()
    if log1p:
        channels = torch.sign(channels) * torch.log1p(torch.abs(channels))
    return channels.reshape(channels.shape[0], -1).transpose(0, 1)


def _bfield_channels(sample: Dict[str, Any], grid_shape: Sequence[int]) -> Tensor:
    channels = []
    for name in ("br", "bphi", "bz"):
        value = sample["bfield"][name]
        if tuple(value.shape) != tuple(grid_shape):
            raise ValueError(
                f"Expected bfield/{name} shape {tuple(grid_shape)}, received {tuple(value.shape)}"
            )
        channels.append(value.reshape(-1).float())
    return torch.stack(channels, dim=-1)


def _reduce_target(target: Tensor, reduction: str, *, log10_target: bool, eps: float) -> Tensor:
    finite = target[torch.isfinite(target)]
    #these remove large outliers, needed for 00292
    finite = torch.where(finite > 1.0, torch.zeros_like(finite), finite)
    if finite.numel() == 0:
        value = torch.tensor(0.0, dtype=torch.float32)
    elif reduction == "mean":
        value = finite.mean()
    elif reduction == "sum":
        value = finite.sum()
    elif reduction == "max":
        value = finite.max()
    else:
        raise ValueError(f"Unsupported target reduction: {reduction}")
    value = value.float().clamp_min(0.0)
    if log10_target:
        value = torch.log10(value + eps)
    return value


def sample_to_transolver_tensors(
    sample: Dict[str, Any],
    *,
    max_nodes: int | None,
    target_reduction: str,
    log10_target: bool,
    target_eps: float,
    profile_log1p: bool,
    generator: torch.Generator,
) -> Tuple[Tensor, Tensor, Tensor]:
    grid_shape = tuple(sample["bfield"]["br"].shape)
    coords = _coordinate_features(sample, grid_shape)
    features = [
        coords,
        _profile_channels(sample["prs_para"], grid_shape, log1p=profile_log1p),
        _profile_channels(sample["prs_perp"], grid_shape, log1p=profile_log1p),
        _bfield_channels(sample, grid_shape),
    ]
    x = torch.cat(features, dim=-1)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    pos = coords

    if max_nodes is not None and x.shape[0] > max_nodes:
        indices = torch.randperm(x.shape[0], generator=generator)[:max_nodes]
        x = x[indices]
        pos = pos[indices]

    y = _reduce_target(
        sample["target"],
        target_reduction,
        log10_target=log10_target,
        eps=target_eps,
    )
    return x, pos, y


def _pad_node_tensors(items: Sequence[Tensor], pad_value: float = 0.0) -> Tuple[Tensor, Tensor]:
    max_nodes = max(item.shape[0] for item in items)
    feature_dim = items[0].shape[1]
    batch = torch.full((len(items), max_nodes, feature_dim), pad_value, dtype=items[0].dtype)
    mask = torch.zeros((len(items), max_nodes), dtype=torch.bool)
    for index, item in enumerate(items):
        if item.shape[1] != feature_dim:
            raise ValueError(
                f"All node tensors must have feature dim {feature_dim}; got {item.shape[1]}"
            )
        batch[index, : item.shape[0]] = item
        mask[index, : item.shape[0]] = True
    return batch, mask


def make_transolver_batch(
    samples: Sequence[Dict[str, Any]],
    *,
    max_nodes: int | None,
    target_reduction: str,
    log10_target: bool,
    target_eps: float,
    profile_log1p: bool,
    generator: torch.Generator,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    xs, positions, targets = [], [], []
    for sample in samples:
        x, pos, target = sample_to_transolver_tensors(
            sample,
            max_nodes=max_nodes,
            target_reduction=target_reduction,
            log10_target=log10_target,
            target_eps=target_eps,
            profile_log1p=profile_log1p,
            generator=generator,
        )
        xs.append(x)
        positions.append(pos)
        targets.append(target)

    x_batch, mask = _pad_node_tensors(xs)
    pos_batch, _ = _pad_node_tensors(positions)
    y_batch = torch.stack(targets)
    return (
        x_batch.to(device),
        pos_batch.to(device),
        mask.to(device),
        y_batch.to(device),
    )


def _predict_sample_scalar(model: torch.nn.Module, x: Tensor, pos: Tensor, mask: Tensor) -> Tensor:
    node_values = model((x, pos, None)).squeeze(-1)
    weights = mask.float()
    return (node_values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


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
            x, pos, mask, target = make_transolver_batch(
                samples,
                max_nodes=args.max_nodes,
                target_reduction=args.target_reduction,
                log10_target=not args.no_log_target,
                target_eps=args.target_eps,
                profile_log1p=not args.no_profile_log1p,
                generator=generator,
                device=device,
            )
            prediction = _predict_sample_scalar(model, x, pos, mask)
            loss = F.mse_loss(prediction, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            maes.append(float((prediction - target).abs().mean().detach().cpu()))

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
    parser.add_argument("--save-dir", type=Path, default=Path("runs/transolver_alpha"))
    parser.add_argument("--analysis-filename", default=DEFAULT_ANALYSIS_FILENAME)
    parser.add_argument("--equilibrium-filename", default=DEFAULT_EQUILIBRIUM_FILENAME)
    parser.add_argument("--bfield-filename", default=DEFAULT_BFIELD_FILENAME)
    parser.add_argument(
        "--target-database",
        type=Path,
        default=None,
        help=(
            "JSON database containing scalar sample targets. Defaults to "
            "'<results-root>/G1600_end_database.json' when that file exists."
        ),
    )
    parser.add_argument("--target-database-key", default=DEFAULT_TARGET_DATABASE_KEY)
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
    parser.add_argument("--target-reduction", choices=("mean", "sum", "max"), default="mean")
    parser.add_argument("--target-eps", type=float, default=1.0e-30)
    parser.add_argument(
        "--log-target",
        action="store_false",
        dest="no_log_target",
        help="Apply log10(target + --target-eps). By default targets are used directly.",
    )
    parser.set_defaults(no_log_target=True)
    parser.add_argument("--no-profile-log1p", action="store_true")
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


def _split_indices(num_items: int, train_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    indices = list(range(num_items))
    random.Random(seed).shuffle(indices)
    if train_fraction >= 1.0:
        return indices, []
    split = (
        max(1, min(num_items - 1, int(num_items * train_fraction)))
        if num_items > 1
        else num_items
    )
    return indices[:split], indices[split:]


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

    results_root = args.results_root.expanduser()
    target_database_path = (
        args.target_database.expanduser()
        if args.target_database is not None
        else _default_target_database_path(results_root)
    )
    args.target_database = target_database_path

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
        strict=True,
        target_database_path=target_database_path,
        target_database_key=args.target_database_key,
    )
    train_indices, val_indices = _split_indices(len(dataset), args.train_fraction, args.seed)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_indices else None

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
    first_sample = dataset[train_indices[0]]
    first_x, _, _ = sample_to_transolver_tensors(
        first_sample,
        max_nodes=min(args.max_nodes or 1024, 1024),
        target_reduction=args.target_reduction,
        log10_target=not args.no_log_target,
        target_eps=args.target_eps,
        profile_log1p=not args.no_profile_log1p,
        generator=batch_generator,
    )
    input_dim = int(first_x.shape[-1])
    model_config = {
        "space_dim": input_dim,
        "fun_dim": 0,
        "out_dim": 1,
        "n_hidden": args.hidden_dim,
        "n_layers": args.layers,
        "n_head": args.heads,
        "slice_num": args.slice_num,
        "dropout": args.dropout,
        "mlp_ratio": args.mlp_ratio,
        "unified_pos": False,
    }
    model = TransolverPlusModel(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        "Training Transolver++ on "
        f"{len(train_dataset)} train samples"
        + (f" and {len(val_dataset)} validation samples" if val_dataset is not None else "")
        + f"; input_dim={input_dim}, device={device}"
        + (
            f", target_database={target_database_path}, "
            f"target_key={args.target_database_key}"
            if target_database_path is not None
            else ", target=hdf5"
        )
        + "."
    )

    if args.dry_run:
        samples = next(iter(train_loader))
        x, pos, mask, target = make_transolver_batch(
            samples,
            max_nodes=args.max_nodes,
            target_reduction=args.target_reduction,
            log10_target=not args.no_log_target,
            target_eps=args.target_eps,
            profile_log1p=not args.no_profile_log1p,
            generator=batch_generator,
            device=device,
        )
        model.eval()
        with torch.no_grad():
            prediction = _predict_sample_scalar(model, x, pos, mask)
        print(
            f"dry_run x={tuple(x.shape)} pos={tuple(pos.shape)} "
            f"target={target.detach().cpu().tolist()} "
            f"prediction={prediction.detach().cpu().tolist()}"
        )
        _cleanup_distributed()
        return

    args.save_dir.mkdir(parents=True, exist_ok=True)
    with (args.save_dir / "config.json").open("w") as file:
        json.dump({"args": vars(args), "model_config": model_config}, file, indent=2, default=str)

    history = []
    best_val = float("inf")
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
