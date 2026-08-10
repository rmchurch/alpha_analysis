"""Train the deterministic per-frame Transolver autoencoder."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
from torch import nn
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .frame_dataset import (
    AscotFrameDataset,
    AscotSequenceDataset,
    LEGACY_TRAINING_SAMPLE_COUNT,
    discover_simulation_folders,
    fit_frame_normalization,
    split_simulation_cohorts,
)
from .transolver_autoencoder import (
    AutoencoderConfig,
    TransolverFrameAutoencoder,
    anti_collapse_losses,
    autoencoder_loss,
)


def parse_optional_int(value: str) -> int | None:
    return None if value.lower() in {"none", "null", "full"} else int(value)


def _distributed_context(requested_device: str) -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        backend = "nccl" if requested_device.startswith("cuda") else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device(requested_device)
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        device = torch.device(requested_device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", 0)
            torch.cuda.set_device(device)
    return rank, local_rank, world_size, device


def _broadcast_normalization(
    normalization: Any | None, device: torch.device, rank: int, world_size: int
) -> Any:
    if world_size == 1:
        if normalization is None:
            raise RuntimeError("Normalization was not computed")
        return normalization
    if rank == 0:
        packed = torch.cat(
            [
                normalization.profile_mean,
                normalization.profile_std,
                normalization.bfield_mean,
                normalization.bfield_std,
                normalization.coordinate_mean,
                normalization.coordinate_std,
            ]
        ).to(device)
    else:
        packed = torch.empty(16, dtype=torch.float32, device=device)
    dist.broadcast(packed, src=0)
    packed = packed.cpu()
    from .frame_dataset import FrameNormalization

    return FrameNormalization(
        profile_mean=packed[0:2],
        profile_std=packed[2:4],
        bfield_mean=packed[4:7],
        bfield_std=packed[7:10],
        coordinate_mean=packed[10:13],
        coordinate_std=packed[13:16],
    )


def _base_model(model: nn.Module) -> TransolverFrameAutoencoder:
    return model.module if isinstance(model, DistributedDataParallel) else model


def frame_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    max_nodes = max(item["profile"].shape[0] for item in batch)

    def pad(value: Tensor) -> Tensor:
        result = value.new_zeros((max_nodes, value.shape[-1]))
        result[: value.shape[0]] = value
        return result

    mask = torch.zeros(len(batch), max_nodes, dtype=torch.bool)
    for index, item in enumerate(batch):
        mask[index, : item["profile"].shape[0]] = True
    context = {}
    for key in ("R_lmn", "Z_lmn"):
        lengths = [item["context"][key].numel() for item in batch]
        context[key] = torch.zeros(len(batch), max(lengths))
        for i, item in enumerate(batch):
            value = item["context"][key].reshape(-1)
            context[key][i, : value.numel()] = value
    return {
        "profile": torch.stack([pad(item["profile"]) for item in batch]),
        "coordinates": torch.stack([pad(item["coordinates"]) for item in batch]),
        "bfield": torch.stack([pad(item["bfield"]) for item in batch]),
        "time": torch.stack([item["time"] for item in batch]),
        "node_mask": mask,
        "context": context,
        "items": list(batch),
    }


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: (
            {name: value.to(device, non_blocking=True) for name, value in item.items()}
            if key == "context"
            else item.to(device, non_blocking=True)
            if isinstance(item, Tensor)
            else item
        )
        for key, item in batch.items()
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument(
        "--training-sample-count",
        type=int,
        default=LEGACY_TRAINING_SAMPLE_COUNT,
        help="Immutable initial cohort size; later folders are prediction-only (default: 578).",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-nodes", type=parse_optional_int, default=None)
    parser.add_argument(
        "--full-mesh-validation", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--encoder-heads", type=int, default=4)
    parser.add_argument("--encoder-slice-num", type=int, default=32)
    parser.add_argument("--latent-tokens", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-heads", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=1)
    parser.add_argument("--field-loss", choices=("huber",), default="huber")
    parser.add_argument("--integral-weight", type=float, default=0.01)
    parser.add_argument("--slice-balance-weight", type=float, default=0.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    integral_weight: float,
    balance_weight: float,
    grad_clip: float,
    dry_run: bool,
    world_size: int,
) -> dict[str, float]:
    model.train(optimizer is not None)
    totals: dict[str, float] = {
        key: 0.0 for key in ("total", "field", "integral", "parallel", "perpendicular")
    }
    count = 0
    for raw in loader:
        batch = _to_device(raw, device)
        with torch.set_grad_enabled(optimizer is not None):
            encoded, reconstruction = model(
                batch["profile"],
                batch["coordinates"],
                batch["bfield"],
                batch["time"],
                batch["context"],
                batch["node_mask"],
            )
            total, metrics = autoencoder_loss(
                reconstruction, batch["profile"], batch["node_mask"], integral_weight
            )
            collapse = anti_collapse_losses(encoded.latent, encoded.slice_norms)
            total = total + balance_weight * collapse["balance"]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        values = {"total": total, **metrics}
        for key in totals:
            totals[key] += float(values[key].detach())
        count += 1
        if dry_run:
            break
    keys = tuple(totals)
    reduced = torch.tensor(
        [*(totals[key] for key in keys), float(count)],
        dtype=torch.float64,
        device=device,
    )
    if world_size > 1:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    global_count = max(float(reduced[-1]), 1.0)
    return {key: float(reduced[index] / global_count) for index, key in enumerate(keys)}


def main() -> None:
    args = build_parser().parse_args()
    rank, local_rank, world_size, device = _distributed_context(args.device)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)
    folders = discover_simulation_folders(args.results_root)
    splits = split_simulation_cohorts(
        folders,
        training_sample_count=args.training_sample_count,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    sequences = AscotSequenceDataset(folders, include_target=False)
    normalization = (
        fit_frame_normalization(sequences, splits["train"]) if rank == 0 else None
    )
    normalization = _broadcast_normalization(normalization, device, rank, world_size)
    train = AscotFrameDataset(
        sequences,
        splits["train"],
        normalization=normalization,
        max_nodes=args.max_nodes,
        seed=args.seed,
        training=True,
    )
    val_max_nodes = None if args.full_mesh_validation else args.max_nodes
    validation = AscotFrameDataset(
        sequences,
        splits["val"],
        normalization=normalization,
        max_nodes=val_max_nodes,
        seed=args.seed,
    )
    first = sequences.load_frame(splits["train"][0], 0)
    original_nodes = first["profile"].shape[0]
    if rank == 0:
        print(
            f"distributed: world_size={world_size} workers_per_rank={args.num_workers} "
            f"total_workers={world_size * args.num_workers}"
        )
        print(
            f"simulations: train={len(splits['train'])} val={len(splits['val'])} later={len(splits['later'])}"
        )
        print(
            f"nodes: original={original_nodes} train-used={args.max_nodes or original_nodes} "
            f"validation-used={'full' if val_max_nodes is None else val_max_nodes}"
        )
    train_sampler = (
        DistributedSampler(
            train,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        if world_size > 1
        else None
    )
    val_sampler = (
        DistributedSampler(
            validation,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        if world_size > 1
        else None
    )
    loader_options = {
        "num_workers": args.num_workers,
        "collate_fn": frame_collate,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_options["prefetch_factor"] = 2
    train_loader = DataLoader(
        train,
        args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_options,
    )
    val_loader = DataLoader(
        validation,
        args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        **loader_options,
    )
    config = AutoencoderConfig(
        args.hidden_dim,
        args.encoder_layers,
        args.encoder_heads,
        args.encoder_slice_num,
        args.latent_tokens,
        args.latent_dim,
        1,
        args.decoder_hidden_dim,
        args.decoder_heads,
        args.decoder_layers,
    )
    model = TransolverFrameAutoencoder(config).to(device)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), args.lr, weight_decay=args.weight_decay
    )
    if rank == 0:
        args.save_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    config_payload = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model_config": asdict(config),
        "time_convention": normalization.time_convention,
        "cohort_policy": {
            "training_sample_count": args.training_sample_count,
            "later_simulations_are_prediction_only": True,
        },
        "distributed": {
            "world_size": world_size,
            "workers_per_rank": args.num_workers,
        },
    }
    if rank == 0:
        (args.save_dir / "config.json").write_text(json.dumps(config_payload, indent=2))
        for name in ("train", "val", "later"):
            (args.save_dir / f"{name}_folders.txt").write_text(
                "\n".join(str(folders[i]) for i in splits[name]) + "\n"
            )
    best_field = best_total = float("inf")
    metrics_path = args.save_dir / "metrics.jsonl"
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        train_metrics = _epoch(
            model,
            train_loader,
            device,
            optimizer,
            args.integral_weight,
            args.slice_balance_weight,
            args.grad_clip,
            args.dry_run,
            world_size,
        )
        validation_metrics = _epoch(
            model,
            val_loader,
            device,
            None,
            args.integral_weight,
            0.0,
            args.grad_clip,
            args.dry_run,
            world_size,
        )
        record = {"epoch": epoch, "train": train_metrics, "val": validation_metrics}
        if rank == 0:
            with metrics_path.open("a") as file:
                file.write(json.dumps(record) + "\n")
            checkpoint = {
                "model_state_dict": _base_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": asdict(config),
                "normalization": normalization.state_dict(),
                "split_indices": splits,
                "epoch": epoch,
                "metrics": record,
                "training_sample_count": args.training_sample_count,
                "world_size": world_size,
            }
            torch.save(checkpoint, args.save_dir / "last.pt")
            if validation_metrics["field"] < best_field:
                best_field = validation_metrics["field"]
                torch.save(checkpoint, args.save_dir / "best_field.pt")
            if validation_metrics["total"] < best_total:
                best_total = validation_metrics["total"]
                torch.save(checkpoint, args.save_dir / "best_total.pt")
            print(json.dumps(record), flush=True)
        if args.dry_run:
            break

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
