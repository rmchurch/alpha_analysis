"""Export internal Transolver++ eidetic slice tokens for train/validation samples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from .dataloader import (
    DEFAULT_ANALYSIS_FILENAME,
    DEFAULT_BFIELD_FILENAME,
    DEFAULT_EQUILIBRIUM_FILENAME,
    DEFAULT_TARGET_DATABASE_KEY,
    Ascot5Dataset,
)
from .train_transolver import (
    _bfield_channels,
    _coordinate_features,
    _discover_sample_folders,
    _ensure_distributed,
    _node_to_slice_tokens,
    _profile_channels,
    _read_folder_manifest,
    _reduce_target,
    _slice_to_node_tokens,
    _split_indices,
)

try:
    from models.Transolver_plus import (
        Model as TransolverPlusModel,
        Physics_Attention_1D_Eidetic,
        gumbel_softmax,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local install
    raise ModuleNotFoundError(
        "Could not import Transolver++. Run "
        "`PYTHON_BIN=/path/to/alpha_analysis/bin/python "
        "bash tools/install_transolver_plus.sh` first."
    ) from exc


TOKEN_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _default_run_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = sorted(
        (repo_root / "runs" / "transolver_alpha").glob("*/config.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].parent
    return Path("runs/transolver_alpha")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text())


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _path_aliases(path: Path) -> list[Path]:
    expanded = path.expanduser()
    aliases = [expanded]
    text = str(expanded)
    if text.startswith("/global/cfs/cdirs/"):
        aliases.append(Path(text.replace("/global/cfs/cdirs/", "/global/cfs/projectdirs/", 1)))
    if text.startswith("/global/cfs/projectdirs/"):
        aliases.append(Path(text.replace("/global/cfs/projectdirs/", "/global/cfs/cdirs/", 1)))
    return aliases


def _resolve_results_root(saved_root: str, override: Path | None) -> Path:
    candidates: list[Path] = []
    if override is not None:
        candidates.extend(_path_aliases(override))
    candidates.extend(_path_aliases(Path(saved_root)))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find results root. Tried: " + ", ".join(str(item) for item in candidates)
    )


def _resolve_target_database(saved_path: str | None, results_root: Path) -> Path | None:
    if saved_path is None:
        candidate = results_root / "G1600_end_database.json"
        return candidate if candidate.is_file() else None

    candidates = _path_aliases(Path(saved_path))
    candidates.append(results_root / Path(saved_path).name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find target database. Tried: " + ", ".join(str(item) for item in candidates)
    )


def _read_folder_list(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing folders file: {path}")
    folders = []
    for line in path.read_text().splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        folders.append(Path(token).expanduser())
    if not folders:
        raise ValueError(f"No sample folders found in {path}")
    return folders


def _all_reduce_sum(tensor: Tensor) -> Tensor:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def install_slice_token_capture(*, deterministic_slices: bool) -> None:
    """Patch Transolver++ attention to retain eidetic slice tokens after forward."""

    def _capturing_attention_forward(self: Physics_Attention_1D_Eidetic, x: Tensor) -> Tensor:
        batch_size, num_nodes, _ = x.shape

        x_mid = (
            self.in_project_x(x)
            .reshape(batch_size, num_nodes, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        temperature = self.proj_temperature(x_mid) + self.bias
        temperature = torch.clamp(temperature, min=0.01)
        slice_logits = self.in_project_slice(x_mid)
        if deterministic_slices:
            slice_weights = F.softmax(slice_logits / temperature, dim=-1)
        else:
            slice_weights = gumbel_softmax(slice_logits, temperature)

        slice_norm = _all_reduce_sum(slice_weights.sum(2))
        slice_token = _node_to_slice_tokens(x_mid, slice_weights)
        slice_token = _all_reduce_sum(slice_token)
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

        self._captured_slice_token = slice_token.detach()
        self._captured_out_slice_token = out_slice_token.detach()
        self._captured_slice_norm = slice_norm.detach()

        out_x = _slice_to_node_tokens(out_slice_token, slice_weights)
        out_x = out_x.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, -1)
        return self.to_out(out_x)

    Physics_Attention_1D_Eidetic.forward = _capturing_attention_forward


def _sample_to_tensors_with_indices(
    sample: dict[str, Any],
    *,
    max_nodes: int | None,
    target_reduction: str,
    log10_target: bool,
    target_eps: float,
    profile_log1p: bool,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor, tuple[int, ...]]:
    grid_shape = tuple(int(size) for size in sample["bfield"]["br"].shape)
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
    node_indices = torch.arange(x.shape[0], dtype=torch.long)

    if max_nodes is not None and x.shape[0] > max_nodes:
        node_indices = torch.randperm(x.shape[0], generator=generator)[:max_nodes]
        x = x[node_indices]
        pos = pos[node_indices]

    y = _reduce_target(
        sample["target"],
        target_reduction,
        log10_target=log10_target,
        eps=target_eps,
    )
    return x, pos, y, node_indices, grid_shape


def _attention_modules(
    model: torch.nn.Module,
) -> list[tuple[str, Physics_Attention_1D_Eidetic]]:
    modules = []
    for name, module in model.named_modules():
        if isinstance(module, Physics_Attention_1D_Eidetic):
            modules.append((name, module))
    if not modules:
        raise ValueError("No Physics_Attention_1D_Eidetic modules found in model.")
    return modules


def _collect_captured_tokens(
    modules: Sequence[tuple[str, Physics_Attention_1D_Eidetic]],
    *,
    token_dtype: torch.dtype,
) -> dict[str, Tensor | list[str]]:
    names = []
    slice_tokens = []
    out_slice_tokens = []
    slice_norms = []
    for name, module in modules:
        if not hasattr(module, "_captured_slice_token"):
            raise RuntimeError(f"Attention module did not capture tokens: {name}")
        names.append(name)
        slice_tokens.append(module._captured_slice_token.detach().squeeze(0).cpu().to(token_dtype))
        out_slice_tokens.append(
            module._captured_out_slice_token.detach().squeeze(0).cpu().to(token_dtype)
        )
        slice_norms.append(module._captured_slice_norm.detach().squeeze(0).cpu().to(torch.float32))
    return {
        "attention_module_names": names,
        "slice_tokens": torch.stack(slice_tokens, dim=0),
        "out_slice_tokens": torch.stack(out_slice_tokens, dim=0),
        "slice_norms": torch.stack(slice_norms, dim=0),
    }


def _profile_channel_count(tensor: Tensor, grid_shape: Sequence[int]) -> int:
    return int(tensor.reshape(-1, *grid_shape).shape[0])


def _feature_layout(sample: dict[str, Any]) -> dict[str, Any]:
    grid_shape = tuple(int(size) for size in sample["bfield"]["br"].shape)
    para_count = _profile_channel_count(sample["prs_para"], grid_shape)
    perp_count = _profile_channel_count(sample["prs_perp"], grid_shape)
    offset = 0
    layout: dict[str, Any] = {
        "grid_shape": list(grid_shape),
        "channels": {},
    }
    layout["channels"]["coords"] = [offset, offset + 3]
    offset += 3
    layout["channels"]["prs_para"] = [offset, offset + para_count]
    offset += para_count
    layout["channels"]["prs_perp"] = [offset, offset + perp_count]
    offset += perp_count
    layout["channels"]["bfield"] = [offset, offset + 3]
    offset += 3
    layout["total_feature_dim"] = offset
    layout["profile_channel_counts"] = {
        "prs_para": para_count,
        "prs_perp": perp_count,
    }
    return layout


def _safe_sample_key(folder: str) -> str:
    return Path(folder).name.replace("/", "_")


def _sample_key_from_token_path(path: Path) -> str | None:
    stem = path.stem
    if "_" not in stem:
        return None
    prefix, key = stem.split("_", 1)
    if not prefix.isdigit():
        return None
    return key


def _existing_sample_keys(output_dir: Path) -> set[str]:
    keys: set[str] = set()
    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.is_file():
        for line in manifest_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            folder = record.get("folder")
            if isinstance(folder, str):
                keys.add(_safe_sample_key(folder))

    if output_dir.is_dir():
        for split_dir in output_dir.iterdir():
            if not split_dir.is_dir():
                continue
            for token_path in split_dir.glob("*.pt"):
                key = _sample_key_from_token_path(token_path)
                if key is not None:
                    keys.add(key)
    return keys


def _prepare_output_dir(output_dir: Path, *, overwrite: bool, allow_existing: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [item for item in output_dir.iterdir() if item.name not in {".DS_Store"}]
    if existing and not overwrite and not allow_existing:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new --output-dir or pass --overwrite."
        )
    if overwrite:
        for filename in ("manifest.jsonl", "metadata.json"):
            path = output_dir / filename
            if path.exists():
                path.unlink()


def _export_one_sample(
    *,
    model: torch.nn.Module,
    attention_modules: Sequence[tuple[str, Physics_Attention_1D_Eidetic]],
    sample: dict[str, Any],
    split: str,
    split_index: int,
    dataset_index: int,
    saved_args: dict[str, Any],
    max_nodes: int | None,
    token_dtype: torch.dtype,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(int(saved_args["seed"]) + int(dataset_index))
    x, pos, target, node_indices, grid_shape = _sample_to_tensors_with_indices(
        sample,
        max_nodes=max_nodes,
        target_reduction=saved_args["target_reduction"],
        log10_target=not saved_args["no_log_target"],
        target_eps=saved_args["target_eps"],
        profile_log1p=not saved_args["no_profile_log1p"],
        generator=generator,
    )

    with torch.no_grad():
        node_values = model((x.unsqueeze(0).to(device), pos.unsqueeze(0).to(device), None)).squeeze(-1)
        prediction = node_values.mean(dim=1).squeeze(0).detach().cpu()

    captured = _collect_captured_tokens(attention_modules, token_dtype=token_dtype)
    original_node_count = int(math.prod(grid_shape))
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    token_path = split_dir / f"{split_index:06d}_{_safe_sample_key(sample['folder'])}.pt"
    relative_token_path = token_path.relative_to(output_dir)

    payload = {
        "split": split,
        "split_index": split_index,
        "dataset_index": dataset_index,
        "folder": sample["folder"],
        "target": target.detach().cpu(),
        "prediction": prediction,
        "grid_shape": grid_shape,
        "node_count": int(x.shape[0]),
        "original_node_count": original_node_count,
        "node_indices": node_indices.cpu(),
        "attention_module_names": captured["attention_module_names"],
        "slice_tokens": captured["slice_tokens"],
        "out_slice_tokens": captured["out_slice_tokens"],
        "slice_norms": captured["slice_norms"],
    }
    torch.save(payload, token_path)

    return {
        "split": split,
        "split_index": split_index,
        "dataset_index": dataset_index,
        "folder": sample["folder"],
        "token_path": str(relative_token_path),
        "target": float(target.detach().cpu().item()),
        "prediction": float(prediction.item()),
        "node_count": int(x.shape[0]),
        "original_node_count": original_node_count,
        "grid_shape": list(grid_shape),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Training run directory containing config.json and checkpoints.",
    )
    parser.add_argument("--checkpoint", default="best.pt", help="Checkpoint filename or path.")
    parser.add_argument("--results-root", type=Path, help="Override results root from config.json.")
    parser.add_argument("--target-database", type=Path, help="Override target database from config.json.")
    parser.add_argument(
        "--folders-file",
        type=Path,
        help=(
            "Optional newline-delimited sample folders. Mainly useful for smoke tests or "
            "explicit partial exports; the train/val split is computed over this list."
        ),
    )
    parser.add_argument(
        "--train-folders",
        type=Path,
        help="Explicit training folder manifest; must be paired with --val-folders.",
    )
    parser.add_argument(
        "--val-folders",
        type=Path,
        help="Explicit validation folder manifest; must be paired with --train-folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for token .pt files, manifest.jsonl, and metadata.json.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val"),
        default=("train", "val"),
        help="Dataset splits to export.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        help="Debug limit applied independently to each selected split.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        help="Nodes sampled per sample. Defaults to the training checkpoint value.",
    )
    parser.add_argument(
        "--full-mesh",
        action="store_true",
        help="Use every node instead of the checkpoint/training max_nodes subsample.",
    )
    parser.add_argument(
        "--token-dtype",
        choices=tuple(TOKEN_DTYPES),
        default="float32",
        help="Storage dtype for slice token tensors.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--stochastic-slices",
        action="store_true",
        help="Use Transolver++ gumbel_softmax slice assignment instead of deterministic softmax.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    parser.add_argument(
        "--new-only",
        action="store_true",
        help=(
            "Export only sample folders not already present in the output manifest or token "
            "directories, and write them under --new-split-name."
        ),
    )
    parser.add_argument(
        "--new-split-name",
        default="new",
        help="Split directory name used with --new-only.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (args.train_folders is None) != (args.val_folders is None):
        raise ValueError("--train-folders and --val-folders must be provided together.")
    if args.folders_file is not None and args.train_folders is not None:
        raise ValueError("Use either --folders-file or explicit train/val manifests, not both.")
    run_dir = (args.run_dir or _default_run_dir()).expanduser().resolve()
    config = _load_json(run_dir / "config.json")
    saved_args = config["args"]

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_path
    checkpoint = _load_checkpoint(checkpoint_path)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = run_dir / f"{checkpoint_path.stem}_slice_tokens"
    output_dir = output_dir.expanduser().resolve()
    if args.new_only and (
        not args.new_split_name or Path(args.new_split_name).name != args.new_split_name
    ):
        raise ValueError("--new-split-name must be a non-empty directory name, not a path.")
    _prepare_output_dir(output_dir, overwrite=args.overwrite, allow_existing=args.new_only)

    results_root = _resolve_results_root(saved_args["results_root"], args.results_root)
    target_database_path = _resolve_target_database(
        str(args.target_database) if args.target_database is not None else saved_args.get("target_database"),
        results_root,
    )

    explicit_split = args.train_folders is not None and args.val_folders is not None
    if explicit_split:
        train_folders = _read_folder_manifest(args.train_folders.expanduser(), results_root)
        val_folders = _read_folder_manifest(args.val_folders.expanduser(), results_root)
        overlap = {folder.name for folder in train_folders} & {folder.name for folder in val_folders}
        if overlap:
            raise ValueError(
                "Training and validation manifests overlap; first duplicate: "
                f"{sorted(overlap)[0]}"
            )
        folders = train_folders + val_folders
    elif args.folders_file is not None:
        folders = _read_folder_list(args.folders_file.expanduser())
    else:
        folders = _discover_sample_folders(
            results_root,
            saved_args.get("analysis_filename", DEFAULT_ANALYSIS_FILENAME),
            saved_args.get("equilibrium_filename", DEFAULT_EQUILIBRIUM_FILENAME),
            saved_args.get("bfield_filename", DEFAULT_BFIELD_FILENAME),
            )
    if saved_args.get("max_samples") is not None:
        folders = folders[: saved_args["max_samples"]]

    discovered_sample_count = len(folders)
    existing_sample_count = None
    new_folder_records: list[tuple[int, Path]] | None = None
    if args.new_only:
        existing_keys = _existing_sample_keys(output_dir)
        existing_sample_count = len(existing_keys)
        new_folder_records = [
            (dataset_index, folder)
            for dataset_index, folder in enumerate(folders)
            if _safe_sample_key(str(folder)) not in existing_keys
        ]
        folders = [folder for _, folder in new_folder_records]
        print(
            "Found "
            f"{len(folders)} new samples out of {discovered_sample_count} discovered "
            f"({existing_sample_count} already exported)."
        )
        if not folders:
            print(f"No new samples to export under {output_dir}")
            return

    dataset = Ascot5Dataset(
        folders,
        analysis_filename=saved_args.get("analysis_filename", DEFAULT_ANALYSIS_FILENAME),
        equilibrium_filename=saved_args.get("equilibrium_filename", DEFAULT_EQUILIBRIUM_FILENAME),
        bfield_filename=saved_args.get("bfield_filename", DEFAULT_BFIELD_FILENAME),
        include_bfield=True,
        strict=True,
        target_database_path=target_database_path,
        target_database_key=saved_args.get("target_database_key", DEFAULT_TARGET_DATABASE_KEY),
        allow_missing_target_database=args.new_only,
    )

    if args.new_only:
        if new_folder_records is None:
            raise RuntimeError("Internal error: missing new folder records.")
        selected_splits = [args.new_split_name]
        split_indices = {
            args.new_split_name: [
                (sample_index, dataset_index)
                for sample_index, (dataset_index, _) in enumerate(new_folder_records)
            ]
        }
    elif explicit_split:
        train_indices = list(range(len(train_folders)))
        val_indices = list(range(len(train_folders), len(folders)))
        split_indices = {
            "train": [(dataset_index, dataset_index) for dataset_index in train_indices],
            "val": [(dataset_index, dataset_index) for dataset_index in val_indices],
        }
        selected_splits = list(dict.fromkeys(args.splits))
        if "val" in selected_splits and not val_indices:
            raise ValueError("This run has no validation split.")
    else:
        train_indices, val_indices = _split_indices(
            len(dataset),
            float(saved_args["train_fraction"]),
            int(saved_args["seed"]),
        )
        split_indices = {
            "train": [(dataset_index, dataset_index) for dataset_index in train_indices],
            "val": [(dataset_index, dataset_index) for dataset_index in val_indices],
        }
        selected_splits = list(dict.fromkeys(args.splits))
        if "val" in selected_splits and not val_indices:
            raise ValueError("This run has no validation split.")

    device = torch.device(args.device)
    _ensure_distributed(device)
    install_slice_token_capture(deterministic_slices=not args.stochastic_slices)

    model_config = checkpoint.get("model_config", config["model_config"])
    model = TransolverPlusModel(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    attention_modules = _attention_modules(model)

    max_nodes = None if args.full_mesh else args.max_nodes
    if max_nodes is None and not args.full_mesh:
        max_nodes = saved_args.get("max_nodes")
    token_dtype = TOKEN_DTYPES[args.token_dtype]

    first_sample = dataset[0]
    metadata = {
        "checkpoint": str(checkpoint_path),
        "run_dir": str(run_dir),
        "results_root": str(results_root),
        "target_database_path": str(target_database_path) if target_database_path else None,
        "model_config": model_config,
        "saved_args": saved_args,
        "export": {
            "deterministic_slices": not args.stochastic_slices,
            "max_nodes": max_nodes,
            "full_mesh": args.full_mesh,
            "token_dtype": args.token_dtype,
            "device": str(device),
            "splits": selected_splits,
            "folders_file": str(args.folders_file.expanduser()) if args.folders_file else None,
            "new_only": args.new_only,
            "new_split_name": args.new_split_name if args.new_only else None,
            "discovered_sample_count": discovered_sample_count,
            "existing_sample_count": existing_sample_count,
        },
        "feature_layout": _feature_layout(first_sample),
        "attention_module_names": [name for name, _ in attention_modules],
        "split_sizes": {split: len(indices) for split, indices in split_indices.items()},
    }
    metadata_name = "metadata_new.json" if args.new_only else "metadata.json"
    (output_dir / metadata_name).write_text(json.dumps(metadata, indent=2, default=str))

    manifest_path = output_dir / "manifest.jsonl"
    total_exported = 0
    manifest_mode = "a" if args.new_only and manifest_path.exists() else "w"
    with manifest_path.open(manifest_mode) as manifest:
        for split in selected_splits:
            indices = split_indices[split]
            if args.max_samples_per_split is not None:
                indices = indices[: args.max_samples_per_split]
            for split_index, (sample_index, dataset_index) in enumerate(indices):
                record = _export_one_sample(
                    model=model,
                    attention_modules=attention_modules,
                    sample=dataset[sample_index],
                    split=split,
                    split_index=split_index,
                    dataset_index=dataset_index,
                    saved_args=saved_args,
                    max_nodes=max_nodes,
                    token_dtype=token_dtype,
                    output_dir=output_dir,
                    device=device,
                )
                manifest.write(json.dumps(record) + "\n")
                total_exported += 1
                if args.log_every > 0 and total_exported % args.log_every == 0:
                    print(f"Exported {total_exported} samples; latest={record['token_path']}")

    print(f"Wrote {total_exported} token files under {output_dir}")
    print(f"Wrote manifest {manifest_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
