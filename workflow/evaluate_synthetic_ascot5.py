#!/usr/bin/env python3
"""Evaluate scalar fast-ion-loss inference on synthetic ASCOT5 trajectories.

The temporal Transolver is conditioned on pitch-averaged ``S(rho, ekin)`` from
the AFSI birth distribution, seeded with the initial ASCOT pressure frame, and
autoregressively generates the later frames. The complete synthetic trajectory is then passed through the scalar
Transolver. Its scalar output and
its captured latent tokens are evaluated against the ASCOT5 fraction lost.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from alpha_analysis.ai.dataloader import Ascot5Dataset
from alpha_analysis.ai.export_transolver_slice_tokens import (
    _attention_modules,
    _collect_captured_tokens,
    _path_aliases,
    _read_folder_list,
    _resolve_results_root,
    _resolve_target_database,
    _sample_to_tensors_with_indices,
    install_slice_token_capture,
)
from alpha_analysis.ai.time_dependent import append_predicted_frames
from alpha_analysis.ai.train_latent_generator import (
    NormalizationStats,
    TokenScalarHead,
    _select_latent,
    _unnormalize_target,
)
from alpha_analysis.ai.train_transolver import (
    TransolverPlusModel,
    _cleanup_distributed,
    _ensure_distributed,
    patch_transolver_attention_for_cuda,
)
from alpha_analysis.ai.train_transolver_timedependent import (
    AFSIContextTransolverModel,
    predict_node_profiles,
    sample_to_temporal_tensors,
)


CSV_FIELDS = (
    "later_index",
    "sample_key",
    "folder",
    "ground_truth_fraction_lost",
    "static_real_fraction_lost",
    "static_synthetic_fraction_lost",
    "latent_real_fraction_lost",
    "latent_synthetic_fraction_lost",
    "future_profile_rmse",
    "future_profile_mae",
    "future_profile_relative_rmse",
    "synthetic_profile_path",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text())


def _checkpoint_path(run_dir: Path, checkpoint: str | Path) -> Path:
    path = Path(checkpoint).expanduser()
    return path if path.is_absolute() else run_dir / path


def _resolve_listed_folders(folder_file: Path, results_root: Path) -> list[Path]:
    """Resolve saved Perlmutter paths against the active results root."""

    resolved = []
    missing = []
    for saved_folder in _read_folder_list(folder_file):
        candidates = [*_path_aliases(saved_folder), results_root / saved_folder.name]
        match = next((candidate for candidate in candidates if candidate.is_dir()), None)
        if match is None:
            missing.append(str(saved_folder))
        else:
            resolved.append(match.resolve())
    if missing:
        raise FileNotFoundError(
            f"Could not resolve {len(missing)} folders from {folder_file}; "
            f"first missing folder: {missing[0]}"
        )
    return resolved


def _folder_keys(folders: Sequence[Path]) -> set[str]:
    return {folder.name for folder in folders}


def validate_split_folders(
    train_folders: Sequence[Path],
    val_folders: Sequence[Path],
    later_folders: Sequence[Path],
) -> None:
    """Require three nonempty, pairwise-disjoint simulation inventories."""

    groups = {
        "train": _folder_keys(train_folders),
        "validation": _folder_keys(val_folders),
        "later": _folder_keys(later_folders),
    }
    for name, keys in groups.items():
        if not keys:
            raise ValueError(f"The {name} folder list is empty.")
    for left, right in (("train", "validation"), ("train", "later"), ("validation", "later")):
        overlap = groups[left] & groups[right]
        if overlap:
            raise ValueError(
                f"The {left} and {right} folder lists overlap; first duplicate: {sorted(overlap)[0]}"
            )


def _load_transolver(
    run_dir: Path,
    checkpoint_name: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], Path]:
    config = _load_json(run_dir / "config.json")
    checkpoint_path = _checkpoint_path(run_dir, checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config", config["model_config"])
    model_class = (
        AFSIContextTransolverModel
        if "context_points" in model_config
        else TransolverPlusModel
    )
    model = model_class(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint_path


def _stats_from_state(state: dict[str, Tensor]) -> NormalizationStats:
    return NormalizationStats(
        condition_mean=state["condition_mean"],
        condition_std=state["condition_std"],
        latent_mean=state["latent_mean"],
        latent_std=state["latent_std"],
        target_mean=state["target_mean"],
        target_std=state["target_std"],
    )


def _load_scalar_head(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[TokenScalarHead, NormalizationStats, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    config_args = config["args"]
    model = TokenScalarHead(
        token_dim=int(config["latent_dim"]),
        num_tokens=int(config["num_tokens"]),
        width=int(config_args["width"]),
        depth=int(config_args["scalar_depth"]),
        heads=int(config_args["heads"]),
        dropout=float(config_args["dropout"]),
        mlp_ratio=int(config_args["mlp_ratio"]),
    ).to(device)
    model.load_state_dict(checkpoint["scalar_head_state_dict"])
    model.eval()
    return model, _stats_from_state(checkpoint["stats"]), config


def _static_frame_count(model_config: dict[str, Any]) -> int:
    # Three coordinate and three B-field channels surround equal para/perp histories.
    profile_channels = int(model_config["space_dim"]) - 6
    if profile_channels <= 0 or profile_channels % 2:
        raise ValueError(
            "Cannot infer equal parallel/perpendicular frame counts from static "
            f"space_dim={model_config['space_dim']}."
        )
    return profile_channels // 2


def rollout_synthetic_profiles(
    model: torch.nn.Module,
    sample: dict[str, Any],
    saved_args: dict[str, Any],
    *,
    total_frames: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Seed from real input frames and autoregress to ``total_frames``."""

    input_frames = int(saved_args["input_frames"])
    output_frames = int(saved_args["output_frames"])
    if total_frames <= input_frames:
        raise ValueError("The synthetic trajectory must contain future frames.")
    if sample["prs_para"].shape[0] < input_frames:
        raise ValueError(
            f"{sample['folder']} has fewer than {input_frames} temporal seed frames."
        )

    if "afsi" not in sample:
        raise KeyError("AFSI-conditioned rollout sample is missing the AFSI distribution.")
    history_para = sample["prs_para"][:input_frames].clone()
    history_perp = sample["prs_perp"][:input_frames].clone()
    grid_shape = tuple(sample["bfield"]["br"].shape)
    generator = torch.Generator().manual_seed(int(saved_args["seed"]))

    patch_transolver_attention_for_cuda()
    with torch.no_grad():
        while history_para.shape[0] < total_frames:
            model_sample = dict(sample)
            model_sample["input_prs_para"] = history_para[-input_frames:]
            model_sample["input_prs_perp"] = history_perp[-input_frames:]
            repeat_shape = (output_frames,) + (1,) * (history_para.ndim - 1)
            model_sample["target_prs_para"] = history_para[-1:].repeat(repeat_shape)
            model_sample["target_prs_perp"] = history_perp[-1:].repeat(repeat_shape)
            x, pos, _, source_context, source_strength = sample_to_temporal_tensors(
                model_sample,
                max_nodes=None,
                profile_log1p=not bool(saved_args["no_profile_log1p"]),
                generator=generator,
                context_points=int(saved_args.get("context_points", 100)),
            )
            mask = torch.ones((1, x.shape[0]), dtype=torch.bool, device=device)
            prediction = predict_node_profiles(
                model,
                x.unsqueeze(0).to(device),
                pos.unsqueeze(0).to(device),
                mask,
                source_context.unsqueeze(0).to(device),
                source_strength.unsqueeze(0).to(device),
            )[0].cpu()
            history_para, history_perp = append_predicted_frames(
                history_para,
                history_perp,
                prediction,
                output_frames=output_frames,
                grid_shape=grid_shape,
                transformed=not bool(saved_args["no_profile_log1p"]),
            )

    return history_para[:total_frames], history_perp[:total_frames]


def _target_units(value: float, saved_args: dict[str, Any]) -> float:
    if saved_args.get("no_log_target", False):
        return value
    return max((10.0**value) - float(saved_args["target_eps"]), 0.0)


def _static_and_latent_prediction(
    model: torch.nn.Module,
    attention_modules: Sequence[tuple[str, torch.nn.Module]],
    scalar_head: TokenScalarHead,
    scalar_stats: NormalizationStats,
    scalar_config: dict[str, Any],
    sample: dict[str, Any],
    saved_args: dict[str, Any],
    *,
    sample_seed: int,
    device: torch.device,
) -> tuple[float, float, float]:
    generator = torch.Generator().manual_seed(sample_seed)
    x, pos, target, _, _ = _sample_to_tensors_with_indices(
        sample,
        max_nodes=saved_args["max_nodes"],
        target_reduction=saved_args["target_reduction"],
        log10_target=not saved_args["no_log_target"],
        target_eps=saved_args["target_eps"],
        profile_log1p=not saved_args["no_profile_log1p"],
        generator=generator,
    )
    with torch.no_grad():
        # The ordinary scalar head was trained with Transolver's native
        # gumbel-softmax slice assignment.  Keep that forward pass intact for
        # its prediction; deterministic token capture is only appropriate for
        # the separately trained latent scalar head.
        patch_transolver_attention_for_cuda()
        torch.manual_seed(sample_seed)
        node_values = model(
            (x.unsqueeze(0).to(device), pos.unsqueeze(0).to(device), None)
        ).squeeze(-1)
        static_prediction = node_values.mean(dim=1).squeeze(0).cpu()

        # Re-run with the deterministic capture patch because the latent head
        # was trained on these reproducible out_slice_tokens.
        install_slice_token_capture(deterministic_slices=True)
        model(
            (x.unsqueeze(0).to(device), pos.unsqueeze(0).to(device), None)
        )
        captured = _collect_captured_tokens(attention_modules, token_dtype=torch.float32)
        latent = _select_latent(
            captured,
            str(scalar_config["token_kind"]),
            str(scalar_config["layers"]),
        )
        latent = (latent - scalar_stats.latent_mean.cpu()) / scalar_stats.latent_std.cpu()
        latent_prediction = scalar_head(latent.unsqueeze(0).to(device))
        latent_prediction = _unnormalize_target(latent_prediction, scalar_stats).squeeze().cpu()

    return (
        _target_units(float(target.item()), saved_args),
        _target_units(float(static_prediction.item()), saved_args),
        float(latent_prediction.item()),
    )


def profile_error_metrics(
    synthetic_para: Tensor,
    synthetic_perp: Tensor,
    truth_para: Tensor,
    truth_perp: Tensor,
    *,
    seed_frames: int,
) -> dict[str, float]:
    predicted = torch.stack(
        (synthetic_para[seed_frames:], synthetic_perp[seed_frames:]), dim=1
    ).float()
    truth = torch.stack((truth_para[seed_frames:], truth_perp[seed_frames:]), dim=1).float()
    finite = torch.isfinite(predicted) & torch.isfinite(truth)
    if not bool(finite.any()):
        return {"rmse": math.nan, "mae": math.nan, "relative_rmse": math.nan}
    error = predicted[finite] - truth[finite]
    rmse = error.square().mean().sqrt()
    mae = error.abs().mean()
    truth_rms = truth[finite].square().mean().sqrt()
    return {
        "rmse": float(rmse.item()),
        "mae": float(mae.item()),
        "relative_rmse": float((rmse / truth_rms.clamp_min(1.0e-30)).item()),
    }


def _metrics(rows: Sequence[dict[str, Any]], prediction_key: str) -> dict[str, float]:
    if not rows:
        return {"mse": math.nan, "mae": math.nan, "r2": math.nan}
    targets = torch.tensor(
        [float(row["ground_truth_fraction_lost"]) for row in rows], dtype=torch.float64
    )
    predictions = torch.tensor(
        [float(row[prediction_key]) for row in rows], dtype=torch.float64
    )
    residual = predictions - targets
    ss_total = (targets - targets.mean()).square().sum()
    r2 = 1.0 - residual.square().sum() / ss_total if float(ss_total) > 0.0 else math.nan
    return {
        "mse": float(residual.square().mean()),
        "mae": float(residual.abs().mean()),
        "r2": float(r2),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _plot_parity(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    targets = [float(row["ground_truth_fraction_lost"]) for row in rows]
    series = (
        ("static_synthetic_fraction_lost", "Static Transolver, synthetic profiles"),
        ("latent_synthetic_fraction_lost", "Latent scalar head, synthetic profiles"),
        ("static_real_fraction_lost", "Static Transolver, real profiles"),
        ("latent_real_fraction_lost", "Latent scalar head, real profiles"),
    )
    all_values = list(targets)
    for key, _ in series:
        all_values.extend(float(row[key]) for row in rows)
    lower, upper = min(all_values), max(all_values)
    padding = 0.05 * max(upper - lower, 1.0e-12)
    lower -= padding
    upper += padding

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, series):
        predictions = [float(row[key]) for row in rows]
        metrics = _metrics(rows, key)
        axis.scatter(targets, predictions, s=22, alpha=0.7)
        axis.plot([lower, upper], [lower, upper], "k--", linewidth=1)
        axis.set(xlim=(lower, upper), ylim=(lower, upper))
        axis.set_xlabel("ASCOT5 fraction_lost")
        axis.set_ylabel("Predicted fraction_lost")
        axis.set_title(title)
        axis.grid(alpha=0.3)
        axis.text(
            0.03,
            0.97,
            f"R²={metrics['r2']:.4g}\nMAE={metrics['mae']:.4g}",
            transform=axis.transAxes,
            va="top",
        )
    fig.suptitle(f"Later/unseen ASCOT5 simulations (n={len(rows)})")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    split_root = Path("runs/frame_autoencoder/56011733")
    parser.add_argument("--train-folders", type=Path, default=split_root / "train_folders.txt")
    parser.add_argument("--val-folders", type=Path, default=split_root / "val_folders.txt")
    parser.add_argument("--later-folders", type=Path, default=split_root / "later_folders.txt")
    parser.add_argument("--results-root", type=Path)
    parser.add_argument(
        "--temporal-run-dir",
        type=Path,
        default=Path("runs/transolver_alpha_timedependent/2896274"),
    )
    parser.add_argument("--temporal-checkpoint", default="best.pt")
    parser.add_argument(
        "--static-run-dir", type=Path, default=Path("runs/transolver_alpha/53562942")
    )
    parser.add_argument("--static-checkpoint", default="best.pt")
    parser.add_argument(
        "--scalar-head-checkpoint",
        type=Path,
        default=Path(
            "runs/transolver_alpha/53562942/best_slice_tokens/"
            "latent_generator_54945515/scalar_head_best.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/synthetic_ascot5_generalization/2896274_53562942"),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-save-profiles", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    temporal_run_dir = args.temporal_run_dir.expanduser().resolve()
    static_run_dir = args.static_run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    static_config = _load_json(static_run_dir / "config.json")
    static_saved_args = static_config["args"]
    results_root = _resolve_results_root(static_saved_args["results_root"], args.results_root)
    target_database = _resolve_target_database(
        static_saved_args.get("target_database"), results_root
    )
    if target_database is None:
        raise ValueError("The static scalar model requires a fraction_lost target database.")

    train_folders = _resolve_listed_folders(args.train_folders, results_root)
    val_folders = _resolve_listed_folders(args.val_folders, results_root)
    later_folders = _resolve_listed_folders(args.later_folders, results_root)
    validate_split_folders(train_folders, val_folders, later_folders)
    selected_folders = later_folders[args.start_index :]
    if args.max_samples is not None:
        selected_folders = selected_folders[: args.max_samples]
    if not selected_folders:
        raise ValueError("No later/unseen folders were selected.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable.")
    _ensure_distributed(device)
    temporal_model, temporal_config, temporal_checkpoint = _load_transolver(
        temporal_run_dir, args.temporal_checkpoint, device
    )
    temporal_saved_args = temporal_config["args"]
    static_model, loaded_static_config, static_checkpoint = _load_transolver(
        static_run_dir, args.static_checkpoint, device
    )
    static_saved_args = loaded_static_config["args"]
    scalar_checkpoint = args.scalar_head_checkpoint.expanduser().resolve()
    scalar_head, scalar_stats, scalar_config = _load_scalar_head(scalar_checkpoint, device)
    attention_modules = _attention_modules(static_model)
    total_frames = _static_frame_count(loaded_static_config["model_config"])

    dataset = Ascot5Dataset(
        selected_folders,
        analysis_filename=static_saved_args["analysis_filename"],
        equilibrium_filename=static_saved_args["equilibrium_filename"],
        bfield_filename=static_saved_args["bfield_filename"],
        afsi_filename=temporal_saved_args["afsi_filename"],
        include_bfield=True,
        include_afsi=True,
        strict=True,
        target_database_path=target_database,
        target_database_key=static_saved_args["target_database_key"],
    )

    csv_path = output_dir / "predictions.csv"
    rows = [] if args.overwrite else _read_existing_rows(csv_path)
    completed = {str(row["sample_key"]) for row in rows}
    profile_dir = output_dir / "synthetic_profiles"
    if not args.no_save_profiles:
        profile_dir.mkdir(parents=True, exist_ok=True)

    try:
        for local_index in range(len(dataset)):
            sample = dataset[local_index]
            folder = Path(sample["folder"])
            sample_key = folder.name
            later_index = args.start_index + local_index
            if sample_key in completed:
                print(f"Skipping completed sample {sample_key}", flush=True)
                continue
            if sample["prs_para"].shape[0] < total_frames:
                raise ValueError(
                    f"{folder} has {sample['prs_para'].shape[0]} frames; the static model "
                    f"requires {total_frames}."
                )

            synthetic_para, synthetic_perp = rollout_synthetic_profiles(
                temporal_model,
                sample,
                temporal_config["args"],
                total_frames=total_frames,
                device=device,
            )
            synthetic_sample = dict(sample)
            synthetic_sample["prs_para"] = synthetic_para
            synthetic_sample["prs_perp"] = synthetic_perp

            # Scalar-head training used deterministic token extraction; the
            # helper uses the native forward pass separately for static output.
            sample_seed = int(static_saved_args["seed"]) + later_index
            truth, static_synthetic, latent_synthetic = _static_and_latent_prediction(
                static_model,
                attention_modules,
                scalar_head,
                scalar_stats,
                scalar_config,
                synthetic_sample,
                static_saved_args,
                sample_seed=sample_seed,
                device=device,
            )
            real_truth, static_real, latent_real = _static_and_latent_prediction(
                static_model,
                attention_modules,
                scalar_head,
                scalar_stats,
                scalar_config,
                sample,
                static_saved_args,
                sample_seed=sample_seed,
                device=device,
            )
            if not math.isclose(truth, real_truth, rel_tol=1.0e-6, abs_tol=1.0e-8):
                raise RuntimeError(f"Target changed between synthetic and real inference for {folder}.")

            errors = profile_error_metrics(
                synthetic_para,
                synthetic_perp,
                sample["prs_para"][:total_frames],
                sample["prs_perp"][:total_frames],
                seed_frames=int(temporal_config["args"]["input_frames"]),
            )
            synthetic_path = profile_dir / f"{sample_key}.pt"
            if not args.no_save_profiles:
                torch.save(
                    {
                        "folder": str(folder),
                        "profile_time": sample["profile_time"][:total_frames],
                        "prs_para": synthetic_para,
                        "prs_perp": synthetic_perp,
                        "seed_frames": int(temporal_config["args"]["input_frames"]),
                        "temporal_checkpoint": str(temporal_checkpoint),
                    },
                    synthetic_path,
                )

            row = {
                "later_index": later_index,
                "sample_key": sample_key,
                "folder": str(folder),
                "ground_truth_fraction_lost": truth,
                "static_real_fraction_lost": static_real,
                "static_synthetic_fraction_lost": static_synthetic,
                "latent_real_fraction_lost": latent_real,
                "latent_synthetic_fraction_lost": latent_synthetic,
                "future_profile_rmse": errors["rmse"],
                "future_profile_mae": errors["mae"],
                "future_profile_relative_rmse": errors["relative_rmse"],
                "synthetic_profile_path": "" if args.no_save_profiles else str(synthetic_path),
            }
            rows.append(row)
            rows.sort(key=lambda item: int(item["later_index"]))
            _write_csv(csv_path, rows)
            completed.add(sample_key)
            print(
                f"{local_index + 1}/{len(dataset)} {sample_key}: truth={truth:.6g}, "
                f"static(synth)={static_synthetic:.6g}, "
                f"latent(synth)={latent_synthetic:.6g}",
                flush=True,
            )
    finally:
        _cleanup_distributed()

    metric_keys = (
        "static_synthetic_fraction_lost",
        "latent_synthetic_fraction_lost",
        "static_real_fraction_lost",
        "latent_real_fraction_lost",
    )
    summary = {
        "samples": len(rows),
        "split_counts": {
            "train": len(train_folders),
            "validation": len(val_folders),
            "later": len(later_folders),
        },
        "checkpoints": {
            "temporal": str(temporal_checkpoint),
            "static": str(static_checkpoint),
            "scalar_head": str(scalar_checkpoint),
        },
        "scalar_head_tokens": {
            "token_kind": scalar_config["token_kind"],
            "layers": scalar_config["layers"],
        },
        "metrics": {key: _metrics(rows, key) for key in metric_keys},
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot_parity(output_dir / "fraction_lost_parity.png", rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {output_dir / 'metrics.json'}")
    print(f"Wrote {output_dir / 'fraction_lost_parity.png'}")


if __name__ == "__main__":
    main()
