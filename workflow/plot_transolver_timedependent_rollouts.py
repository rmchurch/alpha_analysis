#!/usr/bin/env python3
"""Plot autoregressive Transolver rollouts on ASCOT5 validation simulations.

The timedependent Transolver run in this repository is trained to predict one
frame from the preceding state. This script feeds the prediction back into
the model repeatedly, starting from ASCOT frame zero while supplying fixed AFSI
cross-attention context, and compares frame ``t=10`` with the corresponding
ground truth. Each selected validation simulation is
shown as two rows (parallel and perpendicular pressure) and three columns:
initial frame, predicted final frame, and ground-truth final frame.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from alpha_analysis.ai.dataloader import Ascot5Dataset
from alpha_analysis.ai.train_transolver import (
    _cleanup_distributed,
    _discover_sample_folders,
    _ensure_distributed,
    patch_transolver_attention_for_cuda,
)
from alpha_analysis.ai.train_transolver_timedependent import (
    AFSIContextTransolverModel,
    _split_folders,
    predict_node_profiles,
    sample_to_temporal_tensors,
)
from alpha_analysis.ai.time_dependent import append_predicted_frames


PROFILE_NAMES = ("prs_para", "prs_perp")
PROFILE_LABELS = ("Parallel pressure", "Perpendicular pressure")
AXIS_NAMES = ("rho", "theta", "phi")


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent.parent
    default_run_dir = repo_root / "runs" / "transolver_alpha_timedependent" / "2891161"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=default_run_dir,
        help="Timedependent training run containing config.json and best.pt.",
    )
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument(
        "--results-root",
        type=Path,
        help="Override the results root saved in config.json.",
    )
    parser.add_argument(
        "--indices",
        default="0,1,2,3",
        help="Validation-relative indices to plot, e.g. '0,4,7'.",
    )
    parser.add_argument(
        "--total-frames",
        type=int,
        default=10,
        help="Number of frames in the rollout, including the initial frame.",
    )
    parser.add_argument(
        "--slice-axis",
        choices=AXIS_NAMES,
        default="phi",
        help="Grid axis to hold fixed when displaying a 3-D profile.",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        help="Index on --slice-axis. Defaults to the middle index.",
    )
    parser.add_argument(
        "--representation",
        choices=("slice", "rho"),
        default="slice",
        help=(
            "Plot 2-D spatial slices or plot every pressure-grid point against "
            "its rho coordinate."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG. Defaults to <run-dir>/best_timedependent_rollouts.png.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _load_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    return json.loads(config_path.read_text())


def _resolve_results_root(saved_root: str, override: Path | None) -> Path:
    candidates = [override.expanduser()] if override is not None else []
    candidates.append(Path(saved_root).expanduser())

    # Permit configs written on the cluster to be used from a mounted project
    # filesystem with the equivalent cdirs/cfs path spelling.
    saved_text = str(saved_root)
    if saved_text.startswith("/global/cfs/cdirs/"):
        candidates.append(Path(saved_text.replace("/global/cfs/cdirs/", "/global/cfs/projectdirs/", 1)))
    if saved_text.startswith("/global/cfs/projectdirs/"):
        candidates.append(Path(saved_text.replace("/global/cfs/projectdirs/", "/global/cfs/cdirs/", 1)))

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find results root. Tried: " + ", ".join(str(path) for path in candidates)
    )


def _parse_indices(value: str, count: int) -> list[int]:
    indices: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            parts = [int(part) if part else None for part in token.split(":")]
            if len(parts) > 3:
                raise ValueError(f"Invalid validation index range: {token}")
            indices.extend(range(count)[slice(*parts)])
        else:
            indices.append(int(token))

    if not indices:
        raise ValueError("--indices did not select any validation examples.")
    invalid = [index for index in indices if index < 0 or index >= count]
    if invalid:
        raise IndexError(
            f"Validation indices {invalid} are outside the available range [0, {count})."
        )
    return indices


def _checkpoint_path(run_dir: Path, checkpoint: str) -> Path:
    path = Path(checkpoint).expanduser()
    return path if path.is_absolute() else run_dir / path


def _rollout(
    model: torch.nn.Module,
    sample: dict[str, Any],
    saved_args: dict[str, Any],
    *,
    total_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_frames = int(saved_args["input_frames"])
    output_frames = int(saved_args["output_frames"])
    frame_stride = int(saved_args["frame_stride"])
    if input_frames != 1 or output_frames != 1 or frame_stride != 1:
        raise ValueError(
            "This plotting script expects the one-frame, unit-stride run; found "
            f"input_frames={input_frames}, output_frames={output_frames}, "
            f"frame_stride={frame_stride}."
        )

    num_stored = int(sample["prs_para"].shape[0])
    if num_stored < total_frames:
        raise ValueError(
            f"{sample['folder']} has {num_stored} stored frames, but "
            f"--total-frames={total_frames} was requested."
        )

    grid_shape = tuple(int(size) for size in sample["bfield"]["br"].shape)
    if "afsi" not in sample:
        raise KeyError("AFSI-conditioned rollout sample is missing the AFSI distribution.")
    history_para = sample["prs_para"][:input_frames].clone()
    history_perp = sample["prs_perp"][:input_frames].clone()
    generator = torch.Generator().manual_seed(int(saved_args["seed"]))

    with torch.no_grad():
        for _ in range(input_frames, total_frames):
            model_sample = dict(sample)
            model_sample["input_prs_para"] = history_para[-input_frames:]
            model_sample["input_prs_perp"] = history_perp[-input_frames:]
            model_sample["target_prs_para"] = history_para[-1:].clone()
            model_sample["target_prs_perp"] = history_perp[-1:].clone()
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

    initial = torch.stack((sample["prs_para"][0], sample["prs_perp"][0]))
    predicted_final = torch.stack((history_para[-1], history_perp[-1]))
    truth_final = torch.stack(
        (sample["prs_para"][total_frames - 1], sample["prs_perp"][total_frames - 1])
    )
    return initial, predicted_final, truth_final, sample["profile_time"][total_frames - 1]


def _display_slice(field: torch.Tensor, axis: int, index: int | None) -> torch.Tensor:
    if field.ndim < 2:
        raise ValueError(f"Expected a spatial field with at least two dimensions, got {field.shape}.")
    if field.ndim == 2:
        return field
    if axis >= field.ndim:
        raise ValueError(f"Cannot slice axis {axis} from field shape {tuple(field.shape)}.")
    chosen = field.shape[axis] // 2 if index is None else index
    chosen = max(0, min(int(chosen), field.shape[axis] - 1))
    return field.select(axis, chosen)


def _limits(fields: Sequence[torch.Tensor]) -> tuple[float, float]:
    values = torch.cat([field.detach().float().reshape(-1) for field in fields])
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return -1.0, 1.0
    low, high = float(values.min()), float(values.max())
    if math.isclose(low, high):
        padding = max(abs(low) * 0.05, 1.0e-6)
        return low - padding, high + padding
    return low, high


def _symmetric_limit(field: torch.Tensor) -> float:
    values = field.detach().float()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return 1.0
    return max(float(values.abs().max()), 1.0e-12)


def _axis_values(
    coordinate: torch.Tensor | None,
    size: int,
) -> torch.Tensor:
    if coordinate is not None and coordinate.ndim == 1 and coordinate.numel() == size:
        return coordinate.detach().cpu()
    return torch.arange(size, dtype=torch.float32)


def _plot_examples(
    output: Path,
    examples: Sequence[
        tuple[
            int,
            str,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
        ]
    ],
    *,
    slice_axis: int,
    slice_index: int | None,
    total_frames: int,
    final_time: Sequence[torch.Tensor],
) -> None:
    rows = 3 * len(examples)
    height_ratios = []
    for example_number in range(len(examples)):
        spacer_height = 0.02 if example_number == len(examples) - 1 else 0.42
        height_ratios.extend((1.0, 1.0, spacer_height))
    fig, axes = plt.subplots(
        rows,
        4,
        figsize=(14, 3.8 * len(examples)),
        squeeze=False,
        gridspec_kw={
            "wspace": 0.10,
            "hspace": 0.08,
            "height_ratios": height_ratios,
        },
    )
    column_titles = (
        "Initial ASCOT frame (t=1)",
        f"Predicted final (t={total_frames})",
        f"Ground truth (t={total_frames})",
        "Prediction − ground truth",
    )

    for example_number, (
        validation_index,
        folder,
        initial,
        predicted,
        truth,
        coordinates,
    ) in enumerate(examples):
        row_offset = 3 * example_number
        for spacer_axis in axes[row_offset + 2]:
            spacer_axis.set_visible(False)
        time_value = float(final_time[example_number].detach().cpu())
        titles = list(column_titles)
        titles[2] = f"Ground truth (t={total_frames}, time={time_value:.4g})"
        titles[0] = f"Validation {validation_index}: {Path(folder).name}\n{titles[0]}"
        for column, title in enumerate(titles):
            axes[row_offset, column].set_title(title, fontsize=10, pad=3)

        for profile_index, label in enumerate(PROFILE_LABELS):
            row = row_offset + profile_index
            state_fields = (
                initial[profile_index],
                predicted[profile_index],
                truth[profile_index],
            )
            display_fields = [
                _display_slice(field, slice_axis, slice_index) for field in state_fields
            ]
            if state_fields[0].ndim == 2:
                remaining_axes = (0, 1)
            else:
                remaining_axes = tuple(
                    axis for axis in range(state_fields[0].ndim) if axis != slice_axis
                )
            if len(remaining_axes) != 2:
                raise ValueError(
                    "Plotting requires a 2-D field or a 3-D field with one sliced axis; "
                    f"received shape {tuple(state_fields[0].shape)}."
                )

            x_axis, y_axis = remaining_axes
            x_values = _axis_values(coordinates[x_axis], display_fields[0].shape[0])
            y_values = _axis_values(coordinates[y_axis], display_fields[0].shape[1])
            vmin, vmax = _limits(display_fields)
            state_images = []
            for column, field in enumerate(display_fields):
                state_images.append(
                    axes[row, column].pcolormesh(
                        x_values.numpy(),
                        y_values.numpy(),
                        field.transpose(0, 1).numpy(),
                        shading="auto",
                        cmap="viridis",
                        vmin=vmin,
                        vmax=vmax,
                    )
                )

            difference = display_fields[1] - display_fields[2]
            difference_limit = _symmetric_limit(difference)
            difference_image = axes[row, 3].pcolormesh(
                x_values.numpy(),
                y_values.numpy(),
                difference.transpose(0, 1).numpy(),
                shading="auto",
                cmap="RdBu_r",
                vmin=-difference_limit,
                vmax=difference_limit,
            )

            for column in range(4):
                axes[row, column].set_xlabel(AXIS_NAMES[x_axis])
                axes[row, column].set_ylabel(AXIS_NAMES[y_axis] if column == 0 else "")
                axes[row, column].tick_params(labelsize=8, pad=1)
            axes[row, 0].annotate(
                label,
                xy=(-0.27, 0.5),
                xycoords="axes fraction",
                ha="center",
                va="center",
                rotation=90,
                fontsize=10,
            )

            pressure_colorbar = fig.colorbar(
                state_images[-1],
                ax=axes[row, 2],
                fraction=0.045,
                pad=0.012,
                aspect=20,
            )
            pressure_colorbar.set_label("Pressure", fontsize=8)
            difference_colorbar = fig.colorbar(
                difference_image,
                ax=axes[row, 3],
                fraction=0.045,
                pad=0.012,
                aspect=20,
            )
            difference_colorbar.set_label("Prediction − truth", fontsize=8)

    fig.suptitle("Autoregressive Transolver ASCOT5 validation rollouts", fontsize=15)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.045, top=0.955)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _rho_point_coordinates(
    rho_coordinate: torch.Tensor | None,
    field: torch.Tensor,
) -> torch.Tensor:
    rho = _axis_values(rho_coordinate, field.shape[0])
    reshape = (rho.numel(),) + (1,) * (field.ndim - 1)
    return rho.reshape(reshape).expand_as(field).reshape(-1)


def _scatter_all_points(
    axis: plt.Axes,
    rho: torch.Tensor,
    field: torch.Tensor,
    *,
    color: str,
) -> None:
    values = field.detach().float().reshape(-1).cpu()
    finite = torch.isfinite(rho) & torch.isfinite(values)
    axis.scatter(
        rho[finite].numpy(),
        values[finite].numpy(),
        s=0.25,
        alpha=0.25,
        color=color,
        linewidths=0,
        rasterized=True,
    )


def _plot_rho_examples(
    output: Path,
    examples: Sequence[
        tuple[
            int,
            str,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
        ]
    ],
    *,
    total_frames: int,
    final_time: Sequence[torch.Tensor],
) -> None:
    rows = 3 * len(examples)
    height_ratios = []
    for example_number in range(len(examples)):
        spacer_height = 0.02 if example_number == len(examples) - 1 else 0.42
        height_ratios.extend((1.0, 1.0, spacer_height))
    fig, axes = plt.subplots(
        rows,
        4,
        figsize=(14, 3.8 * len(examples)),
        squeeze=False,
        sharex=False,
        gridspec_kw={
            "wspace": 0.18,
            "hspace": 0.08,
            "height_ratios": height_ratios,
        },
    )
    state_colors = ("#2563eb", "#ea580c", "#16a34a")
    column_titles = (
        "Initial ASCOT frame (t=1)",
        f"Predicted final (t={total_frames})",
        f"Ground truth (t={total_frames})",
        "Prediction − ground truth",
    )

    for example_number, (
        validation_index,
        folder,
        initial,
        predicted,
        truth,
        coordinates,
    ) in enumerate(examples):
        row_offset = 3 * example_number
        for spacer_axis in axes[row_offset + 2]:
            spacer_axis.set_visible(False)
        time_value = float(final_time[example_number].detach().cpu())
        titles = list(column_titles)
        titles[0] = f"Validation {validation_index}: {Path(folder).name}\n{titles[0]}"
        titles[2] = f"Ground truth (t={total_frames}, time={time_value:.4g})"
        for column, title in enumerate(titles):
            axes[row_offset, column].set_title(title, fontsize=10, pad=3)

        for profile_index, label in enumerate(PROFILE_LABELS):
            row = row_offset + profile_index
            state_fields = (
                initial[profile_index],
                predicted[profile_index],
                truth[profile_index],
            )
            rho = _rho_point_coordinates(coordinates[0], state_fields[0])
            state_limits = _limits(state_fields)
            for column, (field, color) in enumerate(zip(state_fields, state_colors)):
                _scatter_all_points(axes[row, column], rho, field, color=color)
                axes[row, column].set_ylim(*state_limits)

            difference = state_fields[1] - state_fields[2]
            difference_limit = _symmetric_limit(difference)
            _scatter_all_points(axes[row, 3], rho, difference, color="#7c3aed")
            axes[row, 3].set_ylim(-difference_limit, difference_limit)
            axes[row, 3].axhline(0.0, color="black", linewidth=0.6, alpha=0.6)

            for column in range(4):
                axes[row, column].set_xlabel("rho")
                axes[row, column].grid(True, alpha=0.18, linewidth=0.5)
                axes[row, column].tick_params(labelsize=8, pad=1)
            axes[row, 0].set_ylabel("Pressure")
            axes[row, 3].set_ylabel("Pressure error")
            axes[row, 0].annotate(
                label,
                xy=(-0.28, 0.5),
                xycoords="axes fraction",
                ha="center",
                va="center",
                rotation=90,
                fontsize=10,
            )

    fig.suptitle(
        "Autoregressive Transolver validation rollouts: all pressure points vs rho",
        fontsize=15,
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.045, top=0.955)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if args.total_frames < 2:
        raise ValueError("--total-frames must be at least 2 for a rollout.")

    run_dir = args.run_dir.expanduser().resolve()
    config = _load_config(run_dir)
    saved_args = config["args"]
    results_root = _resolve_results_root(saved_args["results_root"], args.results_root)
    folders = _discover_sample_folders(
        results_root,
        saved_args["analysis_filename"],
        saved_args["equilibrium_filename"],
        saved_args["bfield_filename"],
    )
    folders = [
        folder
        for folder in folders
        if (folder / saved_args["afsi_filename"]).is_file()
    ]
    if saved_args.get("max_samples") is not None:
        folders = folders[: int(saved_args["max_samples"])]
    _, val_folders = _split_folders(folders, float(saved_args["train_fraction"]), int(saved_args["seed"]))
    selected = _parse_indices(args.indices, len(val_folders))

    dataset = Ascot5Dataset(
        [val_folders[index] for index in selected],
        analysis_filename=saved_args["analysis_filename"],
        equilibrium_filename=saved_args["equilibrium_filename"],
        bfield_filename=saved_args["bfield_filename"],
        afsi_filename=saved_args["afsi_filename"],
        include_bfield=True,
        include_afsi=True,
        include_target=False,
        strict=True,
    )

    device = torch.device(args.device)
    _ensure_distributed(device)
    patch_transolver_attention_for_cuda()
    checkpoint_path = _checkpoint_path(run_dir, args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = AFSIContextTransolverModel(
        **checkpoint.get("model_config", config["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    examples = []
    final_times = []
    for validation_index, sample in zip(selected, dataset):
        initial, predicted, truth, final_time = _rollout(
            model,
            sample,
            saved_args,
            total_frames=args.total_frames,
            device=device,
        )
        error = predicted - truth
        mse = float(error.square().mean())
        mae = float(error.abs().mean())
        print(f"validation={validation_index} folder={sample['folder']} final_mse={mse:.6g} final_mae={mae:.6g}")
        coordinates = tuple(sample["bfield"].get(name) for name in AXIS_NAMES)
        examples.append(
            (
                validation_index,
                sample["folder"],
                initial,
                predicted,
                truth,
                coordinates,
            )
        )
        final_times.append(final_time)

    default_name = (
        f"{Path(args.checkpoint).stem}_timedependent_rollouts_rho.png"
        if args.representation == "rho"
        else f"{Path(args.checkpoint).stem}_timedependent_rollouts.png"
    )
    output = args.output or run_dir / default_name
    if args.representation == "rho":
        _plot_rho_examples(
            output.expanduser().resolve(),
            examples,
            total_frames=args.total_frames,
            final_time=final_times,
        )
    else:
        _plot_examples(
            output.expanduser().resolve(),
            examples,
            slice_axis=AXIS_NAMES.index(args.slice_axis),
            slice_index=args.slice_index,
            total_frames=args.total_frames,
            final_time=final_times,
        )
    print(f"Wrote {output.expanduser().resolve()}")
    _cleanup_distributed()


if __name__ == "__main__":
    main()
