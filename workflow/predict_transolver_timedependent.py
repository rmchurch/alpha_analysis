#!/usr/bin/env python3
"""Autoregressively forecast ASCOT profile frames with a temporal Transolver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from alpha_analysis.ai.dataloader import Ascot5Dataset
from alpha_analysis.ai.time_dependent import (
    append_predicted_frames,
)
from alpha_analysis.ai.train_transolver import (
    _cleanup_distributed,
    _ensure_distributed,
    patch_transolver_attention_for_cuda,
)
from alpha_analysis.ai.train_transolver_timedependent import (
    AFSIContextTransolverModel,
    predict_node_profiles,
    sample_to_temporal_tensors,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--sample-folder", type=Path, required=True)
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Stored ASCOT frame index at which the autoregressive seed begins.",
    )
    parser.add_argument(
        "--forecast-frames",
        type=int,
        default=None,
        help="Number of future frames to produce; defaults to all remaining stored frames.",
    )
    parser.add_argument("--output", type=Path, default=Path("temporal_prediction.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _checkpoint_path(run_dir: Path, checkpoint: str) -> Path:
    path = Path(checkpoint)
    return path if path.is_absolute() else run_dir / path


def _load_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing temporal training config: {path}")
    return json.loads(path.read_text())


def _future_times(
    stored_times: torch.Tensor,
    *,
    first_index: int,
    count: int,
    frame_stride: int,
) -> torch.Tensor:
    indices = first_index + torch.arange(count) * frame_stride
    result = torch.empty(count, dtype=stored_times.dtype)
    valid = indices < stored_times.numel()
    result[valid] = stored_times[indices[valid]]
    if bool((~valid).any()):
        if stored_times.numel() > 1:
            base_dt = torch.median(stored_times[1:] - stored_times[:-1])
        else:
            base_dt = stored_times.new_tensor(1.0)
        first_missing = int(valid.sum())
        anchor = result[first_missing - 1] if first_missing else stored_times[-1]
        offsets = torch.arange(
            1, count - first_missing + 1, dtype=stored_times.dtype
        )
        result[first_missing:] = anchor + offsets * base_dt * frame_stride
    return result


def main() -> None:
    args = build_parser().parse_args()
    config = _load_config(args.run_dir)
    saved_args = config["args"]
    input_frames = int(saved_args["input_frames"])
    output_frames = int(saved_args["output_frames"])
    frame_stride = int(saved_args["frame_stride"])
    afsi_filename = saved_args.get("afsi_filename")
    if afsi_filename is None:
        raise ValueError(
            "This checkpoint predates AFSI initial conditioning; train a new temporal "
            "checkpoint with the updated trainer."
        )

    dataset = Ascot5Dataset(
        [args.sample_folder],
        analysis_filename=saved_args["analysis_filename"],
        equilibrium_filename=saved_args["equilibrium_filename"],
        bfield_filename=saved_args["bfield_filename"],
        afsi_filename=afsi_filename,
        include_bfield=True,
        include_afsi=True,
        include_target=False,
        strict=True,
    )
    sample = dataset[0]
    num_stored = int(sample["prs_para"].shape[0])
    seed_indices = args.seed_start + torch.arange(input_frames) * frame_stride
    if args.seed_start < 0 or int(seed_indices[-1]) >= num_stored:
        raise ValueError(
            f"Seed indices {seed_indices.tolist()} are outside the {num_stored} stored frames."
        )
    first_future_index = int(seed_indices[-1]) + frame_stride
    available_future = max(
        0, (num_stored - 1 - first_future_index) // frame_stride + 1
    )
    forecast_frames = (
        available_future if args.forecast_frames is None else args.forecast_frames
    )
    if forecast_frames <= 0:
        raise ValueError("--forecast-frames must be positive and future frames must exist.")

    device = torch.device(args.device)
    _ensure_distributed(device)
    patch_transolver_attention_for_cuda()
    checkpoint_path = _checkpoint_path(args.run_dir, args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config", config["model_config"])
    model = AFSIContextTransolverModel(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    history_para = sample["prs_para"].index_select(0, seed_indices).clone()
    history_perp = sample["prs_perp"].index_select(0, seed_indices).clone()
    grid_shape = tuple(sample["bfield"]["br"].shape)
    generated = 0
    generator = torch.Generator().manual_seed(int(saved_args["seed"]))
    with torch.no_grad():
        while generated < forecast_frames:
            model_sample = dict(sample)
            model_sample["input_prs_para"] = history_para[-input_frames:]
            model_sample["input_prs_perp"] = history_perp[-input_frames:]
            # This placeholder is not read when forming inference features.
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
            generated += output_frames

    predicted_para = history_para[input_frames : input_frames + forecast_frames]
    predicted_perp = history_perp[input_frames : input_frames + forecast_frames]
    truth_indices = first_future_index + torch.arange(forecast_frames) * frame_stride
    valid_truth = truth_indices < num_stored
    ground_truth_para = sample["prs_para"].index_select(0, truth_indices[valid_truth])
    ground_truth_perp = sample["prs_perp"].index_select(0, truth_indices[valid_truth])
    prediction_times = _future_times(
        sample["profile_time"],
        first_index=first_future_index,
        count=forecast_frames,
        frame_stride=frame_stride,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sample_folder": sample["folder"],
            "checkpoint": str(checkpoint_path),
            "seed_indices": seed_indices,
            "seed_times": sample["profile_time"].index_select(0, seed_indices),
            "afsi_context": "pitch_averaged_radial_source",
            "afsi_filename": afsi_filename,
            "prediction_times": prediction_times,
            "predicted_prs_para": predicted_para,
            "predicted_prs_perp": predicted_perp,
            "ground_truth_indices": truth_indices[valid_truth],
            "ground_truth_prs_para": ground_truth_para,
            "ground_truth_prs_perp": ground_truth_perp,
            "input_frames": input_frames,
            "output_frames": output_frames,
            "frame_stride": frame_stride,
        },
        args.output,
    )
    print(
        f"Wrote {forecast_frames} forecast frames "
        f"({int(valid_truth.sum())} with ground truth) to {args.output}"
    )
    _cleanup_distributed()


if __name__ == "__main__":
    main()
