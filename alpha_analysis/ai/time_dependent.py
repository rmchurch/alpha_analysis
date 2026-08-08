"""Temporal window and tensor utilities for profile-field forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset


def signed_log1p(value: Tensor) -> Tensor:
    """Compress profile magnitudes while preserving their sign."""

    return torch.sign(value) * torch.log1p(torch.abs(value))


def signed_expm1(value: Tensor) -> Tensor:
    """Inverse of :func:`signed_log1p`."""

    return torch.sign(value) * torch.expm1(torch.abs(value))


def _validate_profiles(para: Tensor, perp: Tensor) -> None:
    if para.shape != perp.shape:
        raise ValueError(
            "Parallel and perpendicular profiles must have identical shapes; "
            f"received {tuple(para.shape)} and {tuple(perp.shape)}."
        )
    if para.ndim < 2:
        raise ValueError(
            "Profiles must have a leading time dimension and at least one spatial "
            f"dimension; received shape {tuple(para.shape)}."
        )


@dataclass(frozen=True)
class TemporalWindow:
    """Location of one input/target window in a simulation."""

    sample_index: int
    start: int


class TemporalWindowDataset(Dataset):
    """Expose temporal windows from a dataset of complete simulations.

    A window contains ``input_frames`` profile frames followed immediately by
    ``output_frames`` target frames. ``frame_stride`` controls the spacing
    between frames and ``window_stride`` controls the spacing between starts.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        input_frames: int = 1,
        output_frames: int = 1,
        frame_stride: int = 1,
        window_stride: int = 1,
    ) -> None:
        for name, value in (
            ("input_frames", input_frames),
            ("output_frames", output_frames),
            ("frame_stride", frame_stride),
            ("window_stride", window_stride),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive; received {value}.")

        self.dataset = dataset
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.frame_stride = frame_stride
        self.window_stride = window_stride
        self.windows: List[TemporalWindow] = []
        required_span = (input_frames + output_frames - 1) * frame_stride + 1

        sample_paths = getattr(dataset, "samples", None)
        for sample_index in range(len(dataset)):
            if sample_paths is not None:
                with h5py.File(sample_paths[sample_index].analysis_path, "r") as analysis_file:
                    para_shape = analysis_file["profiles/prs_para"].shape
                    perp_shape = analysis_file["profiles/prs_perp"].shape
                if para_shape != perp_shape or len(para_shape) < 2:
                    raise ValueError(
                        "Parallel and perpendicular profile datasets must have matching "
                        f"[time, *grid] shapes; received {para_shape} and {perp_shape}."
                    )
                num_frames = int(para_shape[0])
            else:
                sample = dataset[sample_index]
                para, perp = sample["prs_para"], sample["prs_perp"]
                _validate_profiles(para, perp)
                num_frames = int(para.shape[0])
            for start in range(0, num_frames - required_span + 1, window_stride):
                self.windows.append(TemporalWindow(sample_index, start))

        if not self.windows:
            raise ValueError(
                "No temporal windows are available. The simulations are shorter "
                f"than the requested span of {required_span} stored frames."
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        window = self.windows[index]
        all_indices = window.start + torch.arange(
            self.input_frames + self.output_frames, dtype=torch.long
        ) * self.frame_stride
        input_indices = all_indices[: self.input_frames]
        target_indices = all_indices[self.input_frames :]

        temporal_reader = getattr(self.dataset, "read_temporal_window", None)
        if callable(temporal_reader):
            return temporal_reader(
                window.sample_index,
                input_indices.tolist(),
                target_indices.tolist(),
            )

        sample = self.dataset[window.sample_index]

        result = dict(sample)
        result.update(
            {
                "input_prs_para": sample["prs_para"].index_select(0, input_indices),
                "input_prs_perp": sample["prs_perp"].index_select(0, input_indices),
                "target_prs_para": sample["prs_para"].index_select(0, target_indices),
                "target_prs_perp": sample["prs_perp"].index_select(0, target_indices),
                "input_times": sample["profile_time"].index_select(0, input_indices),
                "target_times": sample["profile_time"].index_select(0, target_indices),
                "input_indices": input_indices,
                "target_indices": target_indices,
            }
        )
        return result


def profiles_to_node_channels(
    para: Tensor,
    perp: Tensor,
    *,
    transform: bool,
) -> Tensor:
    """Convert ``[T, *grid]`` profile pairs to ``[nodes, 2*T]``.

    Channels are interleaved by frame:
    ``para(t0), perp(t0), para(t1), perp(t1), ...``.
    """

    _validate_profiles(para, perp)
    values = torch.stack((para.float(), perp.float()), dim=1)
    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if transform:
        values = signed_log1p(values)
    return values.reshape(values.shape[0] * 2, -1).transpose(0, 1).contiguous()


def node_channels_to_profiles(
    channels: Tensor,
    *,
    num_frames: int,
    grid_shape: Sequence[int],
    transformed: bool,
) -> Tuple[Tensor, Tensor]:
    """Convert ``[nodes, 2*T]`` channels back to two ``[T, *grid]`` fields."""

    expected = 2 * num_frames
    if channels.ndim != 2 or channels.shape[1] != expected:
        raise ValueError(
            f"Expected node channels [nodes, {expected}], received {tuple(channels.shape)}."
        )
    values = channels.transpose(0, 1).reshape(num_frames, 2, *grid_shape)
    if transformed:
        values = signed_expm1(values)
    return values[:, 0], values[:, 1]


def append_predicted_frames(
    history_para: Tensor,
    history_perp: Tensor,
    predicted_channels: Tensor,
    *,
    output_frames: int,
    grid_shape: Sequence[int],
    transformed: bool,
) -> Tuple[Tensor, Tensor]:
    """Append one multi-frame prediction block to an autoregressive history."""

    para, perp = node_channels_to_profiles(
        predicted_channels,
        num_frames=output_frames,
        grid_shape=grid_shape,
        transformed=transformed,
    )
    return (
        torch.cat((history_para, para), dim=0),
        torch.cat((history_perp, perp), dim=0),
    )


__all__ = [
    "TemporalWindow",
    "TemporalWindowDataset",
    "append_predicted_frames",
    "node_channels_to_profiles",
    "profiles_to_node_channels",
    "signed_expm1",
    "signed_log1p",
]
