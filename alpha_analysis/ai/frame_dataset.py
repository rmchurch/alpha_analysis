"""Per-simulation and per-frame ASCOT datasets for latent dynamics.

The historical G1600 run was fit to the first 578 readable simulations.  Newer
folders are deliberately kept as a ``later`` cohort: they may be encoded and
predicted, but must never affect fitting or normalization statistics.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .dataloader import (
    DEFAULT_ANALYSIS_FILENAME,
    DEFAULT_BFIELD_FILENAME,
    DEFAULT_EQUILIBRIUM_FILENAME,
    _read_fraction_lost_target,
    _read_required_dataset,
)

LEGACY_TRAINING_SAMPLE_COUNT = 578
TIME_DATASET_CANDIDATES = (
    "profiles/time",
    "profiles/times",
    "profiles/time_s",
    "time",
    "times",
)


def signed_log1p(value: Tensor) -> Tensor:
    return torch.sign(value) * torch.log1p(torch.abs(value))


def signed_expm1(value: Tensor) -> Tensor:
    return torch.sign(value) * torch.expm1(torch.abs(value))


def _read_coordinate_grid(
    bfield_file: h5py.File, grid_shape: tuple[int, ...], folder: Path
) -> list[Tensor]:
    if not all(key in bfield_file for key in ("rho", "theta", "phi")):
        axes = [torch.linspace(0.0, 1.0, size) for size in grid_shape]
        mesh = list(torch.meshgrid(*axes, indexing="ij"))
        while len(mesh) < 3:
            mesh.append(torch.zeros(grid_shape))
        return mesh[:3]

    coordinate_values = [
        _read_required_dataset(bfield_file, key) for key in ("rho", "theta", "phi")
    ]
    coordinate_shapes = [tuple(value.shape) for value in coordinate_values]
    if all(shape == grid_shape for shape in coordinate_shapes):
        return coordinate_values
    if (
        len(grid_shape) == 3
        and all(value.ndim == 1 for value in coordinate_values)
        and tuple(value.numel() for value in coordinate_values) == grid_shape
    ):
        return list(torch.meshgrid(*coordinate_values, indexing="ij"))
    raise ValueError(
        "Coordinate datasets must either match the field grid or be "
        f"one-dimensional grid axes in {folder}; got "
        f"{coordinate_shapes} for grid {grid_shape}"
    )


@dataclass
class FrameNormalization:
    profile_mean: Tensor
    profile_std: Tensor
    bfield_mean: Tensor
    bfield_std: Tensor
    coordinate_mean: Tensor
    coordinate_std: Tensor
    time_convention: str = "physical-if-present-else-normalized-frame-index"

    def state_dict(self) -> dict[str, Any]:
        return {
            "profile_mean": self.profile_mean.cpu(),
            "profile_std": self.profile_std.cpu(),
            "bfield_mean": self.bfield_mean.cpu(),
            "bfield_std": self.bfield_std.cpu(),
            "coordinate_mean": self.coordinate_mean.cpu(),
            "coordinate_std": self.coordinate_std.cpu(),
            "time_convention": self.time_convention,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "FrameNormalization":
        return cls(**{key: value for key, value in state.items()})

    def transform_profile(self, profile: Tensor) -> Tensor:
        return (signed_log1p(profile) - self.profile_mean) / self.profile_std

    def inverse_profile(self, profile: Tensor) -> Tensor:
        return signed_expm1(profile * self.profile_std + self.profile_mean)

    def transform_bfield(self, value: Tensor) -> Tensor:
        return (value - self.bfield_mean) / self.bfield_std

    def transform_coordinates(self, value: Tensor) -> Tensor:
        return (value - self.coordinate_mean) / self.coordinate_std


def discover_simulation_folders(results_root: Path) -> list[Path]:
    folders: list[Path] = []
    for folder in sorted(results_root.expanduser().iterdir()):
        try:
            if all(
                (folder / name).is_file()
                for name in (
                    DEFAULT_ANALYSIS_FILENAME,
                    DEFAULT_EQUILIBRIUM_FILENAME,
                    DEFAULT_BFIELD_FILENAME,
                )
            ):
                folders.append(folder)
        except PermissionError:
            continue
    if not folders:
        raise ValueError(f"No complete simulation folders found under {results_root}")
    return folders


def split_simulation_cohorts(
    folders: Sequence[Path],
    *,
    training_sample_count: int = LEGACY_TRAINING_SAMPLE_COUNT,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> dict[str, list[int]]:
    """Split only the historical cohort; reserve all later simulations."""
    if training_sample_count <= 0 or training_sample_count > len(folders):
        raise ValueError("training_sample_count must be in [1, len(folders)]")
    initial = list(range(training_sample_count))
    random.Random(seed).shuffle(initial)
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError("train_fraction must be in (0, 1]")
    cut = (
        len(initial)
        if train_fraction == 1.0
        else max(1, min(len(initial) - 1, int(len(initial) * train_fraction)))
    )
    return {
        "train": initial[:cut],
        "val": initial[cut:],
        "later": list(range(training_sample_count, len(folders))),
    }


def _flatten_grid(value: Tensor, grid_shape: tuple[int, ...], channels: int) -> Tensor:
    if tuple(value.shape[-len(grid_shape) :]) != grid_shape:
        raise ValueError(
            f"Expected trailing grid {grid_shape}, got {tuple(value.shape)}"
        )
    return value.reshape(channels, -1).transpose(0, 1).contiguous()


class AscotSequenceDataset(Dataset):
    """One item per ASCOT simulation, with explicit ``[T,N,2]`` profiles."""

    def __init__(
        self,
        folders: Iterable[str | Path],
        *,
        strict: bool = True,
        include_target: bool = True,
    ) -> None:
        self.folders = [Path(folder).expanduser() for folder in folders]
        self.strict = strict
        self.include_target = include_target
        if not self.folders:
            raise ValueError("AscotSequenceDataset requires at least one folder")

    def __len__(self) -> int:
        return len(self.folders)

    def frame_count(self, index: int) -> int:
        folder = self.folders[index]
        with h5py.File(folder / DEFAULT_ANALYSIS_FILENAME, "r") as analysis:
            para = analysis["profiles/prs_para"]
            perp = analysis["profiles/prs_perp"]
            if para.shape != perp.shape or para.ndim < 2:
                raise ValueError(
                    f"profiles must have matching [T,...grid] shapes in {folder}; "
                    f"got {tuple(para.shape)} and {tuple(perp.shape)}"
                )
            return int(para.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        folder = self.folders[index]
        with (
            h5py.File(folder / DEFAULT_ANALYSIS_FILENAME, "r") as analysis,
            h5py.File(folder / DEFAULT_EQUILIBRIUM_FILENAME, "r") as equilibrium,
            h5py.File(folder / DEFAULT_BFIELD_FILENAME, "r") as bfield_file,
        ):
            para = _read_required_dataset(analysis, "profiles/prs_para")
            perp = _read_required_dataset(analysis, "profiles/prs_perp")
            if para.shape != perp.shape or para.ndim < 2:
                raise ValueError(
                    f"profiles must have matching [T,...grid] shapes in {folder}; "
                    f"got {tuple(para.shape)} and {tuple(perp.shape)}"
                )
            time_count = para.shape[0]
            grid_shape = tuple(int(size) for size in para.shape[1:])
            field_values = [
                _read_required_dataset(bfield_file, key) for key in ("br", "bphi", "bz")
            ]
            if any(tuple(value.shape) != grid_shape for value in field_values):
                raise ValueError(
                    f"B field grid does not match profile grid in {folder}"
                )
            coordinate_values = _read_coordinate_grid(bfield_file, grid_shape, folder)
            physical_times = None
            for key in TIME_DATASET_CANDIDATES:
                if key in analysis:
                    candidate = torch.as_tensor(
                        analysis[key][...], dtype=torch.float32
                    ).reshape(-1)
                    if (
                        candidate.numel() == time_count
                        and torch.isfinite(candidate).all()
                    ):
                        physical_times = candidate
                        break
            times = (
                physical_times
                if physical_times is not None
                else torch.linspace(0.0, 1.0, time_count)
            )
            target = (
                _read_fraction_lost_target(analysis)
                if self.include_target
                else torch.zeros(1, dtype=torch.float32)
            )
            context = {
                "R_lmn": _read_required_dataset(equilibrium, "_R_lmn"),
                "Z_lmn": _read_required_dataset(equilibrium, "_Z_lmn"),
            }
        profiles = torch.stack((para, perp), dim=-1).reshape(time_count, -1, 2)
        coordinates = torch.stack(coordinate_values, dim=-1).reshape(-1, 3)
        bfield = torch.stack(field_values, dim=-1).reshape(-1, 3)
        for name, value in {
            "profiles": profiles,
            "coordinates": coordinates,
            "bfield": bfield,
        }.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite {name} in {folder}")
        return {
            "folder": str(folder),
            "coordinates": coordinates,
            "bfield": bfield,
            "profiles": profiles,
            "times": times,
            "context": context,
            "target": target.reshape(()),
            "grid_shape": grid_shape,
            "time_convention": "physical"
            if physical_times is not None
            else "normalized_frame_index",
        }

    def load_frame(self, index: int, frame_index: int) -> dict[str, Any]:
        """Read one profile frame without materializing its entire time sequence."""
        folder = self.folders[index]
        with (
            h5py.File(folder / DEFAULT_ANALYSIS_FILENAME, "r") as analysis,
            h5py.File(folder / DEFAULT_EQUILIBRIUM_FILENAME, "r") as equilibrium,
            h5py.File(folder / DEFAULT_BFIELD_FILENAME, "r") as bfield_file,
        ):
            para_dataset = analysis["profiles/prs_para"]
            perp_dataset = analysis["profiles/prs_perp"]
            if para_dataset.shape != perp_dataset.shape or para_dataset.ndim < 2:
                raise ValueError(
                    f"profiles must have matching [T,...grid] shapes in {folder}; "
                    f"got {tuple(para_dataset.shape)} and {tuple(perp_dataset.shape)}"
                )
            time_count = int(para_dataset.shape[0])
            if frame_index < 0 or frame_index >= time_count:
                raise IndexError(
                    f"Frame {frame_index} is outside [0, {time_count}) for {folder}"
                )
            grid_shape = tuple(int(size) for size in para_dataset.shape[1:])
            para = torch.as_tensor(para_dataset[frame_index], dtype=torch.float32)
            perp = torch.as_tensor(perp_dataset[frame_index], dtype=torch.float32)
            field_values = [
                _read_required_dataset(bfield_file, key) for key in ("br", "bphi", "bz")
            ]
            if any(tuple(value.shape) != grid_shape for value in field_values):
                raise ValueError(
                    f"B field grid does not match profile grid in {folder}"
                )
            coordinate_values = _read_coordinate_grid(bfield_file, grid_shape, folder)
            time = torch.tensor(
                frame_index / max(time_count - 1, 1), dtype=torch.float32
            )
            for key in TIME_DATASET_CANDIDATES:
                if key in analysis:
                    candidate = torch.as_tensor(
                        analysis[key][...], dtype=torch.float32
                    ).reshape(-1)
                    if (
                        candidate.numel() == time_count
                        and torch.isfinite(candidate).all()
                    ):
                        time = candidate[frame_index]
                        break
            target = (
                _read_fraction_lost_target(analysis).reshape(())
                if self.include_target
                else torch.tensor(0.0)
            )
            context = {
                "R_lmn": _read_required_dataset(equilibrium, "_R_lmn"),
                "Z_lmn": _read_required_dataset(equilibrium, "_Z_lmn"),
            }
        profile = torch.stack((para, perp), dim=-1).reshape(-1, 2)
        coordinates = torch.stack(coordinate_values, dim=-1).reshape(-1, 3)
        bfield = torch.stack(field_values, dim=-1).reshape(-1, 3)
        for name, value in {
            "profile": profile,
            "coordinates": coordinates,
            "bfield": bfield,
        }.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite {name} in {folder}")
        return {
            "folder": str(folder),
            "time": time,
            "coordinates": coordinates,
            "bfield": bfield,
            "profile": profile,
            "context": context,
            "target": target,
            "grid_shape": grid_shape,
        }


class AscotFrameDataset(Dataset):
    """A deterministic frame view over fixed simulation indices."""

    def __init__(
        self,
        sequences: AscotSequenceDataset,
        simulation_indices: Sequence[int],
        *,
        normalization: FrameNormalization | None = None,
        max_nodes: int | None = None,
        seed: int = 0,
        training: bool = False,
    ) -> None:
        if len(set(simulation_indices)) != len(simulation_indices):
            raise ValueError("simulation_indices contains duplicates")
        self.sequences = sequences
        self.simulation_indices = list(simulation_indices)
        self.normalization = normalization
        self.max_nodes = max_nodes
        self.seed = seed
        self.training = training
        self.frame_map: list[tuple[int, int]] = []
        for simulation_index in self.simulation_indices:
            self.frame_map.extend(
                (simulation_index, frame)
                for frame in range(sequences.frame_count(simulation_index))
            )

    def __len__(self) -> int:
        return len(self.frame_map)

    def __getitem__(self, index: int) -> dict[str, Any]:
        simulation_index, frame_index = self.frame_map[index]
        frame = self.sequences.load_frame(simulation_index, frame_index)
        profile = frame["profile"]
        coordinates, bfield = frame["coordinates"], frame["bfield"]
        node_indices = None
        if self.max_nodes is not None and coordinates.shape[0] > self.max_nodes:
            extra = random.randrange(2**31) if self.training else 0
            generator = torch.Generator().manual_seed(
                self.seed + simulation_index * 1009 + frame_index + extra
            )
            node_indices = (
                torch.randperm(coordinates.shape[0], generator=generator)[
                    : self.max_nodes
                ]
                .sort()
                .values
            )
            coordinates, bfield, profile = (
                coordinates[node_indices],
                bfield[node_indices],
                profile[node_indices],
            )
        if self.normalization is not None:
            profile = self.normalization.transform_profile(profile)
            coordinates = self.normalization.transform_coordinates(coordinates)
            bfield = self.normalization.transform_bfield(bfield)
        return {
            "folder": frame["folder"],
            "simulation_index": simulation_index,
            "frame_index": frame_index,
            "time": frame["time"],
            "coordinates": coordinates,
            "bfield": bfield,
            "profile": profile,
            "context": frame["context"],
            "target": frame["target"],
            "grid_shape": frame["grid_shape"],
            "node_indices": node_indices,
        }


def _stream_stats(chunks: Iterator[Tensor], channels: int) -> tuple[Tensor, Tensor]:
    count = 0
    total = torch.zeros(channels, dtype=torch.float64)
    total_sq = torch.zeros(channels, dtype=torch.float64)
    for chunk in chunks:
        flat = chunk.reshape(-1, channels).double()
        count += flat.shape[0]
        total += flat.sum(0)
        total_sq += flat.square().sum(0)
    if count == 0:
        raise ValueError("Cannot fit normalization on an empty training split")
    mean = total / count
    std = (total_sq / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
    if not (torch.isfinite(mean).all() and torch.isfinite(std).all()):
        raise ValueError("Nonfinite normalization statistics")
    return mean.float(), std.float()


def fit_frame_normalization(
    dataset: AscotSequenceDataset, train_indices: Sequence[int]
) -> FrameNormalization:
    # This function accepts train indices only by design: callers cannot accidentally
    # include validation or later simulations in fitted statistics.
    accumulators = {
        "profile": [
            0,
            torch.zeros(2, dtype=torch.float64),
            torch.zeros(2, dtype=torch.float64),
        ],
        "bfield": [
            0,
            torch.zeros(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
        ],
        "coordinates": [
            0,
            torch.zeros(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
        ],
    }
    for index in train_indices:
        sample = dataset[index]
        chunks = {
            "profile": signed_log1p(sample["profiles"]),
            "bfield": sample["bfield"],
            "coordinates": sample["coordinates"],
        }
        for name, chunk in chunks.items():
            flat = chunk.reshape(-1, chunk.shape[-1]).double()
            accumulators[name][0] += flat.shape[0]
            accumulators[name][1] += flat.sum(0)
            accumulators[name][2] += flat.square().sum(0)

    def finalize(name: str) -> tuple[Tensor, Tensor]:
        count, total, total_sq = accumulators[name]
        if count == 0:
            raise ValueError("Cannot fit normalization on an empty training split")
        mean = total / count
        std = (total_sq / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
        if not (torch.isfinite(mean).all() and torch.isfinite(std).all()):
            raise ValueError(f"Nonfinite {name} normalization statistics")
        return mean.float(), std.float()

    profile_mean, profile_std = finalize("profile")
    field_mean, field_std = finalize("bfield")
    coord_mean, coord_std = finalize("coordinates")
    return FrameNormalization(
        profile_mean, profile_std, field_mean, field_std, coord_mean, coord_std
    )


def save_split(
    path: Path, folders: Sequence[Path], splits: Mapping[str, Sequence[int]]
) -> None:
    path.write_text(
        json.dumps(
            {
                name: [str(folders[i]) for i in indices]
                for name, indices in splits.items()
            },
            indent=2,
        )
    )


__all__ = [
    "AscotSequenceDataset",
    "AscotFrameDataset",
    "FrameNormalization",
    "LEGACY_TRAINING_SAMPLE_COUNT",
    "discover_simulation_folders",
    "split_simulation_cohorts",
    "fit_frame_normalization",
    "signed_log1p",
]
