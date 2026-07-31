"""PyTorch dataset utilities for DESC/analysis result folders.

Each sample folder is expected to contain:
    - ``desc_equilibrium.h5``
    - ``analysis_results.h5``
    - optionally ``bfield.h5``

The dataset reads:
    - ``desc_equilibrium.h5``: ``_R_lmn`` and ``_Z_lmn``
    - ``analysis_results.h5``: ``profiles/prs_para``, ``profiles/prs_perp``, and
      the loss-energy target. The observed target path in this workspace is
      ``losses/losses/energy``, but ``losses/energy`` is also supported as a
      fallback.
    - when requested, ``bfield.h5``: ``br``, ``bphi``, and ``bz``.

Set ``include_target=False`` for self-supervised or temporal profile tasks that
do not use the scalar loss target.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import h5py
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

PathLike = Union[str, Path]

DEFAULT_ANALYSIS_FILENAME = "analysis_results.h5"
DEFAULT_EQUILIBRIUM_FILENAME = "desc_equilibrium.h5"
DEFAULT_BFIELD_FILENAME = "bfield.h5"
BFIELD_DATASETS = ("br", "bphi", "bz")
BFIELD_COORDINATE_DATASETS = ("rho", "theta", "phi")
PROFILE_TIME_DATASET = "profiles/time"
TARGET_DATASET_CANDIDATES = ("losses/losses/energy", "losses/energy")
DEFAULT_TARGET_DATABASE_KEY = "fraction_lost"

# Retained for callers that imported these names from this module before bfield
# loading switched to precomputed bfield.h5 files.
DEFAULT_ASCOT_FILENAME = "ascot_output.h5"
ASCOT_FILENAME_CANDIDATES = ("ascot_output.h5", "ascot_results.h5")


@dataclass(frozen=True)
class SamplePaths:
    """Filesystem locations for a single training sample."""

    folder: Path
    analysis_path: Path
    equilibrium_path: Path
    bfield_path: Union[Path, None]


def _resolve_sample_paths(
    folders: Iterable[PathLike],
    analysis_filename: str,
    equilibrium_filename: str,
    bfield_filename: str,
    include_bfield: bool,
    strict: bool,
) -> Tuple[List[SamplePaths], List[str]]:
    samples: List[SamplePaths] = []
    skipped: List[str] = []

    for folder in folders:
        folder_path = Path(folder).expanduser()
        try:
            if not folder_path.is_dir():
                raise FileNotFoundError(f"Sample folder does not exist: {folder_path}")
        except PermissionError as exc:
            if strict:
                raise PermissionError(f"Cannot access sample folder: {folder_path}") from exc
            skipped.append(f"{folder_path}: cannot access folder")
            continue

        analysis_path = folder_path / analysis_filename
        equilibrium_path = folder_path / equilibrium_filename
        bfield_path = folder_path / bfield_filename

        try:
            required_paths = [analysis_path, equilibrium_path]
            if include_bfield:
                required_paths.append(bfield_path)
            missing_files = [
                str(path.name)
                for path in required_paths
                if not path.is_file()
            ]
        except PermissionError as exc:
            required_names = [analysis_filename, equilibrium_filename]
            if include_bfield:
                required_names.append(bfield_filename)
            message = "{}: cannot access required files ({})".format(
                folder_path, ", ".join(required_names)
            )
            if strict:
                raise PermissionError(message) from exc
            skipped.append(message)
            continue
        if missing_files:
            message = f"{folder_path}: missing required files: {', '.join(missing_files)}"
            if strict:
                raise FileNotFoundError(message)
            skipped.append(message)
            continue

        samples.append(
            SamplePaths(
                folder=folder_path,
                analysis_path=analysis_path,
                equilibrium_path=equilibrium_path,
                bfield_path=bfield_path if include_bfield else None,
            )
        )

    if not samples:
        details = f" Skipped entries: {skipped}" if skipped else ""
        raise ValueError(f"No readable samples were found from the provided folders.{details}")

    return samples, skipped


def _read_required_dataset(h5_file: h5py.File, dataset_name: str) -> Tensor:
    if dataset_name not in h5_file:
        raise KeyError(f"Missing dataset '{dataset_name}' in {h5_file.filename}")
    return torch.as_tensor(h5_file[dataset_name][...], dtype=torch.float32)


def _read_target_dataset(h5_file: h5py.File) -> Tensor:
    for dataset_name in TARGET_DATASET_CANDIDATES:
        if dataset_name in h5_file:
            return torch.as_tensor(h5_file[dataset_name][...], dtype=torch.float32)
    raise KeyError(
        f"Missing target dataset in {h5_file.filename}. "
        f"Tried: {', '.join(TARGET_DATASET_CANDIDATES)}"
    )


def _read_fraction_lost_target(h5_file: h5py.File) -> Tensor:
    required = (
        "losses/initial/energy",
        "losses/initial/weight",
        "losses/losses/energy",
        "losses/losses/weight",
    )
    missing = [dataset_name for dataset_name in required if dataset_name not in h5_file]
    if missing:
        raise KeyError(
            f"Cannot derive fraction_lost from {h5_file.filename}; missing: {missing}"
        )

    initial_energy = torch.as_tensor(h5_file["losses/initial/energy"][...], dtype=torch.float64)
    initial_weight = torch.as_tensor(h5_file["losses/initial/weight"][...], dtype=torch.float64)
    lost_energy = torch.as_tensor(h5_file["losses/losses/energy"][...], dtype=torch.float64)
    lost_weight = torch.as_tensor(h5_file["losses/losses/weight"][...], dtype=torch.float64)
    initial_total = torch.sum(initial_energy * initial_weight)
    if not torch.isfinite(initial_total) or float(initial_total) <= 0.0:
        raise ValueError(f"Cannot derive fraction_lost from {h5_file.filename}; invalid denominator.")
    fraction_lost = torch.sum(lost_energy * lost_weight) / initial_total
    return fraction_lost.reshape(1).to(torch.float32)


def _sample_database_key(folder: Path) -> str:
    return folder.name.rsplit("_", 1)[-1]


def _load_target_database(database_path: Path, target_key: str) -> Dict[str, float]:
    with database_path.expanduser().open() as file:
        database = json.load(file)
    if not isinstance(database, dict):
        raise ValueError(f"Expected {database_path} to contain a JSON object.")

    targets: Dict[str, float] = {}
    for sample_key, sample_data in database.items():
        if isinstance(sample_data, dict) and target_key in sample_data:
            targets[str(sample_key)] = float(sample_data[target_key])
    if not targets:
        raise ValueError(f"No '{target_key}' values found in {database_path}.")
    return targets


def _read_bfield_file(bfield_path: Path) -> Dict[str, Tensor]:
    with h5py.File(bfield_path, "r") as bfield_file:
        bfield = {
            dataset_name: _read_required_dataset(bfield_file, dataset_name)
            for dataset_name in BFIELD_DATASETS
        }
        for dataset_name in BFIELD_COORDINATE_DATASETS:
            if dataset_name in bfield_file:
                bfield[dataset_name] = _read_required_dataset(bfield_file, dataset_name)
    if not (bfield["br"].shape == bfield["bphi"].shape == bfield["bz"].shape):
        raise ValueError(
            f"br, bphi, and bz must have matching shapes in {bfield_path}. "
            "Received {}, {}, and {}.".format(
                bfield["br"].shape,
                bfield["bphi"].shape,
                bfield["bz"].shape,
            )
        )
    return bfield


def _pad_tensor_batch(
    tensors: Sequence[Tensor],
    pad_value: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pad a list of equal-rank tensors to the maximum shape in the batch."""

    if not tensors:
        raise ValueError("Cannot pad an empty tensor batch.")

    rank = tensors[0].ndim
    for tensor in tensors:
        if tensor.ndim != rank:
            raise ValueError(
                "All tensors in a batch must have the same rank. "
                f"Observed ranks: {[item.ndim for item in tensors]}"
            )

    max_shape = [max(tensor.shape[dim] for tensor in tensors) for dim in range(rank)]

    padded_tensors: List[Tensor] = []
    masks: List[Tensor] = []
    shapes: List[Tensor] = []
    for tensor in tensors:
        shapes.append(torch.tensor(tensor.shape, dtype=torch.long))

        pad_widths: List[int] = []
        for dim in reversed(range(rank)):
            pad_widths.extend((0, max_shape[dim] - tensor.shape[dim]))
        padded_tensors.append(F.pad(tensor, pad_widths, value=pad_value))

        mask = torch.zeros(max_shape, dtype=torch.bool)
        mask[tuple(slice(0, size) for size in tensor.shape)] = True
        masks.append(mask)

    return (
        torch.stack(padded_tensors, dim=0),
        torch.stack(masks, dim=0),
        torch.stack(shapes, dim=0),
    )


class Ascot5Dataset(Dataset):
    """Lazy dataset for simulation folders containing DESC and analysis HDF5 files."""

    def __init__(
        self,
        folders: Iterable[PathLike],
        *,
        analysis_filename: str = DEFAULT_ANALYSIS_FILENAME,
        equilibrium_filename: str = DEFAULT_EQUILIBRIUM_FILENAME,
        bfield_filename: str = DEFAULT_BFIELD_FILENAME,
        ascot_filename: Union[str, None] = None,
        include_bfield: bool = False,
        strict: bool = True,
        target_database_path: Union[PathLike, None] = None,
        target_database_key: str = DEFAULT_TARGET_DATABASE_KEY,
        allow_missing_target_database: bool = False,
        include_target: bool = True,
    ) -> None:
        del ascot_filename
        self.include_bfield = include_bfield
        self.samples, self.skipped_folders = _resolve_sample_paths(
            folders=folders,
            analysis_filename=analysis_filename,
            equilibrium_filename=equilibrium_filename,
            bfield_filename=bfield_filename,
            include_bfield=include_bfield,
            strict=strict,
        )
        self.target_database: Union[Dict[str, float], None] = None
        self.include_target = include_target
        self.target_database_key = target_database_key
        self.allow_missing_target_database = allow_missing_target_database
        if include_target and target_database_path is not None:
            database_path = Path(target_database_path).expanduser()
            if not database_path.is_file():
                raise FileNotFoundError(f"Target database does not exist: {database_path}")
            self.target_database = _load_target_database(database_path, target_database_key)
            missing = [
                str(sample.folder)
                for sample in self.samples
                if _sample_database_key(sample.folder) not in self.target_database
            ]
            if missing:
                message = (
                    f"Missing '{target_database_key}' target database entries for "
                    f"{len(missing)} samples. First missing sample: {missing[0]}"
                )
                if strict and not allow_missing_target_database:
                    raise KeyError(message)
                if not allow_missing_target_database:
                    self.skipped_folders.extend(missing)
                    self.samples = [
                        sample
                        for sample in self.samples
                        if _sample_database_key(sample.folder) in self.target_database
                    ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_paths = self.samples[index]
        with h5py.File(sample_paths.equilibrium_path, "r") as equilibrium_file, h5py.File(
            sample_paths.analysis_path, "r"
        ) as analysis_file:
            r_lmn = _read_required_dataset(equilibrium_file, "_R_lmn")
            z_lmn = _read_required_dataset(equilibrium_file, "_Z_lmn")
            prs_para = _read_required_dataset(analysis_file, "profiles/prs_para")
            prs_perp = _read_required_dataset(analysis_file, "profiles/prs_perp")
            profile_time = (
                _read_required_dataset(analysis_file, PROFILE_TIME_DATASET)
                if PROFILE_TIME_DATASET in analysis_file
                else torch.arange(prs_para.shape[0], dtype=torch.float32)
            )
            if self.include_target and self.target_database is None:
                target = _read_target_dataset(analysis_file)
            elif self.include_target:
                sample_key = _sample_database_key(sample_paths.folder)
                if sample_key in self.target_database:
                    target = torch.tensor(
                        [self.target_database[sample_key]],
                        dtype=torch.float32,
                    )
                elif (
                    self.allow_missing_target_database
                    and self.target_database_key == DEFAULT_TARGET_DATABASE_KEY
                ):
                    target = _read_fraction_lost_target(analysis_file)
                else:
                    raise KeyError(
                        f"Missing '{self.target_database_key}' target database entry "
                        f"for {sample_paths.folder}"
                    )

        sample = {
            "folder": str(sample_paths.folder),
            "prs_para": prs_para,
            "prs_perp": prs_perp,
            "profile_time": profile_time,
            "context": {
                "R_lmn": r_lmn,
                "Z_lmn": z_lmn,
            },
        }
        if self.include_target:
            sample["target"] = target
        if self.include_bfield:
            if sample_paths.bfield_path is None:
                raise ValueError(f"No bfield file is configured for {sample_paths.folder}")
            sample["bfield"] = _read_bfield_file(sample_paths.bfield_path)
        return sample


def simulation_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad variable-length tensors and create masks for batching."""

    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    prs_para_batch, prs_para_mask, prs_para_shapes = _pad_tensor_batch(
        [item["prs_para"] for item in batch]  # type: ignore[index]
    )
    prs_perp_batch, prs_perp_mask, prs_perp_shapes = _pad_tensor_batch(
        [item["prs_perp"] for item in batch]  # type: ignore[index]
    )
    r_lmn_batch, r_lmn_mask, r_lmn_shapes = _pad_tensor_batch(
        [item["context"]["R_lmn"] for item in batch]  # type: ignore[index]
    )
    z_lmn_batch, z_lmn_mask, z_lmn_shapes = _pad_tensor_batch(
        [item["context"]["Z_lmn"] for item in batch]  # type: ignore[index]
    )
    target_batch, target_mask, target_shapes = _pad_tensor_batch(
        [item["target"] for item in batch]  # type: ignore[index]
    )
    has_bfield = "bfield" in batch[0]
    if any(("bfield" in item) != has_bfield for item in batch):
        raise ValueError("Mixed batches with and without bfield payloads are not supported.")

    if prs_para_batch.shape != prs_perp_batch.shape:
        raise ValueError(
            "profiles/prs_para and profiles/prs_perp must have matching shapes within a batch. "
            "Received padded shapes {} and {}.".format(prs_para_batch.shape, prs_perp_batch.shape)
        )

    collated = {
        "folders": [item["folder"] for item in batch],
        "input_channels": ("prs_para", "prs_perp"),
        "inputs": torch.stack((prs_para_batch, prs_perp_batch), dim=1),
        "input_mask": torch.stack((prs_para_mask, prs_perp_mask), dim=1),
        "context": {
            "R_lmn": r_lmn_batch,
            "R_lmn_mask": r_lmn_mask,
            "Z_lmn": z_lmn_batch,
            "Z_lmn_mask": z_lmn_mask,
        },
        "target": target_batch,
        "target_mask": target_mask,
        "shapes": {
            "prs_para": prs_para_shapes,
            "prs_perp": prs_perp_shapes,
            "R_lmn": r_lmn_shapes,
            "Z_lmn": z_lmn_shapes,
            "target": target_shapes,
        },
    }
    if has_bfield:
        br_batch, br_mask, br_shapes = _pad_tensor_batch(
            [item["bfield"]["br"] for item in batch]  # type: ignore[index]
        )
        bphi_batch, bphi_mask, bphi_shapes = _pad_tensor_batch(
            [item["bfield"]["bphi"] for item in batch]  # type: ignore[index]
        )
        bz_batch, bz_mask, bz_shapes = _pad_tensor_batch(
            [item["bfield"]["bz"] for item in batch]  # type: ignore[index]
        )
        if not (br_batch.shape == bphi_batch.shape == bz_batch.shape):
            raise ValueError(
                "br, bphi, and bz must have matching shapes within a batch. "
                f"Received {br_batch.shape}, {bphi_batch.shape}, and {bz_batch.shape}."
            )
        collated["bfield_channels"] = ("br", "bphi", "bz")
        collated["bfield"] = torch.stack((br_batch, bphi_batch, bz_batch), dim=1)
        collated["bfield_mask"] = torch.stack((br_mask, bphi_mask, bz_mask), dim=1)
        collated["shapes"]["br"] = br_shapes
        collated["shapes"]["bphi"] = bphi_shapes
        collated["shapes"]["bz"] = bz_shapes
    return collated


def build_simulation_dataloader(
    folders: Iterable[PathLike],
    *,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    strict: bool = True,
    analysis_filename: str = DEFAULT_ANALYSIS_FILENAME,
    equilibrium_filename: str = DEFAULT_EQUILIBRIUM_FILENAME,
    bfield_filename: str = DEFAULT_BFIELD_FILENAME,
    ascot_filename: Union[str, None] = None,
    include_bfield: bool = False,
    target_database_path: Union[PathLike, None] = None,
    target_database_key: str = DEFAULT_TARGET_DATABASE_KEY,
) -> DataLoader:
    """Create a DataLoader for a list of simulation result folders."""

    dataset = Ascot5Dataset(
        folders,
        analysis_filename=analysis_filename,
        equilibrium_filename=equilibrium_filename,
        bfield_filename=bfield_filename,
        ascot_filename=ascot_filename,
        include_bfield=include_bfield,
        strict=strict,
        target_database_path=target_database_path,
        target_database_key=target_database_key,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=simulation_collate_fn,
    )


__all__ = [
    "ASCOT_FILENAME_CANDIDATES",
    "BFIELD_COORDINATE_DATASETS",
    "BFIELD_DATASETS",
    "DEFAULT_ANALYSIS_FILENAME",
    "DEFAULT_ASCOT_FILENAME",
    "DEFAULT_BFIELD_FILENAME",
    "DEFAULT_EQUILIBRIUM_FILENAME",
    "DEFAULT_TARGET_DATABASE_KEY",
    "PROFILE_TIME_DATASET",
    "Ascot5Dataset",
    "TARGET_DATASET_CANDIDATES",
    "build_simulation_dataloader",
    "simulation_collate_fn",
]
