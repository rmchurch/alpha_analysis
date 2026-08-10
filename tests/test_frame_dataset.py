import h5py
import numpy as np
import torch

from alpha_analysis.ai.frame_dataset import (
    AscotFrameDataset,
    AscotSequenceDataset,
    fit_frame_normalization,
    split_simulation_cohorts,
)


def _sample(root, name, frames):
    folder = root / name
    folder.mkdir()
    shape = (frames, 2, 3, 4)
    with h5py.File(folder / "analysis_results.h5", "w") as file:
        file["profiles/prs_para"] = np.arange(np.prod(shape)).reshape(shape)
        file["profiles/prs_perp"] = np.arange(np.prod(shape)).reshape(shape) + 100
        file["losses/initial/energy"] = [1.0]
        file["losses/initial/weight"] = [1.0]
        file["losses/losses/energy"] = [0.2]
        file["losses/losses/weight"] = [1.0]
    with h5py.File(folder / "desc_equilibrium.h5", "w") as file:
        file["_R_lmn"] = [1, 2]
        file["_Z_lmn"] = [3, 4]
    with h5py.File(folder / "bfield.h5", "w") as file:
        for key, offset in zip(("br", "bphi", "bz", "rho", "theta", "phi"), range(6)):
            file[key] = np.ones(shape[1:]) * offset
    return folder


def test_sequence_frame_shape_channel_order_and_variable_t(tmp_path):
    folders = [_sample(tmp_path, "G1600_00000", 2), _sample(tmp_path, "G1600_00001", 3)]
    sequences = AscotSequenceDataset(folders)
    assert sequences[0]["profiles"].shape == (2, 24, 2)
    assert sequences[1]["profiles"].shape == (3, 24, 2)
    assert torch.equal(
        sequences[0]["profiles"][..., 0].reshape(2, 2, 3, 4),
        torch.arange(48).reshape(2, 2, 3, 4).float(),
    )
    frames = AscotFrameDataset(sequences, [1, 0])
    assert len(frames) == 5
    assert (frames[0]["simulation_index"], frames[0]["frame_index"]) == (1, 0)
    loaded_frame = sequences.load_frame(1, 2)
    assert sequences.frame_count(1) == 3
    assert torch.equal(loaded_frame["profile"], sequences[1]["profiles"][2])


def test_sequence_expands_one_dimensional_coordinate_axes(tmp_path):
    folder = _sample(tmp_path, "G1600_00000", 2)
    with h5py.File(folder / "bfield.h5", "a") as file:
        del file["rho"]
        del file["theta"]
        del file["phi"]
        file["rho"] = np.arange(2, dtype=np.float32)
        file["theta"] = np.arange(3, dtype=np.float32)
        file["phi"] = np.arange(4, dtype=np.float32)

    sample = AscotSequenceDataset([folder])[0]
    expected = (
        torch.stack(
            torch.meshgrid(
                torch.arange(2), torch.arange(3), torch.arange(4), indexing="ij"
            ),
            dim=-1,
        )
        .reshape(-1, 3)
        .float()
    )
    assert torch.equal(sample["coordinates"], expected)


def test_split_normalization_and_sampling_are_isolated(tmp_path):
    folders = [_sample(tmp_path, f"G1600_{i:05d}", 2) for i in range(4)]
    split = split_simulation_cohorts(
        folders, training_sample_count=3, train_fraction=2 / 3, seed=0
    )
    assert set(split["train"]).isdisjoint(split["val"])
    assert split["later"] == [3]
    sequences = AscotSequenceDataset(folders)
    stats = fit_frame_normalization(sequences, split["train"])
    frames = AscotFrameDataset(
        sequences, split["val"], normalization=stats, max_nodes=7, seed=4
    )
    first, again = frames[0], frames[0]
    assert torch.equal(first["node_indices"], again["node_indices"])
    assert first["profile"].shape == first["coordinates"].shape[:-1] + (2,)
