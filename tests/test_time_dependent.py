import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from alpha_analysis.ai import dataloader as dataloader_module
from alpha_analysis.ai.dataloader import Ascot5Dataset
from alpha_analysis.ai.time_dependent import (
    TemporalWindowDataset,
    append_predicted_frames,
    node_channels_to_profiles,
    profiles_to_node_channels,
)


class _SimulationDataset(Dataset):
    def __init__(self):
        values = torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3)
        self.sample = {
            "folder": "sample",
            "prs_para": values,
            "prs_perp": values + 100,
            "profile_time": torch.arange(5, dtype=torch.float32) * 0.25,
            "afsi": {
                "source": torch.arange(6, dtype=torch.float32).reshape(2, 3, 1),
                "rho": torch.tensor([0.25, 0.75]),
                "ekin": torch.arange(3, dtype=torch.float32),
                "xi": torch.tensor([0.0]),
            },
        }

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return self.sample


def test_temporal_windows_support_multi_input_and_output():
    dataset = TemporalWindowDataset(
        _SimulationDataset(),
        input_frames=2,
        output_frames=2,
        frame_stride=1,
        window_stride=1,
    )
    assert len(dataset) == 2
    first = dataset[0]
    assert first["input_indices"].tolist() == [0, 1]
    assert first["target_indices"].tolist() == [2, 3]
    second = dataset[1]
    assert second["input_indices"].tolist() == [1, 2]
    assert second["target_indices"].tolist() == [3, 4]
    assert second["input_times"].tolist() == [0.25, 0.5]

def test_ascot_temporal_windows_read_only_selected_frames_and_cache_static_data(
    tmp_path, monkeypatch
):
    folder = tmp_path / "sample"
    folder.mkdir()
    values = np.arange(5 * 2 * 3, dtype=np.float64).reshape(5, 2, 3)
    with h5py.File(folder / "analysis_results.h5", "w") as analysis_file:
        analysis_file.create_dataset("profiles/prs_para", data=values)
        analysis_file.create_dataset("profiles/prs_perp", data=values + 100)
        analysis_file.create_dataset("profiles/time", data=np.arange(5) * 0.25)
    with h5py.File(folder / "desc_equilibrium.h5", "w") as equilibrium_file:
        equilibrium_file.create_dataset("_R_lmn", data=np.arange(3))
        equilibrium_file.create_dataset("_Z_lmn", data=np.arange(3) + 10)
    with h5py.File(folder / "bfield.h5", "w") as bfield_file:
        for offset, name in enumerate(("br", "bphi", "bz")):
            bfield_file.create_dataset(name, data=np.full((2, 3), offset + 1.0))
    with h5py.File(folder / "afsi_initial.h5", "w") as afsi_file:
        distribution = afsi_file.create_group("afsi_distribution")
        values = np.arange(1 * 2 * 1 * 3 * 2 * 1 * 1).reshape(1, 2, 1, 3, 2, 1, 1)
        source = distribution.create_dataset("distribution_function", data=values)
        source.attrs["dimensions"] = '["phi", "rho", "theta", "ekin", "xi", "time", "charge"]'
        coordinates = distribution.create_group("coordinates")
        coordinates.create_dataset("rho", data=[0.25, 0.75])
        coordinates.create_dataset("rho_edges", data=[0.0, 0.5, 1.0])
        coordinates.create_dataset("ekin", data=[1.0, 2.0, 3.0])
        coordinates.create_dataset("ekin_edges", data=[0.5, 1.5, 2.5, 3.5])
        coordinates.create_dataset("xi", data=[-0.5, 0.5])

    bfield_reads = []
    original_read_bfield = dataloader_module._read_bfield_file

    def counted_read_bfield(path):
        bfield_reads.append(path)
        return original_read_bfield(path)

    monkeypatch.setattr(dataloader_module, "_read_bfield_file", counted_read_bfield)
    simulations = Ascot5Dataset(
        [folder],
        include_bfield=True,
        include_afsi=True,
        include_target=False,
        temporal_static_cache_size=2,
    )
    windows = TemporalWindowDataset(
        simulations,
        input_frames=2,
        output_frames=1,
    )

    first = windows[0]
    second = windows[1]
    assert second["input_indices"].tolist() == [1, 2]
    assert second["target_indices"].tolist() == [3]
    torch.testing.assert_close(second["input_prs_para"], torch.tensor(values[1:3]).float())
    torch.testing.assert_close(
        second["target_prs_perp"], torch.tensor(values[3:4] + 100).float()
    )
    assert second["input_times"].tolist() == [0.25, 0.5]
    assert "prs_para" not in second
    assert "context" not in second
    assert len(bfield_reads) == 1
    assert first["bfield"] is second["bfield"]
    assert first["afsi"] is second["afsi"]

    assert first["input_indices"].tolist() == [0, 1]
    assert first["afsi"]["source"].shape == (2, 3, 1)
    torch.testing.assert_close(
        first["afsi"]["source"],
        torch.tensor([[[0.5], [2.5], [4.5]], [[6.5], [8.5], [10.5]]]),
    )

    complete_sample = simulations[0]
    assert complete_sample["prs_para"].shape == (5, 2, 3)
    assert "context" in complete_sample


def test_profile_node_channel_round_trip():
    para = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    perp = para + 50
    channels = profiles_to_node_channels(para, perp, transform=True)
    restored_para, restored_perp = node_channels_to_profiles(
        channels,
        num_frames=2,
        grid_shape=(3, 4),
        transformed=True,
    )
    torch.testing.assert_close(restored_para, para)
    torch.testing.assert_close(restored_perp, perp)
    assert channels.shape == (12, 4)


def test_append_multi_frame_prediction():
    history_para = torch.zeros((2, 2, 2))
    history_perp = torch.ones((2, 2, 2))
    future_para = torch.full((2, 2, 2), 3.0)
    future_perp = torch.full((2, 2, 2), 4.0)
    channels = profiles_to_node_channels(future_para, future_perp, transform=False)
    para, perp = append_predicted_frames(
        history_para,
        history_perp,
        channels,
        output_frames=2,
        grid_shape=(2, 2),
        transformed=False,
    )
    assert para.shape == (4, 2, 2)
    torch.testing.assert_close(para[-2:], future_para)
    torch.testing.assert_close(perp[-2:], future_perp)
