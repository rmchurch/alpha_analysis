import torch
from torch.utils.data import Dataset

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
    second = dataset[1]
    assert second["input_indices"].tolist() == [1, 2]
    assert second["target_indices"].tolist() == [3, 4]
    assert second["input_times"].tolist() == [0.25, 0.5]


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
