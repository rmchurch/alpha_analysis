import math

import pytest
import torch

from workflow.evaluate_synthetic_ascot5 import (
    _metrics,
    _static_frame_count,
    profile_error_metrics,
    validate_split_folders,
)


def test_split_folder_validation_requires_disjoint_sets(tmp_path):
    train = [tmp_path / "sample_1"]
    val = [tmp_path / "sample_2"]
    later = [tmp_path / "sample_3"]
    validate_split_folders(train, val, later)

    with pytest.raises(ValueError, match="overlap"):
        validate_split_folders(train, val, [tmp_path / "sample_1"])


def test_static_frame_count_matches_scalar_transolver_layout():
    assert _static_frame_count({"space_dim": 26}) == 10
    with pytest.raises(ValueError, match="space_dim"):
        _static_frame_count({"space_dim": 25})


def test_profile_errors_ignore_seed_frame():
    truth_para = torch.ones((3, 2, 2))
    truth_perp = torch.full((3, 2, 2), 2.0)
    synthetic_para = truth_para.clone()
    synthetic_perp = truth_perp.clone()
    synthetic_para[0] = 1000.0
    synthetic_para[1:] += 1.0

    metrics = profile_error_metrics(
        synthetic_para,
        synthetic_perp,
        truth_para,
        truth_perp,
        seed_frames=1,
    )
    assert metrics["rmse"] == pytest.approx(math.sqrt(0.5))
    assert metrics["mae"] == pytest.approx(0.5)


def test_scalar_metrics():
    rows = [
        {"ground_truth_fraction_lost": 1.0, "prediction": 1.0},
        {"ground_truth_fraction_lost": 2.0, "prediction": 2.0},
    ]
    assert _metrics(rows, "prediction") == {"mse": 0.0, "mae": 0.0, "r2": 1.0}
