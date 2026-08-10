import torch

from alpha_analysis.ai.latent_dynamics import (
    DynamicsConfig,
    ResidualLatentDynamics,
    persistence_rollout,
    rollout_mse,
)


def test_dynamics_rollout_shape_and_gradient():
    model = ResidualLatentDynamics(DynamicsConfig(4, 8, 16, 4, 1))
    initial = torch.randn(2, 4, 8, requires_grad=True)
    rollout = model.rollout(initial, torch.linspace(0, 1, 5))
    assert rollout.shape == (2, 5, 4, 8)
    rollout.sum().backward()
    assert initial.grad is not None


def test_persistence_baseline():
    initial = torch.randn(2, 4, 8)
    result = persistence_rollout(initial, 3)
    assert result.shape == (2, 3, 4, 8)
    assert torch.equal(result[:, 0], result[:, 2])
    assert torch.equal(rollout_mse(result, result), torch.zeros(2, 3))
