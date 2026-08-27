import torch

from alpha_analysis.ai import train_transolver as train_transolver_module
from alpha_analysis.ai.train_transolver_timedependent import (
    AFSIContextTransolverModel,
    DistributedEvalSampler,
    afsi_source_to_context,
    masked_field_metrics,
)


def test_distributed_eval_sampler_partitions_without_duplicates():
    dataset = list(range(11))
    partitions = [
        list(DistributedEvalSampler(dataset, num_replicas=3, rank=rank))
        for rank in range(3)
    ]

    assert sorted(index for partition in partitions for index in partition) == list(
        range(len(dataset))
    )
    assert sum(len(partition) for partition in partitions) == len(dataset)


def test_masked_field_metrics_ignore_padded_nodes():
    prediction = torch.tensor([[[1.0, 3.0], [100.0, 100.0]]])
    target = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
    mask = torch.tensor([[True, False]])

    mse, mae = masked_field_metrics(prediction, target, mask)

    torch.testing.assert_close(mse, torch.tensor(2.5))
    torch.testing.assert_close(mae, torch.tensor(1.5))


def test_attention_patch_disables_activation_reduction_for_ddp():
    try:
        train_transolver_module.patch_transolver_attention_for_cuda(data_parallel=True)
        assert not train_transolver_module._REDUCE_ATTENTION_ACROSS_RANKS
    finally:
        train_transolver_module.patch_transolver_attention_for_cuda(data_parallel=False)

    assert train_transolver_module._REDUCE_ATTENTION_ACROSS_RANKS


def test_afsi_context_uses_energy_bin_widths_and_preserves_total_strength():
    source = torch.tensor(
        [
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]],
        ]
    )
    sample = {
        "afsi": {
            "source": source,
            "rho": torch.tensor([0.125, 0.625]),
            "rho_edges": torch.tensor([0.0, 0.25, 1.0]),
            "ekin_edges": torch.tensor([0.0, 1.0, 3.0, 6.0]),
        }
    }
    context, strength = afsi_source_to_context(sample, context_points=2)

    radial = torch.tensor([14.0, 32.0])  # sum S * [1, 2, 3]
    total = radial @ torch.tensor([0.25, 0.75])
    reference = total
    expected = torch.stack(
        (sample["afsi"]["rho"], torch.log1p(radial / reference)), dim=-1
    )
    torch.testing.assert_close(context, expected)
    torch.testing.assert_close(strength, torch.log1p(total).reshape(1))


def test_afsi_encoder_and_every_cross_attention_block_receive_gradients():
    train_transolver_module.patch_transolver_attention_for_cuda(data_parallel=True)
    try:
        model = AFSIContextTransolverModel(
            space_dim=8,
            fun_dim=0,
            out_dim=2,
            n_hidden=8,
            n_layers=2,
            n_head=2,
            slice_num=2,
            dropout=0.0,
            mlp_ratio=1,
            unified_pos=False,
            context_points=4,
            context_hidden_dim=64,
        )
        prediction = model(
            (
                torch.randn(1, 6, 8),
                torch.randn(1, 6, 3),
                torch.randn(1, 4, 2),
                torch.randn(1, 1),
            )
        )
        prediction.square().mean().backward()

        assert model.source_encoder[0].weight.grad is not None
        assert all(
            block.Attn.context_cross_attention.in_proj_weight.grad is not None
            for block in model.blocks
        )
    finally:
        train_transolver_module.patch_transolver_attention_for_cuda(data_parallel=False)
