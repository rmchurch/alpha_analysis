import torch

from alpha_analysis.ai import train_transolver as train_transolver_module
from alpha_analysis.ai.train_transolver_timedependent import (
    DistributedEvalSampler,
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
