from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetSplit:
    inputs: torch.Tensor
    targets: torch.Tensor


class ModularAdditionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.inputs = inputs.long()
        self.targets = targets.long()

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def create_data_splits(
    prime: int,
    train_fraction: float = 0.3,
    seed: int = 0,
) -> tuple[DatasetSplit, DatasetSplit]:
    if prime <= 2:
        raise ValueError("prime must be greater than 2")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    equals_token = prime
    all_examples = []
    all_targets = []

    for a in range(prime):
        for b in range(prime):
            all_examples.append([a, b, equals_token])
            all_targets.append((a + b) % prime)

    inputs = torch.tensor(all_examples, dtype=torch.long)
    targets = torch.tensor(all_targets, dtype=torch.long)

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(inputs.size(0), generator=generator)
    train_size = int(inputs.size(0) * train_fraction)
    train_idx = permutation[:train_size]
    test_idx = permutation[train_size:]

    return (
        DatasetSplit(inputs[train_idx], targets[train_idx]),
        DatasetSplit(inputs[test_idx], targets[test_idx]),
    )
