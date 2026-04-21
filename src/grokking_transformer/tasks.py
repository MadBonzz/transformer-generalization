from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import Dataset


OperatorFn = Callable[[int, int, int], int]


def _mod_inverse(value: int, modulus: int) -> int:
    if math.gcd(value, modulus) != 1:
        raise ValueError(f"{value} has no modular inverse mod {modulus}")
    return pow(value, -1, modulus)


def apply_operator(operator: str, a: int, b: int, modulus: int) -> int:
    if operator == "add":
        return (a + b) % modulus
    if operator == "sub":
        return (a - b) % modulus
    if operator == "mul":
        return (a * b) % modulus
    if operator == "div":
        if b == 0:
            raise ValueError("division task requires excluding zero from the second operand")
        return (a * _mod_inverse(b, modulus)) % modulus
    if operator == "poly":
        return (a * a + a * b + b * b) % modulus
    raise ValueError(f"unsupported operator={operator}")


@dataclass(frozen=True)
class DatasetInfo:
    vocab_size: int
    target_vocab_size: int
    seq_len: int
    eq_token_id: int
    operator_token_ids: dict[str, int]


class TaskDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        true_targets: torch.Tensor | None = None,
        corrupted_mask: torch.Tensor | None = None,
    ) -> None:
        self.inputs = inputs.long()
        self.targets = targets.long()
        self.true_targets = self.targets.clone() if true_targets is None else true_targets.long()
        self.corrupted_mask = (
            torch.zeros_like(self.targets, dtype=torch.bool)
            if corrupted_mask is None
            else corrupted_mask.bool()
        )

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


@dataclass(frozen=True)
class TaskSplit:
    train: TaskDataset
    test: TaskDataset
    info: DatasetInfo


@dataclass(frozen=True)
class RangeTransferSplit:
    train: TaskDataset
    test: TaskDataset
    cross_add: TaskDataset
    cross_mul: TaskDataset
    info: DatasetInfo


def _enumerate_domain(modulus: int, include_zero: bool) -> list[int]:
    start = 0 if include_zero else 1
    return list(range(start, modulus))


def _make_inputs(
    examples: list[list[int]],
    targets: list[int],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = torch.tensor(examples, dtype=torch.long)
    labels = torch.tensor(targets, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(inputs.size(0), generator=generator)
    train_size = int(inputs.size(0) * train_fraction)
    train_idx = permutation[:train_size]
    test_idx = permutation[train_size:]
    return inputs, labels, train_idx, test_idx


def _corrupt_targets(
    labels: torch.Tensor,
    *,
    target_vocab_size: int,
    corruption_fraction: float,
    seed: int,
    forbid_true_label: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if corruption_fraction <= 0.0:
        return labels.clone(), torch.zeros_like(labels, dtype=torch.bool)

    generator = torch.Generator().manual_seed(seed)
    num_items = labels.numel()
    num_corrupted = int(num_items * corruption_fraction)
    corruption_perm = torch.randperm(num_items, generator=generator)
    corrupted_indices = corruption_perm[:num_corrupted]
    corrupted_mask = torch.zeros_like(labels, dtype=torch.bool)
    corrupted_mask[corrupted_indices] = True
    corrupted_labels = labels.clone()

    replacements = torch.randint(0, target_vocab_size, (num_corrupted,), generator=generator)
    if forbid_true_label:
        true_subset = labels[corrupted_indices]
        collisions = replacements == true_subset
        while collisions.any():
            replacements[collisions] = torch.randint(
                0,
                target_vocab_size,
                (int(collisions.sum().item()),),
                generator=generator,
            )
            collisions = replacements == true_subset

    corrupted_labels[corrupted_indices] = replacements
    return corrupted_labels, corrupted_mask


def build_single_operator_task(
    *,
    modulus: int,
    operator: str,
    train_fraction: float,
    seed: int,
    corruption_fraction: float = 0.0,
    corruption_seed: int | None = None,
    include_zero: bool = True,
) -> TaskSplit:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    domain = _enumerate_domain(modulus, include_zero=include_zero)
    eq_token = modulus
    examples: list[list[int]] = []
    targets: list[int] = []

    for a in domain:
        for b in domain:
            if operator == "div" and b == 0:
                continue
            examples.append([a, b, eq_token])
            targets.append(apply_operator(operator, a, b, modulus))

    inputs, labels, train_idx, test_idx = _make_inputs(examples, targets, train_fraction=train_fraction, seed=seed)
    true_train_targets = labels[train_idx].clone()
    corrupted_train_targets, corrupted_mask = _corrupt_targets(
        true_train_targets,
        target_vocab_size=modulus,
        corruption_fraction=corruption_fraction,
        seed=seed if corruption_seed is None else corruption_seed,
    )

    train = TaskDataset(
        inputs[train_idx],
        corrupted_train_targets,
        true_targets=true_train_targets,
        corrupted_mask=corrupted_mask,
    )
    test = TaskDataset(inputs[test_idx], labels[test_idx], true_targets=labels[test_idx])
    info = DatasetInfo(
        vocab_size=modulus + 1,
        target_vocab_size=modulus,
        seq_len=3,
        eq_token_id=eq_token,
        operator_token_ids={},
    )
    return TaskSplit(train=train, test=test, info=info)


def build_range_transfer_task(
    *,
    modulus: int,
    output_modulus: int | None = None,
    train_fraction: float,
    seed: int,
    add_offset: int = 0,
    mul_offset: int = 1000,
    include_zero: bool = True,
) -> RangeTransferSplit:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    output_modulus = modulus if output_modulus is None else output_modulus
    domain = _enumerate_domain(modulus, include_zero=include_zero)
    max_numeric_token = max(add_offset + modulus - 1, mul_offset + modulus - 1)
    add_token = max_numeric_token + 1
    mul_token = max_numeric_token + 2
    eq_token = max_numeric_token + 3

    def tokenized_examples(offset: int, operator: str, operator_token: int) -> tuple[list[list[int]], list[int]]:
        examples: list[list[int]] = []
        targets: list[int] = []
        for a in domain:
            for b in domain:
                examples.append([offset + a, operator_token, offset + b, eq_token])
                targets.append(apply_operator(operator, a, b, output_modulus))
        return examples, targets

    add_examples, add_targets = tokenized_examples(add_offset, "add", add_token)
    mul_examples, mul_targets = tokenized_examples(mul_offset, "mul", mul_token)

    all_examples = add_examples + mul_examples
    all_targets = add_targets + mul_targets
    inputs, labels, train_idx, test_idx = _make_inputs(
        all_examples,
        all_targets,
        train_fraction=train_fraction,
        seed=seed,
    )

    train = TaskDataset(inputs[train_idx], labels[train_idx], true_targets=labels[train_idx])
    test = TaskDataset(inputs[test_idx], labels[test_idx], true_targets=labels[test_idx])

    cross_add_examples, cross_add_targets = tokenized_examples(mul_offset, "add", add_token)
    cross_mul_examples, cross_mul_targets = tokenized_examples(add_offset, "mul", mul_token)
    cross_add = TaskDataset(
        torch.tensor(cross_add_examples, dtype=torch.long),
        torch.tensor(cross_add_targets, dtype=torch.long),
    )
    cross_mul = TaskDataset(
        torch.tensor(cross_mul_examples, dtype=torch.long),
        torch.tensor(cross_mul_targets, dtype=torch.long),
    )

    info = DatasetInfo(
        vocab_size=eq_token + 1,
        target_vocab_size=output_modulus,
        seq_len=4,
        eq_token_id=eq_token,
        operator_token_ids={"add": add_token, "mul": mul_token},
    )
    return RangeTransferSplit(train=train, test=test, cross_add=cross_add, cross_mul=cross_mul, info=info)
