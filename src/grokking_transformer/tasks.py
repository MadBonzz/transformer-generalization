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


@dataclass(frozen=True)
class Study3Split:
    train: TaskDataset
    test: TaskDataset
    final_evals: dict[str, TaskDataset]
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
    eq_token = max_numeric_token + 1

    def tokenized_examples(offset: int, operator: str) -> tuple[list[list[int]], list[int]]:
        examples: list[list[int]] = []
        targets: list[int] = []
        for a in domain:
            for b in domain:
                examples.append([offset + a, offset + b, eq_token])
                targets.append(apply_operator(operator, a, b, output_modulus))
        return examples, targets

    add_examples, add_targets = tokenized_examples(add_offset, "add")
    mul_examples, mul_targets = tokenized_examples(mul_offset, "mul")

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

    cross_add_examples, cross_add_targets = tokenized_examples(mul_offset, "add")
    cross_mul_examples, cross_mul_targets = tokenized_examples(add_offset, "mul")
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
        seq_len=3,
        eq_token_id=eq_token,
        operator_token_ids={},
    )
    return RangeTransferSplit(train=train, test=test, cross_add=cross_add, cross_mul=cross_mul, info=info)


def _build_contiguous_sets(modulus: int, add_set_size: int | None) -> tuple[list[int], list[int]]:
    if add_set_size is None:
        add_set_size = (modulus + 1) // 2
    if not 0 < add_set_size < modulus:
        raise ValueError("add_set_size must be between 1 and modulus - 1")
    values = list(range(modulus))
    return values[:add_set_size], values[add_set_size:]


def _build_interleaved_sets(modulus: int, chunk_size: int) -> tuple[list[int], list[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    set_a: list[int] = []
    set_b: list[int] = []
    values = list(range(modulus))
    for chunk_index, start in enumerate(range(0, modulus, chunk_size)):
        chunk = values[start : start + chunk_size]
        if chunk_index % 2 == 0:
            set_a.extend(chunk)
        else:
            set_b.extend(chunk)
    if not set_a or not set_b:
        raise ValueError("interleaved split must produce two non-empty sets")
    return set_a, set_b


def _ordered_pairs(values_a: list[int], values_b: list[int]) -> list[tuple[int, int]]:
    return [(a, b) for a in values_a for b in values_b]


def _build_study3_sets(
    *,
    modulus: int,
    scenario: str,
    add_set_size: int | None,
    interleave_chunk_size: int | None,
) -> tuple[list[int], list[int]]:
    if scenario.startswith("interleaved"):
        if interleave_chunk_size is None:
            raise ValueError("interleave_chunk_size is required for interleaved scenarios")
        return _build_interleaved_sets(modulus, interleave_chunk_size)
    return _build_contiguous_sets(modulus, add_set_size)


def _build_study3_token_layout(
    *,
    modulus: int,
    use_task_token: bool,
) -> tuple[int | None, int | None, int, int, dict[str, int]]:
    max_numeric_token = modulus - 1
    if use_task_token:
        add_token = max_numeric_token + 1
        mul_token = max_numeric_token + 2
        eq_token = max_numeric_token + 3
        seq_len = 4
        operator_token_ids = {"add": add_token, "mul": mul_token}
        return add_token, mul_token, eq_token, seq_len, operator_token_ids

    eq_token = max_numeric_token + 1
    return None, None, eq_token, 3, {}


def build_study3_task(
    *,
    modulus: int,
    output_modulus: int | None = None,
    train_fraction: float,
    seed: int,
    scenario: str,
    use_task_token: bool,
    add_set_size: int | None = None,
    interleave_chunk_size: int | None = None,
) -> Study3Split:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    output_modulus = modulus if output_modulus is None else output_modulus
    set_a, set_b = _build_study3_sets(
        modulus=modulus,
        scenario=scenario,
        add_set_size=add_set_size,
        interleave_chunk_size=interleave_chunk_size,
    )
    add_token, mul_token, eq_token, seq_len, operator_token_ids = _build_study3_token_layout(
        modulus=modulus,
        use_task_token=use_task_token,
    )

    def encode_example(a: int, b: int, operator: str) -> list[int]:
        if use_task_token:
            operator_token = add_token if operator == "add" else mul_token
            assert operator_token is not None
            return [a, operator_token, b, eq_token]
        return [a, b, eq_token]

    def make_dataset_from_pairs(pairs: list[tuple[int, int]], operator: str) -> TaskDataset:
        examples = [encode_example(a, b, operator) for a, b in pairs]
        targets = [apply_operator(operator, a, b, output_modulus) for a, b in pairs]
        return TaskDataset(torch.tensor(examples, dtype=torch.long), torch.tensor(targets, dtype=torch.long))

    train_examples: list[list[int]] = []
    train_targets: list[int] = []

    def append_examples(pairs: list[tuple[int, int]], operator: str) -> None:
        for a, b in pairs:
            train_examples.append(encode_example(a, b, operator))
            train_targets.append(apply_operator(operator, a, b, output_modulus))

    aa_pairs = _ordered_pairs(set_a, set_a)
    bb_pairs = _ordered_pairs(set_b, set_b)
    ab_pairs = _ordered_pairs(set_a, set_b)
    ba_pairs = _ordered_pairs(set_b, set_a)
    cross_pairs = ab_pairs + ba_pairs
    all_values = list(range(modulus))
    all_pairs = _ordered_pairs(all_values, all_values)

    if scenario in {"partitioned_ops", "interleaved_partitioned_ops"}:
        append_examples(aa_pairs, "add")
        append_examples(bb_pairs, "mul")
        final_evals = {
            "reverse_add": make_dataset_from_pairs(bb_pairs, "add"),
            "reverse_mul": make_dataset_from_pairs(aa_pairs, "mul"),
            "cross_pair_add": make_dataset_from_pairs(cross_pairs, "add"),
            "cross_pair_mul": make_dataset_from_pairs(cross_pairs, "mul"),
        }
    elif scenario == "both_ops_within_set":
        append_examples(aa_pairs, "add")
        append_examples(aa_pairs, "mul")
        append_examples(bb_pairs, "add")
        append_examples(bb_pairs, "mul")
        final_evals = {
            "cross_pair_add": make_dataset_from_pairs(cross_pairs, "add"),
            "cross_pair_mul": make_dataset_from_pairs(cross_pairs, "mul"),
        }
    elif scenario == "both_ops_all_pairs":
        if not use_task_token:
            raise ValueError("both_ops_all_pairs requires use_task_token=True because add and mul labels would collide")
        append_examples(all_pairs, "add")
        append_examples(all_pairs, "mul")
        final_evals = {
            "cross_pair_add": make_dataset_from_pairs(cross_pairs, "add"),
            "cross_pair_mul": make_dataset_from_pairs(cross_pairs, "mul"),
        }
    else:
        raise ValueError(f"unsupported Study 3 scenario={scenario}")

    inputs, labels, train_idx, test_idx = _make_inputs(train_examples, train_targets, train_fraction=train_fraction, seed=seed)
    train = TaskDataset(inputs[train_idx], labels[train_idx], true_targets=labels[train_idx])
    test = TaskDataset(inputs[test_idx], labels[test_idx], true_targets=labels[test_idx])
    info = DatasetInfo(
        vocab_size=eq_token + 1,
        target_vocab_size=output_modulus,
        seq_len=seq_len,
        eq_token_id=eq_token,
        operator_token_ids=operator_token_ids,
    )
    return Study3Split(train=train, test=test, final_evals=final_evals, info=info)
