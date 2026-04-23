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


TWO_OPERATOR_SET = ("add", "mul")
FOUR_OPERATOR_SET = ("add", "sub", "mul", "div")


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


def _build_contiguous_n_sets(modulus: int, num_sets: int) -> list[list[int]]:
    if num_sets <= 0:
        raise ValueError("num_sets must be positive")
    base_size = modulus // num_sets
    remainder = modulus % num_sets
    values = list(range(modulus))
    groups: list[list[int]] = []
    offset = 0
    for index in range(num_sets):
        group_size = base_size + (1 if index < remainder else 0)
        if group_size <= 0:
            raise ValueError("num_sets is too large for the modulus")
        groups.append(values[offset : offset + group_size])
        offset += group_size
    return groups


def _study3_operators_for_scenario(scenario: str) -> tuple[str, ...]:
    if scenario in {
        "partitioned_ops",
        "interleaved_partitioned_ops",
        "both_ops_within_set",
        "both_ops_all_pairs",
        "all_pairs_operator_complement",
        "all_pairs_pair_split_both_ops",
    }:
        return TWO_OPERATOR_SET
    if scenario in {
        "four_set_missing_ops_within_set",
        "four_set_all_ops_within_set",
        "four_set_all_ops_all_pairs",
    }:
        return FOUR_OPERATOR_SET
    raise ValueError(f"unsupported Study 3 scenario={scenario}")


def _build_study3_groups(
    *,
    modulus: int,
    scenario: str,
    add_set_size: int | None,
    interleave_chunk_size: int | None,
) -> list[list[int]]:
    if scenario in {"all_pairs_operator_complement", "all_pairs_pair_split_both_ops"}:
        return [list(range(modulus))]
    if scenario in {
        "four_set_missing_ops_within_set",
        "four_set_all_ops_within_set",
        "four_set_all_ops_all_pairs",
    }:
        return _build_contiguous_n_sets(modulus, 4)
    if scenario.startswith("interleaved"):
        if interleave_chunk_size is None:
            raise ValueError("interleave_chunk_size is required for interleaved scenarios")
        set_a, set_b = _build_interleaved_sets(modulus, interleave_chunk_size)
        return [set_a, set_b]
    set_a, set_b = _build_contiguous_sets(modulus, add_set_size)
    return [set_a, set_b]


def _build_study3_token_layout(
    *,
    modulus: int,
    use_task_token: bool,
    operators: tuple[str, ...],
) -> tuple[dict[str, int], int, int]:
    max_numeric_token = modulus - 1
    if use_task_token:
        operator_token_ids: dict[str, int] = {}
        next_token_id = max_numeric_token + 1
        for operator in operators:
            operator_token_ids[operator] = next_token_id
            next_token_id += 1
        eq_token = next_token_id
        seq_len = 4
        return operator_token_ids, eq_token, seq_len

    eq_token = max_numeric_token + 1
    return {}, eq_token, 3


def _filter_pairs_for_operator(pairs: list[tuple[int, int]], operator: str) -> list[tuple[int, int]]:
    if operator != "div":
        return pairs
    return [(a, b) for a, b in pairs if b != 0]


def _cross_pairs_across_groups(groups: list[list[int]]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for source_index, group_a in enumerate(groups):
        for target_index, group_b in enumerate(groups):
            if source_index == target_index:
                continue
            pairs.extend(_ordered_pairs(group_a, group_b))
    return pairs


def _build_dataset_from_examples(
    examples: list[list[int]],
    targets: list[int],
) -> TaskDataset:
    return TaskDataset(torch.tensor(examples, dtype=torch.long), torch.tensor(targets, dtype=torch.long))


def _split_items_evenly(
    items: list[tuple[int, int]],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(items), generator=generator).tolist()
    train_size = int(len(items) * train_fraction)
    train_items = [items[index] for index in permutation[:train_size]]
    test_items = [items[index] for index in permutation[train_size:]]
    return train_items, test_items


def _study3_example_count(
    *,
    modulus: int,
    scenario: str,
    add_set_size: int | None,
    interleave_chunk_size: int | None,
) -> int:
    groups = _build_study3_groups(
        modulus=modulus,
        scenario=scenario,
        add_set_size=add_set_size,
        interleave_chunk_size=interleave_chunk_size,
    )
    if scenario == "all_pairs_operator_complement":
        return modulus * modulus
    if scenario == "all_pairs_pair_split_both_ops":
        return modulus * modulus

    operators = _study3_operators_for_scenario(scenario)
    within_group_pairs = [_ordered_pairs(group, group) for group in groups]
    if scenario in {"partitioned_ops", "interleaved_partitioned_ops"}:
        return len(within_group_pairs[0]) + len(within_group_pairs[1])
    if scenario == "both_ops_within_set":
        return sum(len(_filter_pairs_for_operator(pairs, operator)) for pairs in within_group_pairs for operator in operators)
    if scenario == "both_ops_all_pairs":
        all_pairs = _ordered_pairs(list(range(modulus)), list(range(modulus)))
        return sum(len(_filter_pairs_for_operator(all_pairs, operator)) for operator in operators)
    if scenario == "four_set_missing_ops_within_set":
        return sum(
            len(_filter_pairs_for_operator(within_group_pairs[group_index], operator))
            for group_index, missing_operator in enumerate(FOUR_OPERATOR_SET)
            for operator in FOUR_OPERATOR_SET
            if operator != missing_operator
        )
    if scenario == "four_set_all_ops_within_set":
        return sum(len(_filter_pairs_for_operator(pairs, operator)) for pairs in within_group_pairs for operator in operators)
    if scenario == "four_set_all_ops_all_pairs":
        all_pairs = _ordered_pairs(list(range(modulus)), list(range(modulus)))
        return sum(len(_filter_pairs_for_operator(all_pairs, operator)) for operator in operators)
    raise ValueError(f"unsupported Study 3 scenario={scenario}")


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
    if scenario in {"all_pairs_operator_complement", "all_pairs_pair_split_both_ops"} and not math.isclose(
        train_fraction,
        0.5,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{scenario} requires train_fraction=0.5")

    output_modulus = modulus if output_modulus is None else output_modulus
    groups = _build_study3_groups(
        modulus=modulus,
        scenario=scenario,
        add_set_size=add_set_size,
        interleave_chunk_size=interleave_chunk_size,
    )
    operators = _study3_operators_for_scenario(scenario)
    operator_token_ids, eq_token, seq_len = _build_study3_token_layout(
        modulus=modulus,
        use_task_token=use_task_token,
        operators=operators,
    )

    def encode_example(a: int, b: int, operator: str) -> list[int]:
        if use_task_token:
            return [a, operator_token_ids[operator], b, eq_token]
        return [a, b, eq_token]

    def make_dataset_from_pairs(pairs: list[tuple[int, int]], operator: str) -> TaskDataset:
        filtered_pairs = _filter_pairs_for_operator(pairs, operator)
        examples = [encode_example(a, b, operator) for a, b in filtered_pairs]
        targets = [apply_operator(operator, a, b, output_modulus) for a, b in filtered_pairs]
        return _build_dataset_from_examples(examples, targets)

    train_examples: list[list[int]] = []
    train_targets: list[int] = []

    def append_examples(pairs: list[tuple[int, int]], operator: str) -> None:
        for a, b in _filter_pairs_for_operator(pairs, operator):
            train_examples.append(encode_example(a, b, operator))
            train_targets.append(apply_operator(operator, a, b, output_modulus))
    
    within_group_pairs = [_ordered_pairs(group, group) for group in groups]
    cross_pairs = _cross_pairs_across_groups(groups)
    all_values = list(range(modulus))
    all_pairs = _ordered_pairs(all_values, all_values)

    if scenario in {"partitioned_ops", "interleaved_partitioned_ops"}:
        add_group_pairs = within_group_pairs[0]
        mul_group_pairs = within_group_pairs[1]
        append_examples(add_group_pairs, "add")
        append_examples(mul_group_pairs, "mul")
        final_evals = {
            "reverse_add": make_dataset_from_pairs(mul_group_pairs, "add"),
            "reverse_mul": make_dataset_from_pairs(add_group_pairs, "mul"),
            "cross_pair_add": make_dataset_from_pairs(cross_pairs, "add"),
            "cross_pair_mul": make_dataset_from_pairs(cross_pairs, "mul"),
        }
        inputs, labels, train_idx, test_idx = _make_inputs(train_examples, train_targets, train_fraction=train_fraction, seed=seed)
        train = TaskDataset(inputs[train_idx], labels[train_idx], true_targets=labels[train_idx])
        test = TaskDataset(inputs[test_idx], labels[test_idx], true_targets=labels[test_idx])
    elif scenario == "both_ops_within_set":
        for pairs in within_group_pairs:
            for operator in TWO_OPERATOR_SET:
                append_examples(pairs, operator)
        final_evals = {
            "cross_pair_add": make_dataset_from_pairs(cross_pairs, "add"),
            "cross_pair_mul": make_dataset_from_pairs(cross_pairs, "mul"),
        }
        inputs, labels, train_idx, test_idx = _make_inputs(train_examples, train_targets, train_fraction=train_fraction, seed=seed)
        train = TaskDataset(inputs[train_idx], labels[train_idx], true_targets=labels[train_idx])
        test = TaskDataset(inputs[test_idx], labels[test_idx], true_targets=labels[test_idx])
    elif scenario == "both_ops_all_pairs":
        if not use_task_token:
            raise ValueError("both_ops_all_pairs requires use_task_token=True because add and mul labels would collide")
        for operator in TWO_OPERATOR_SET:
            append_examples(all_pairs, operator)
        final_evals = {
            "cross_pair_add": make_dataset_from_pairs(cross_pairs, "add"),
            "cross_pair_mul": make_dataset_from_pairs(cross_pairs, "mul"),
        }
        inputs, labels, train_idx, test_idx = _make_inputs(train_examples, train_targets, train_fraction=train_fraction, seed=seed)
        train = TaskDataset(inputs[train_idx], labels[train_idx], true_targets=labels[train_idx])
        test = TaskDataset(inputs[test_idx], labels[test_idx], true_targets=labels[test_idx])
    elif scenario == "all_pairs_operator_complement":
        if not use_task_token:
            raise ValueError("all_pairs_operator_complement requires use_task_token=True because task identity must be explicit")
        train_pairs, test_pairs = _split_items_evenly(all_pairs, train_fraction=train_fraction, seed=seed)
        train_examples = [encode_example(a, b, "add") for a, b in train_pairs] + [encode_example(a, b, "mul") for a, b in test_pairs]
        train_targets = [apply_operator("add", a, b, output_modulus) for a, b in train_pairs] + [
            apply_operator("mul", a, b, output_modulus) for a, b in test_pairs
        ]
        test_examples = [encode_example(a, b, "mul") for a, b in train_pairs] + [encode_example(a, b, "add") for a, b in test_pairs]
        test_targets = [apply_operator("mul", a, b, output_modulus) for a, b in train_pairs] + [
            apply_operator("add", a, b, output_modulus) for a, b in test_pairs
        ]
        train = _build_dataset_from_examples(train_examples, train_targets)
        test = _build_dataset_from_examples(test_examples, test_targets)
        final_evals = {}
    elif scenario == "all_pairs_pair_split_both_ops":
        if not use_task_token:
            raise ValueError("all_pairs_pair_split_both_ops requires use_task_token=True because both operators appear for the same pair")
        train_pairs, test_pairs = _split_items_evenly(all_pairs, train_fraction=train_fraction, seed=seed)
        train_examples = []
        train_targets = []
        test_examples = []
        test_targets = []
        for operator in TWO_OPERATOR_SET:
            train_examples.extend(encode_example(a, b, operator) for a, b in _filter_pairs_for_operator(train_pairs, operator))
            train_targets.extend(apply_operator(operator, a, b, output_modulus) for a, b in _filter_pairs_for_operator(train_pairs, operator))
            test_examples.extend(encode_example(a, b, operator) for a, b in _filter_pairs_for_operator(test_pairs, operator))
            test_targets.extend(apply_operator(operator, a, b, output_modulus) for a, b in _filter_pairs_for_operator(test_pairs, operator))
        train = _build_dataset_from_examples(train_examples, train_targets)
        test = _build_dataset_from_examples(test_examples, test_targets)
        final_evals = {}
    elif scenario == "four_set_missing_ops_within_set":
        if not use_task_token:
            raise ValueError("four_set_missing_ops_within_set requires use_task_token=True because three operators are trained per set")
        withheld_by_group = list(FOUR_OPERATOR_SET)
        for group_index, pairs in enumerate(within_group_pairs):
            withheld_operator = withheld_by_group[group_index]
            for operator in FOUR_OPERATOR_SET:
                if operator == withheld_operator:
                    continue
                append_examples(pairs, operator)
        final_evals = {
            f"reverse_{operator}": make_dataset_from_pairs(within_group_pairs[group_index], operator)
            for group_index, operator in enumerate(FOUR_OPERATOR_SET)
        }
        for operator in FOUR_OPERATOR_SET:
            final_evals[f"cross_pair_{operator}"] = make_dataset_from_pairs(cross_pairs, operator)
        inputs, labels, train_idx, test_idx = _make_inputs(train_examples, train_targets, train_fraction=train_fraction, seed=seed)
        train = TaskDataset(inputs[train_idx], labels[train_idx], true_targets=labels[train_idx])
        test = TaskDataset(inputs[test_idx], labels[test_idx], true_targets=labels[test_idx])
    elif scenario == "four_set_all_ops_within_set":
        if not use_task_token:
            raise ValueError("four_set_all_ops_within_set requires use_task_token=True because four operators are trained per set")
        for pairs in within_group_pairs:
            for operator in FOUR_OPERATOR_SET:
                append_examples(pairs, operator)
        final_evals = {
            f"cross_pair_{operator}": make_dataset_from_pairs(cross_pairs, operator)
            for operator in FOUR_OPERATOR_SET
        }
        inputs, labels, train_idx, test_idx = _make_inputs(train_examples, train_targets, train_fraction=train_fraction, seed=seed)
        train = TaskDataset(inputs[train_idx], labels[train_idx], true_targets=labels[train_idx])
        test = TaskDataset(inputs[test_idx], labels[test_idx], true_targets=labels[test_idx])
    elif scenario == "four_set_all_ops_all_pairs":
        if not use_task_token:
            raise ValueError("four_set_all_ops_all_pairs requires use_task_token=True because four operators are trained over all pairs")
        for operator in FOUR_OPERATOR_SET:
            append_examples(all_pairs, operator)
        final_evals = {
            f"cross_pair_{operator}": make_dataset_from_pairs(cross_pairs, operator)
            for operator in FOUR_OPERATOR_SET
        }
        inputs, labels, train_idx, test_idx = _make_inputs(train_examples, train_targets, train_fraction=train_fraction, seed=seed)
        train = TaskDataset(inputs[train_idx], labels[train_idx], true_targets=labels[train_idx])
        test = TaskDataset(inputs[test_idx], labels[test_idx], true_targets=labels[test_idx])
    else:
        raise ValueError(f"unsupported Study 3 scenario={scenario}")

    info = DatasetInfo(
        vocab_size=eq_token + 1,
        target_vocab_size=output_modulus,
        seq_len=seq_len,
        eq_token_id=eq_token,
        operator_token_ids=operator_token_ids,
    )
    return Study3Split(train=train, test=test, final_evals=final_evals, info=info)
