from __future__ import annotations

import torch


def _partial_absolute_reward(
    actions: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_vocab_size: int,
) -> torch.Tensor:
    distance = (actions - targets).abs().float()
    max_difference = max(target_vocab_size - 1, 1)
    return (1.0 - distance / max_difference).clamp(min=0.0, max=1.0)


def compute_reward(
    actions: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_vocab_size: int,
    reward_mode: str,
) -> torch.Tensor:
    actions = actions.long()
    targets = targets.long()

    if reward_mode == "binary":
        return (actions == targets).float()

    if reward_mode == "partial_absolute":
        return _partial_absolute_reward(actions, targets, target_vocab_size=target_vocab_size)

    if reward_mode == "binary_plus_partial_absolute":
        partial_reward = _partial_absolute_reward(actions, targets, target_vocab_size=target_vocab_size)
        return torch.where(actions == targets, torch.ones_like(partial_reward), partial_reward)

    if reward_mode == "partial_circular":
        raw_distance = (actions - targets).abs()
        circular_distance = torch.minimum(raw_distance, target_vocab_size - raw_distance).float()
        denominator = max(target_vocab_size // 2, 1)
        return (1.0 - circular_distance / denominator).clamp(min=0.0, max=1.0)

    raise ValueError(f"unsupported reward_mode={reward_mode}")
