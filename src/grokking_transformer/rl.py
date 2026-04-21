from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from .rewards import compute_reward


@dataclass(frozen=True)
class GRPOConfig:
    n_samples: int = 8
    clip_eps: float = 0.2
    policy_epochs: int = 4
    entropy_coef: float = 1e-3
    kl_coef: float = 0.0
    temperature: float = 1.0
    max_grad_norm: float = 1.0
    reward_mode: str = "binary"

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class PPOConfig:
    n_samples: int = 8
    clip_eps: float = 0.2
    policy_epochs: int = 4
    entropy_coef: float = 1e-3
    value_coef: float = 0.5
    temperature: float = 1.0
    max_grad_norm: float = 1.0
    reward_mode: str = "binary"

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def create_reference_model(model: nn.Module) -> nn.Module:
    reference_model = copy.deepcopy(model)
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    return reference_model


def _group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, correction=0)
    return centered / (std + 1e-8)


def grpo_update(
    *,
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_vocab_size: int,
    config: GRPOConfig,
    reference_model: nn.Module | None = None,
) -> dict[str, float]:
    model.train()
    tokens, targets = batch
    tokens = tokens.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        rollout_logits = model(tokens)[:, :target_vocab_size] / config.temperature
        rollout_dist = Categorical(logits=rollout_logits)
        sampled_actions = rollout_dist.sample((config.n_samples,))
        old_log_probs = rollout_dist.log_prob(sampled_actions)
        reward_targets = targets.unsqueeze(0).expand_as(sampled_actions)
        rewards = compute_reward(
            sampled_actions,
            reward_targets,
            target_vocab_size=target_vocab_size,
            reward_mode=config.reward_mode,
        )
        sampled_actions = sampled_actions.transpose(0, 1)
        old_log_probs = old_log_probs.transpose(0, 1)
        rewards = rewards.transpose(0, 1)
        advantages = _group_advantages(rewards)
        ref_logits = None
        if reference_model is not None:
            ref_logits = reference_model(tokens)[:, :target_vocab_size] / config.temperature

    last_loss = 0.0
    last_entropy = 0.0
    last_kl = 0.0
    clip_fraction = 0.0

    for _ in range(config.policy_epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)[:, :target_vocab_size] / config.temperature
        dist = Categorical(logits=logits)
        current_log_probs = dist.log_prob(sampled_actions.transpose(0, 1)).transpose(0, 1)
        ratios = (current_log_probs - old_log_probs).exp()
        clipped_ratios = ratios.clamp(1.0 - config.clip_eps, 1.0 + config.clip_eps)
        surrogate = torch.minimum(ratios * advantages, clipped_ratios * advantages)
        policy_loss = -surrogate.mean()
        entropy = dist.entropy().mean()

        if ref_logits is not None and config.kl_coef > 0.0:
            with torch.no_grad():
                ref_probs = torch.softmax(ref_logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            kl = (ref_probs * (torch.log(ref_probs + 1e-8) - log_probs)).sum(dim=-1).mean()
        else:
            kl = torch.tensor(0.0, device=device)

        loss = policy_loss - config.entropy_coef * entropy + config.kl_coef * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        last_loss = float(loss.item())
        last_entropy = float(entropy.item())
        last_kl = float(kl.item())
        clip_fraction = float(((ratios < 1.0 - config.clip_eps) | (ratios > 1.0 + config.clip_eps)).float().mean().item())

    return {
        "loss": last_loss,
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(rewards.std(correction=0).item()),
        "entropy": last_entropy,
        "kl": last_kl,
        "clip_fraction": clip_fraction,
    }


def ppo_update(
    *,
    model: nn.Module,
    value_head: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_vocab_size: int,
    config: PPOConfig,
) -> dict[str, float]:
    model.train()
    value_head.train()
    tokens, targets = batch
    tokens = tokens.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        rollout_logits = model(tokens)[:, :target_vocab_size] / config.temperature
        rollout_dist = Categorical(logits=rollout_logits)
        sampled_actions = rollout_dist.sample((config.n_samples,))
        old_log_probs = rollout_dist.log_prob(sampled_actions)
        reward_targets = targets.unsqueeze(0).expand_as(sampled_actions)
        rewards = compute_reward(
            sampled_actions,
            reward_targets,
            target_vocab_size=target_vocab_size,
            reward_mode=config.reward_mode,
        )
        sampled_actions = sampled_actions.transpose(0, 1)
        old_log_probs = old_log_probs.transpose(0, 1)
        rewards = rewards.transpose(0, 1)

    last_loss = 0.0
    last_entropy = 0.0
    last_value_loss = 0.0
    clip_fraction = 0.0

    for _ in range(config.policy_epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)[:, :target_vocab_size] / config.temperature
        dist = Categorical(logits=logits)
        current_log_probs = dist.log_prob(sampled_actions.transpose(0, 1)).transpose(0, 1)
        state_values = value_head(logits.detach()).squeeze(-1)
        expanded_values = state_values.unsqueeze(1).expand_as(rewards)
        advantages = rewards - expanded_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std(correction=0) + 1e-8)
        ratios = (current_log_probs - old_log_probs).exp()
        clipped_ratios = ratios.clamp(1.0 - config.clip_eps, 1.0 + config.clip_eps)
        policy_loss = -torch.minimum(ratios * advantages, clipped_ratios * advantages).mean()
        value_loss = torch.nn.functional.mse_loss(expanded_values, rewards)
        entropy = dist.entropy().mean()
        loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(value_head.parameters()), config.max_grad_norm)
        optimizer.step()

        last_loss = float(loss.item())
        last_entropy = float(entropy.item())
        last_value_loss = float(value_loss.item())
        clip_fraction = float(((ratios < 1.0 - config.clip_eps) | (ratios > 1.0 + config.clip_eps)).float().mean().item())

    return {
        "loss": last_loss,
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(rewards.std(correction=0).item()),
        "entropy": last_entropy,
        "value_loss": last_value_loss,
        "clip_fraction": clip_fraction,
    }
