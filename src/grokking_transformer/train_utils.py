from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


@dataclass
class EvalMetrics:
    loss: float
    accuracy: float


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    target_vocab_size: int,
    loss_type: str = "cross_entropy",
) -> EvalMetrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    with torch.no_grad():
        for tokens, targets in dataloader:
            tokens = tokens.to(device)
            targets = targets.to(device)
            logits, loss = forward_and_loss(model, tokens, targets, target_vocab_size, loss_type)
            total_loss += loss.item() * tokens.size(0)
            total_correct += (logits.argmax(dim=-1) == targets).sum().item()
            total_items += tokens.size(0)

    return EvalMetrics(
        loss=total_loss / max(total_items, 1),
        accuracy=total_correct / max(total_items, 1),
    )


def train_step(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_vocab_size: int,
    loss_type: str = "cross_entropy",
) -> float:
    model.train()
    tokens, targets = batch
    tokens = tokens.to(device)
    targets = targets.to(device)

    optimizer.zero_grad(set_to_none=True)
    _, loss = forward_and_loss(model, tokens, targets, target_vocab_size, loss_type)
    loss.backward()
    optimizer.step()
    return loss.item()


def forward_and_loss(
    model: nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    target_vocab_size: int,
    loss_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(tokens)[:, :target_vocab_size]

    if loss_type == "cross_entropy":
        loss = F.cross_entropy(logits, targets)
    elif loss_type == "mse_one_hot":
        target_vectors = F.one_hot(targets, num_classes=target_vocab_size).float()
        loss = F.mse_loss(logits, target_vectors)
    elif loss_type == "mae_one_hot":
        target_vectors = F.one_hot(targets, num_classes=target_vocab_size).float()
        loss = F.l1_loss(logits, target_vectors)
    else:
        raise ValueError(f"unsupported loss_type={loss_type}")

    return logits, loss
