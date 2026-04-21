from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MLPConfig:
    prime: int
    hidden_dim: int = 512

    @property
    def input_dim(self) -> int:
        return 2 * self.prime

    @property
    def output_dim(self) -> int:
        return self.prime


class ModularMLP(nn.Module):
    """Two-layer MLP for modular arithmetic, matching the paper's U/V/W form."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim, bias=False)
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.input_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 2 or tokens.size(1) < 2:
            raise ValueError("tokens must have shape [batch, seq_len] with at least two input positions")

        a = tokens[:, 0]
        b = tokens[:, 1]
        a_one_hot = F.one_hot(a, num_classes=self.config.prime).float()
        b_one_hot = F.one_hot(b, num_classes=self.config.prime).float()
        inputs = torch.cat([a_one_hot, b_one_hot], dim=-1)

        hidden = self.input_proj(inputs)
        hidden = hidden.square()
        return self.output_proj(hidden)
