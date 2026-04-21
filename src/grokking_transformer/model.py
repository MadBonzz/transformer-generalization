from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    seq_len: int = 3
    d_model: int = 128
    n_heads: int = 4
    d_head: int = 32
    d_mlp: int = 512
    n_layers: int = 1

    def __post_init__(self) -> None:
        if self.n_layers != 1:
            raise ValueError("This project implements the 1-layer setting only.")
        if self.n_heads * self.d_head != self.d_model:
            raise ValueError("n_heads * d_head must equal d_model")


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.out_proj = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)

        mask = torch.tril(torch.ones(config.seq_len, config.seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.seq_len, config.seq_len), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_scores = attn_scores.masked_fill(~self.causal_mask[:, :, :seq_len, :seq_len], float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.d_head)
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_mlp),
            nn.GELU(),
            nn.Linear(config.d_mlp, config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GrokkingTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(config.seq_len, config.d_model))
        self.block = TransformerBlock(config)
        self.final_ln = nn.LayerNorm(config.d_model)
        self.unembed = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.unembed.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 2:
            raise ValueError("tokens must have shape [batch, seq_len]")
        if tokens.size(1) != self.config.seq_len:
            raise ValueError(f"expected seq_len={self.config.seq_len}, got {tokens.size(1)}")

        x = self.token_embed(tokens) + self.pos_embed.unsqueeze(0)
        x = self.block(x)
        x = self.final_ln(x)
        return self.unembed(x[:, -1, :])
