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
    activation: str = "relu"
    norm_style: str = "none"
    positional_embedding_type: str = "learned"
    dropout: float = 0.0
    mlp_bias: bool = True
    final_norm: bool = False

    def __post_init__(self) -> None:
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.n_heads * self.d_head != self.d_model:
            raise ValueError("n_heads * d_head must equal d_model")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError("activation must be one of: relu, gelu")
        if self.norm_style not in {"none", "pre", "post"}:
            raise ValueError("norm_style must be one of: none, pre, post")
        if self.positional_embedding_type not in {"learned", "sinusoidal"}:
            raise ValueError("positional_embedding_type must be one of: learned, sinusoidal")

    @classmethod
    def neel_nanda(cls, *, vocab_size: int, seq_len: int) -> "TransformerConfig":
        return cls(
            vocab_size=vocab_size,
            seq_len=seq_len,
            d_model=128,
            n_heads=4,
            d_head=32,
            d_mlp=512,
            n_layers=1,
            activation="relu",
            norm_style="none",
            positional_embedding_type="learned",
            dropout=0.0,
            mlp_bias=True,
            final_norm=False,
        )

    @classmethod
    def power_grokking(cls, *, vocab_size: int, seq_len: int) -> "TransformerConfig":
        return cls(
            vocab_size=vocab_size,
            seq_len=seq_len,
            d_model=128,
            n_heads=4,
            d_head=32,
            d_mlp=512,
            n_layers=2,
            activation="relu",
            norm_style="post",
            positional_embedding_type="sinusoidal",
            dropout=0.0,
            mlp_bias=False,
            final_norm=False,
        )


def _activation_layer(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"unsupported activation={name}")


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
        self.norm_style = config.norm_style
        self.ln1 = nn.LayerNorm(config.d_model) if config.norm_style != "none" else nn.Identity()
        self.attn = MultiHeadSelfAttention(config)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
        self.ln2 = nn.LayerNorm(config.d_model) if config.norm_style != "none" else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_mlp, bias=config.mlp_bias),
            _activation_layer(config.activation),
            nn.Linear(config.d_mlp, config.d_model, bias=config.mlp_bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_style == "pre":
            x = x + self.dropout(self.attn(self.ln1(x)))
            x = x + self.dropout(self.mlp(self.ln2(x)))
            return x
        if self.norm_style == "post":
            x = self.ln1(x + self.dropout(self.attn(x)))
            x = self.ln2(x + self.dropout(self.mlp(x)))
            return x
        x = x + self.dropout(self.attn(x))
        x = x + self.dropout(self.mlp(x))
        return x


class GrokkingTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        if config.positional_embedding_type == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(config.seq_len, config.d_model))
        else:
            self.register_buffer("pos_embed", self._sinusoidal_position_encoding(config.seq_len, config.d_model), persistent=False)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_ln = nn.LayerNorm(config.d_model) if config.final_norm else nn.Identity()
        self.unembed = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        if isinstance(self.pos_embed, nn.Parameter):
            nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.unembed.weight, mean=0.0, std=0.02)

    @staticmethod
    def _sinusoidal_position_encoding(seq_len: int, d_model: int) -> torch.Tensor:
        positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pos_embed = torch.zeros(seq_len, d_model, dtype=torch.float32)
        pos_embed[:, 0::2] = torch.sin(positions * div_term)
        pos_embed[:, 1::2] = torch.cos(positions * div_term)
        return pos_embed

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 2:
            raise ValueError("tokens must have shape [batch, seq_len]")
        if tokens.size(1) != self.config.seq_len:
            raise ValueError(f"expected seq_len={self.config.seq_len}, got {tokens.size(1)}")

        x = self.token_embed(tokens) + self.pos_embed.unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        return self.unembed(x[:, -1, :])
