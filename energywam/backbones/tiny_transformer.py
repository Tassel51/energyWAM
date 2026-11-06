"""Tiny causal energy backbone — **explicit tiny test mode only**.

This is a small hand-written Transformer so the solver / losses / stage logic can
be exercised on CPU deterministically.  Attention is implemented with plain
``matmul`` + ``softmax`` (no fused kernel) so double backward is guaranteed.

It is NOT a stand-in for the real Qwen3-VL path: the condition tokens come from a
deterministic mock encoder, and the whole module is only constructed when
``backbone.backend == 'tiny'``.  Real mode raises a clear error if the required
weights are missing (see :mod:`energywam.backbones.qwen3vl_adapter`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from ..energy.types import ConditionBundle, TokenGroup


def _causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Additive mask [1, 1, S, S]: 0 for allowed, -inf for blocked."""
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, seq_len, seq_len)


class TinySelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # [B, S, H, hd]
        q = q.transpose(1, 2)  # [B, H, S, hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + _causal_mask(S, x.device, scores.dtype)
        if key_padding_mask is not None:
            # padding suppresses *keys* only
            km = key_padding_mask.to(dtype=torch.bool)  # [B, S] True = valid
            km = km[:, None, None, :]  # [B, 1, 1, S]
            scores = scores.masked_fill(~km, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        # rows that are fully masked (queries whose keys are all invalid) would
        # become NaN; replace those rows with zeros — we never read them.
        attn = torch.nan_to_num(attn, nan=0.0)
        out = attn @ v  # [B, H, S, hd]
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.proj(out)


class TinyBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = TinySelfAttention(dim, num_heads)
        self.n2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.n1(x), key_padding_mask)
        x = x + self.mlp(self.n2(x))
        return x


class TinyEnergyBackbone(nn.Module):
    """Causal Transformer over ``[C, T_future, T_action, E]``."""

    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_dim
        self.blocks = nn.ModuleList([TinyBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = tokens
        for blk in self.blocks:
            x = blk(x, key_padding_mask)
        return self.final_norm(x)


class TinyConditionEncoder(nn.Module):
    """Deterministic mock condition encoder (tiny mode only).

    Takes a raw condition vector per sample and produces ``N_c`` condition tokens
    through a fixed, seeded linear map + a learned projection.  Deterministic
    given the input; no sampling.
    """

    def __init__(self, hidden_dim: int, cond_dim: int, num_tokens: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_dim
        self.proj = nn.Linear(cond_dim, num_tokens * hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, cond_vec: torch.Tensor) -> torch.Tensor:
        """cond_vec: [B, cond_dim] -> [B, num_tokens, hidden_dim]"""
        B = cond_vec.shape[0]
        out = self.proj(cond_vec).view(B, self.num_tokens, self.hidden_size)
        return self.norm(out)


@dataclass
class TinyBundle:
    cond_tokens: torch.Tensor  # [B, Nc, D]
    cond_mask: torch.Tensor  # [B, Nc]
