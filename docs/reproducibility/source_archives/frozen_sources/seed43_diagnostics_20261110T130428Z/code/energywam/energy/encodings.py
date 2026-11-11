"""Explicit, semantically-named encodings for the appended candidate tokens.

The research plan appends continuous candidates to the Qwen sequence.  Qwen's
native mRoPE coordinates are used for the *native* multimodal sequence only; the
appended block gets a fresh, strictly-after position range (see the backbone
adapter) plus these **independent** encodings which carry the actual semantics:

* future time    — which of the three future offsets this patch group belongs to
* future spatial — patch index inside the DINO grid (row-major)
* camera         — which view the tokens come from
* token type     — condition / future / action / energy
* action step    — index inside the action chunk

Everything is additive in the *backbone hidden space*, i.e. applied after the
projectors, so the encodings do not interfere with the raw candidate values that
receive gradients.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


TOKEN_TYPE = {
    "cond_vl": 0,
    "dino_cur": 1,
    "state": 2,
    "future": 3,
    "action": 4,
    "energy": 5,
    "placeholder": 6,
    "pad": 7,
}


class SinusoidalEncoding(nn.Module):
    """Fixed sin/cos encoding of an integer index into ``dim`` features."""

    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, index: torch.Tensor) -> torch.Tensor:
        """index: integer tensor of any shape -> same shape + (dim,)"""
        if index.numel() == 0:
            return index.new_zeros(*index.shape, self.dim, dtype=self.pe.dtype)
        idx = index.long().clamp(0, self.pe.shape[0] - 1)
        return self.pe[idx.to(self.pe.device)].to(dtype=torch.float32)


class CandidateEncoding(nn.Module):
    """All additive encodings used by the EnergyWAM token builders."""

    def __init__(self, hidden_dim: int, cfg_geo, *, positional_scale=1.0, type_embedding_std=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.positional_scale = float(positional_scale)
        self.future_time = SinusoidalEncoding(hidden_dim, max_len=64)
        self.future_spatial = SinusoidalEncoding(hidden_dim, max_len=1024)
        self.camera = SinusoidalEncoding(hidden_dim, max_len=32)
        self.action_step = SinusoidalEncoding(hidden_dim, max_len=512)
        self.type_embed = nn.Embedding(len(TOKEN_TYPE), hidden_dim)
        if type_embedding_std != 1.0:
            with torch.no_grad():
                self.type_embed.weight.mul_(float(type_embedding_std))

    def type_of(self, kind: str) -> torch.Tensor:
        return self.type_embed.weight[TOKEN_TYPE[kind]]

    # -- future ------------------------------------------------------------
    def future(
        self,
        batch: int,
        num_frames: int,
        num_cameras: int,
        num_patches: int,
        device: torch.device,
        dtype: torch.dtype,
        offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return additive encoding [B, F*V*P, D] laid out frame-major then camera then patch.

        Layout matches ``Z`` reshaped as ``[B, F*V*P, D]``.
        """
        N = num_frames * num_cameras * num_patches
        out = torch.zeros(batch, N, self.hidden_dim, device=device, dtype=torch.float32)
        t_idx = []
        c_idx = []
        p_idx = []
        for f in range(num_frames):
            for v in range(num_cameras):
                for p in range(num_patches):
                    t_idx.append(f)
                    c_idx.append(v)
                    p_idx.append(p)
        t = torch.tensor(t_idx, device=device, dtype=torch.long)
        c = torch.tensor(c_idx, device=device, dtype=torch.long)
        p = torch.tensor(p_idx, device=device, dtype=torch.long)
        # future time: use the *actual offset* when available (explicit semantics)
        if offsets is not None:
            if not isinstance(offsets, torch.Tensor):
                offsets = torch.as_tensor(offsets, device=device, dtype=torch.long)
            if offsets.dim() == 1:
                offsets = offsets.unsqueeze(0).expand(batch, -1)
            # offsets: [B, F] -> [B, N]
            t_b = offsets[:, t]  # [B, N]
            enc_t = self.future_time(t_b)  # [B, N, D]
        else:
            enc_t = self.future_time(t).unsqueeze(0).expand(batch, -1, -1)
        enc_c = self.camera(c).unsqueeze(0).expand(batch, -1, -1)
        enc_p = self.future_spatial(p).unsqueeze(0).expand(batch, -1, -1)
        out = ((enc_t.to(dtype) + enc_c.to(dtype) + enc_p.to(dtype)) * self.positional_scale
               + self.type_of("future").to(device=device, dtype=dtype))
        return out

    # -- action ------------------------------------------------------------
    def action(self, batch: int, horizon: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        step = torch.arange(horizon, device=device, dtype=torch.long)
        enc = self.action_step(step).unsqueeze(0).expand(batch, -1, -1)
        out = enc.to(dtype) * self.positional_scale + self.type_of("action").to(device=device, dtype=dtype)
        return out

    # -- energy ------------------------------------------------------------
    def energy(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.type_of("energy").to(device=device, dtype=dtype).view(1, 1, -1).expand(batch, 1, -1)

    # -- condition extras --------------------------------------------------
    def dino_current(
        self, batch: int, num_cameras: int, num_patches: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        rows = []
        for v in range(num_cameras):
            for p in range(num_patches):
                rows.append((v, p))
        c = torch.tensor([r[0] for r in rows], device=device, dtype=torch.long)
        p = torch.tensor([r[1] for r in rows], device=device, dtype=torch.long)
        out = (
            (self.camera(c).to(dtype)
            + self.future_spatial(p).to(dtype)) * self.positional_scale
            + self.type_of("dino_cur").to(device=device, dtype=dtype)
        )
        return out.unsqueeze(0).expand(batch, -1, -1)

    def state(self, batch: int, num_state_tokens: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        out = self.type_of("state").to(device=device, dtype=dtype).view(1, 1, -1)
        return out.expand(batch, num_state_tokens, -1)

    def placeholder_future(self, batch: int, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        out = self.type_of("placeholder").to(device=device, dtype=dtype).view(1, 1, -1)
        return out.expand(batch, length, -1)
