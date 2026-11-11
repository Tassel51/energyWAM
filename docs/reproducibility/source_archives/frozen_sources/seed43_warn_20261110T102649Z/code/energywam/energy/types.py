"""Shared tensor containers for the EnergyWAM core.

Everything here is pure data-holding; no heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class TokenGroup:
    """A contiguous block of tokens appended to (or forming) the energy sequence.

    Attributes
    ----------
    embeds : [B, N, D] float tensor
    mask   : [B, N] bool — True where the token is a real (non-padding) token.
             Padding may only suppress *keys*; query rows for padded positions
             are allowed to produce garbage since we never read them.
    kind   : short label used by the backbone for position bookkeeping.
    """

    embeds: torch.Tensor
    mask: torch.Tensor
    kind: str

    def __post_init__(self) -> None:
        if self.embeds.dim() != 3:
            raise ValueError(f"embeds must be [B,N,D], got {tuple(self.embeds.shape)}")
        if self.mask.shape != self.embeds.shape[:2]:
            raise ValueError(f"mask {tuple(self.mask.shape)} != embeds {tuple(self.embeds.shape[:2])}")

    @property
    def batch(self) -> int:
        return self.embeds.shape[0]

    @property
    def length(self) -> int:
        return self.embeds.shape[1]


@dataclass
class ConditionBundle:
    """The condition block ``C``.

    For the real Qwen path, ``native`` carries the processor outputs required to
    run the *original* image token replacement / DeepStack / mRoPE path; the
    ``group`` holds the additional current-DINO and robot-state tokens that are
    appended after the native vision-language sequence.

    ``native['inputs_embeds']`` is the result of Qwen ViT/visual merger and
    text embedding lookup, BEFORE any Qwen language-transformer layer.  It is
    reused across energy evaluations; each complete candidate sequence runs
    through the language transformer once in ``backbone.run_sequence``.
    """

    group: Optional[TokenGroup]
    native: Dict[str, torch.Tensor] = field(default_factory=dict)
    #: length of the native (pre-append) sequence per sample, [B]
    native_lengths: Optional[torch.Tensor] = None
    #: native mRoPE position ids, [3, B, S_native]
    native_position_ids: Optional[torch.Tensor] = None
    #: native attention mask, [B, S_native]
    native_attention_mask: Optional[torch.Tensor] = None
    #: deepstack visual embeds (list of [num_visual_tokens, D]) — Qwen only
    deepstack_visual_embeds: Optional[List[torch.Tensor]] = None
    visual_pos_masks: Optional[torch.Tensor] = None


@dataclass
class CandidateMeta:
    """Geometry describing the candidate token layout (for encodings & readout)."""

    action_horizon: int
    action_dim: int
    num_future_frames: int
    num_cameras: int
    future_patches: int
    future_dim: int
    #: per-sample real offsets actually used, [B, F]
    future_offsets: Optional[torch.Tensor] = None


@dataclass
class EnergyReadout:
    energy: torch.Tensor  # [B]
    energy_token_index: torch.Tensor  # [B] long
    hidden: Optional[torch.Tensor] = None  # [B, D]
    diagnostics: Dict[str, torch.Tensor] = field(default_factory=dict)
