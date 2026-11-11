"""Tiny-mode backbone adapter (explicit tiny test mode only).

Implements the two-method backbone interface used by
:class:`energywam.models.energy_core.EnergyWAMCore`:

``encode_condition(batch) -> ConditionBundle``
``run_sequence(cond, appended_groups) -> (hidden [B,S,D], energy_index [B])``

Real mode is never routed here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..config import EnergyWAMConfig
from ..energy.types import ConditionBundle, TokenGroup
from .tiny_transformer import TinyConditionEncoder, TinyEnergyBackbone


class TinyBackboneAdapter(nn.Module):
    def __init__(self, cfg: EnergyWAMConfig, hidden_dim: int):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_dim
        b = cfg.backbone
        self.cond_encoder = TinyConditionEncoder(hidden_dim, b.tiny_state_dim + b.tiny_action_dim, b.tiny_num_cond_tokens)
        self.transformer = TinyEnergyBackbone(hidden_dim, b.tiny_num_layers, b.tiny_num_heads)
        self.num_cond_tokens = b.tiny_num_cond_tokens

    # ------------------------------------------------------------------
    def encode_condition(self, batch: Dict[str, torch.Tensor]) -> ConditionBundle:
        """batch must contain ``cond_vec``: [B, cond_dim] (deterministic mock)."""
        if "cond_vec" not in batch:
            raise ValueError("tiny backbone requires batch['cond_vec'] of shape [B, cond_dim]")
        cond_vec = batch["cond_vec"]
        tokens = self.cond_encoder(cond_vec)
        mask = torch.ones(tokens.shape[0], tokens.shape[1], device=tokens.device, dtype=torch.bool)
        group = TokenGroup(embeds=tokens, mask=mask, kind="cond_mock")
        return ConditionBundle(group=group)

    # ------------------------------------------------------------------
    def run_sequence(
        self, cond: ConditionBundle, appended: Sequence[TokenGroup]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cond.group is None:
            raise ValueError("tiny backbone requires cond.group")
        groups = [cond.group, *appended]
        embeds = torch.cat([g.embeds for g in groups], dim=1)
        mask = torch.cat([g.mask for g in groups], dim=1)
        hidden = self.transformer(embeds, mask)
        # energy token is the LAST appended group, position 0 within it
        lengths = [g.length for g in groups]
        e_index = torch.tensor(
            [sum(lengths[:-1]) + lengths[-1] - 1] * embeds.shape[0],
            device=embeds.device,
            dtype=torch.long,
        )
        return hidden, e_index

    # ------------------------------------------------------------------
    def encode_condition_raw(self, cond_vec: torch.Tensor) -> ConditionBundle:
        return self.encode_condition({"cond_vec": cond_vec})
