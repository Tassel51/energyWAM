"""EnergyWAM core: candidate encoding + scalar energy forward.

The module owns

* the candidate projectors (future / action / state / current-DINO),
* the additive semantic encodings,
* the learnable Energy token,
* the scalar energy head,

and delegates the actual sequence modelling to a *backbone* object exposing

``encode_condition(batch) -> ConditionBundle``
``run_sequence(cond, appended_groups) -> (hidden [B,S,D], energy_index [B])``

This keeps tiny and real backbones on exactly the same solver / loss / stage
code path.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import EnergyWAMConfig
from ..energy.encodings import CandidateEncoding
from ..energy.iteration import IterationConditioning, BoundedStepSizes
from ..energy.patch_packing import pack_patches
from ..energy.losses import broadcast_bool_mask
from ..energy.types import ConditionBundle, EnergyReadout, TokenGroup



# ---------------------------------------------------------------------------
# mask helpers
# ---------------------------------------------------------------------------
def _to_bool_mask(mask, ref: torch.Tensor, *, value_rank: int, batch: int) -> torch.Tensor:
    """Broadcast a caller mask onto ``ref``'s value shape and return bool.

    Accepted forms (see tests/test_mask_regression.py):

    * the exact value shape ``[..., D]``;
    * reduced batch-first shapes (``[B, F]``, ``[B, F, V]``, ``[B, F, V, P]``…);
    * batch-shared reduced shapes (``[F, V, P]`` …);
    * standard right-aligned broadcast shapes (``[1, 1, ...]``).
    """
    if mask is None:
        return torch.ones_like(ref, dtype=torch.bool)
    m = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask)
    m = m.to(device=ref.device)
    if m.dtype != torch.bool:
        m = m != 0
    if m.shape == ref.shape:
        return m
    # 1) left-align to ref rank
    cands = []
    left = m
    while left.dim() < ref.dim():
        left = left.unsqueeze(-1)
    cands.append(left)
    # 2) batch-shared: prepend a batch dim
    if m.dim() < ref.dim():
        shared = m.unsqueeze(0)
        while shared.dim() < ref.dim():
            shared = shared.unsqueeze(-1)
        cands.append(shared)
    # 3) raw torch right-aligned broadcast
    cands.append(m)
    for cand in cands:
        try:
            return torch.broadcast_to(cand, ref.shape).contiguous()
        except RuntimeError:
            continue
    raise ValueError(
        f"mask shape {tuple(mask.shape)} not broadcastable to value shape {tuple(ref.shape)}"
    )


def _apply_value_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero masked entries *before* any projection.

    ``torch.where`` is used rather than multiplication so a NaN/Inf sitting in a
    masked-out slot cannot leak through the projector.
    """
    return torch.where(mask, x, torch.zeros((), dtype=x.dtype, device=x.device))


def _token_mask(mask: torch.Tensor, *, trailing: int) -> torch.Tensor:
    """Collapse a value mask to one flag per token: valid iff ANY channel is."""
    m = mask
    for _ in range(trailing):
        m = m.any(dim=-1)
    return m


class EnergyHead(nn.Module):
    """MLP mapping the Energy token hidden state to a per-sample scalar."""

    def __init__(self, hidden_dim: int, head_hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, D] -> [B]

        The readout may arrive from a reduced-precision backbone (e.g. BF16)
        while this head stays FP32: ``h`` is converted to the head's parameter
        dtype before the MLP runs.  The head is never force-cast to the
        backbone dtype — the whole core does not need to become BF16.
        """
        p = next(self.parameters())
        if h.dtype != p.dtype:
            h = h.to(dtype=p.dtype)
        return self.net(h).squeeze(-1)


class EnergyWAMCore(nn.Module):
    """Condition/candidate encoding and scalar energy forward."""

    def __init__(self, cfg: EnergyWAMConfig, backbone: nn.Module, geometry: Dict[str, int]):
        """geometry: keys ``action_dim``, ``state_dim``, ``future_dim``,
        ``future_patches``, ``num_cameras``, ``action_horizon``,
        ``num_future_frames``, ``dino_dim``."""
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.hidden_size = int(backbone.hidden_size)
        H = self.hidden_size

        if cfg.framework.quadratic_candidate_prior:
            self.register_buffer("prior_action_precision", torch.tensor(1.0/cfg.solver.alpha_action, dtype=torch.float32))
            self.register_buffer("prior_future_precision", torch.tensor(1.0/cfg.solver.alpha_future, dtype=torch.float32))
        self.geometry = dict(geometry)
        self.action_dim = int(geometry["action_dim"])
        self.state_dim = int(geometry["state_dim"])
        self.future_dim = int(geometry["future_dim"])
        self.future_patches = int(geometry["future_patches"])
        self.num_cameras = int(geometry["num_cameras"])
        self.action_horizon = int(geometry["action_horizon"])
        self.num_future_frames = int(geometry["num_future_frames"])
        self.dino_dim = int(geometry.get("dino_dim", self.future_dim))
        self.num_state_tokens = int(cfg.framework.num_state_tokens)
        self.patch_pack_factor = cfg.framework.dino_patch_pack_factor
        # Validate geometry before allocating larger projector parameters.
        if self.patch_pack_factor > 1:
            import math
            side = math.isqrt(self.future_patches)
            if side * side != self.future_patches or side % self.patch_pack_factor:
                raise ValueError('patch packing requires a square, divisible DINO grid')
        packed_channels = self.patch_pack_factor ** 2
        self.future_token_patches = self.future_patches // packed_channels

        # --- projectors ---
        self.future_projector = nn.Sequential(
            nn.Linear(self.future_dim, H),
            nn.GELU(),
            nn.Linear(H, H),
        )
        self.action_projector = nn.Sequential(
            nn.Linear(self.action_dim, H),
            nn.GELU(),
            nn.Linear(H, H),
        )
        self.state_projector = nn.Sequential(
            nn.Linear(self.state_dim, H),
            nn.GELU(),
            nn.Linear(H, self.num_state_tokens * H),
        )
        self.current_dino_projector = nn.Sequential(
            nn.Linear(self.dino_dim, H),
            nn.GELU(),
            nn.Linear(H, H),
        )

        self.encodings = CandidateEncoding(H, self.geometry,
            positional_scale=cfg.framework.positional_encoding_scale,
            type_embedding_std=cfg.framework.type_embedding_init_std)

        # --- learnable Energy readout token ---
        self.energy_token = nn.Parameter(torch.zeros(1, 1, H))
        nn.init.normal_(self.energy_token, std=0.02)

        # learned placeholder for an absent future modality (action_only mode)
        self.future_placeholder = nn.Parameter(torch.zeros(1, 1, H))
        nn.init.normal_(self.future_placeholder, std=0.02)

        self.energy_head = EnergyHead(H, cfg.framework.energy_head_hidden, cfg.framework.energy_head_dropout)
        if cfg.framework.energy_head_zero_init:
            nn.init.zeros_(self.energy_head.net[-1].weight)
            nn.init.zeros_(self.energy_head.net[-1].bias)
        # Optional-module initialization must not change common weights or later
        # candidate RNG when comparing an iteration-conditioned ablation.
        with torch.random.fork_rng(devices=[]):
            self.iteration_conditioner = (IterationConditioning(
                H, cfg.framework.iteration_embedding_dim, cfg.framework.iteration_mlp_hidden,
                cfg.framework.iteration_progress_scale) if cfg.framework.iteration_conditioning else None)
        self.step_sizes = (BoundedStepSizes(
            cfg.solver.alpha_action, cfg.solver.alpha_future,
            cfg.solver.step_size_min_ratio, cfg.solver.step_size_max_ratio)
            if cfg.solver.learnable_step_sizes else None)
        if packed_channels > 1:
            # Keep every unchanged-shape parameter and the subsequent RNG
            # stream identical to factor=1. Only the two input weight shapes
            # differ, so a tokenization ablation does not also randomize the
            # action/state/head/iteration modules.
            with torch.random.fork_rng(devices=[]):
                for projector, channels in ((self.future_projector, self.future_dim),
                                            (self.current_dino_projector, self.dino_dim)):
                    previous = projector[0]
                    replacement = nn.Linear(channels * packed_channels, H)
                    with torch.no_grad():replacement.bias.copy_(previous.bias)
                    projector[0] = replacement

    def effective_step_sizes(self):
        if self.step_sizes is not None:
            return self.step_sizes()
        return self.cfg.solver.alpha_action, self.cfg.solver.alpha_future

    def _add_iteration(self, embeds, progress, direction):
        if self.iteration_conditioner is None:
            return embeds
        return embeds + self.iteration_conditioner(progress, direction).to(embeds)


    # ------------------------------------------------------------------
    # token builders
    # ------------------------------------------------------------------
    def build_future_tokens(
        self,
        Z: torch.Tensor,
        *,
        offsets: Optional[torch.Tensor] = None,
        future_mask: Optional[torch.Tensor] = None,
        iteration_progress=None,
    ) -> TokenGroup:
        """Z: [B,F,V,P,D] -> F*V*P/factor² tokens; candidates stay full resolution.

        ``future_mask`` may be reduced (``[B,F]`` / ``[B,F,V]`` / ``[B,F,V,P]``)
        or full (``[B,F,V,P,D]``); masked **values** are zeroed before the
        projector so they can never influence the token, and a token is valid
        iff at least one of its channels is.
        """
        if Z.dim() != 5:
            raise ValueError(f"Z must be [B,F,V,P,D], got {tuple(Z.shape)}")
        B, F, V, P, D = Z.shape
        if D != self.future_dim:
            raise ValueError(f"Z dim {D} != configured future_dim {self.future_dim}")
        if F != self.num_future_frames or V != self.num_cameras or P != self.future_patches:
            raise ValueError(
                f"Z geometry {(F, V, P)} != configured "
                f"{(self.num_future_frames, self.num_cameras, self.future_patches)}"
            )

        value_mask = _to_bool_mask(future_mask, Z, value_rank=5, batch=B)
        Zc = _apply_value_mask(Z, value_mask)
        Zc = pack_patches(Zc, self.patch_pack_factor)
        packed_mask = pack_patches(value_mask, self.patch_pack_factor)
        tok_mask_5d = _token_mask(packed_mask, trailing=1)
        P_tokens, D_tokens = Zc.shape[-2:]
        flat = Zc.reshape(B, F * V * P_tokens, D_tokens)
        emb = self.future_projector(flat) * self.cfg.framework.projector_output_scale
        emb = emb + self.encodings.future(B, F, V, P_tokens, Z.device, emb.dtype, offsets=offsets)
        emb = self._add_iteration(emb, iteration_progress, "future")
        mask = tok_mask_5d.reshape(B, F * V * P_tokens)
        emb = emb * mask.unsqueeze(-1).to(emb.dtype)
        return TokenGroup(embeds=emb, mask=mask, kind="future")

    def build_action_tokens(
        self, A: torch.Tensor, *, action_mask: Optional[torch.Tensor] = None, iteration_progress=None
    ) -> TokenGroup:
        """A: [B, H, Da] -> TokenGroup of length H.

        ``action_mask`` may be per-element ``[B, H, Da]`` or per-step ``[B, H]``.
        Masked **dimensions** are zeroed before the projector (so a huge or NaN
        value in an invalid slot cannot change the token), and a token is valid
        iff at least one of its dimensions is.
        """
        if A.dim() != 3:
            raise ValueError(f"A must be [B,H,Da], got {tuple(A.shape)}")
        B, H, Da = A.shape
        if Da != self.action_dim:
            raise ValueError(f"A dim {Da} != configured action_dim {self.action_dim}")

        value_mask = _to_bool_mask(action_mask, A, value_rank=3, batch=B)
        tok_mask = _token_mask(value_mask, trailing=1)          # [B,H]
        Ac = _apply_value_mask(A, value_mask)

        emb = self.action_projector(Ac) * self.cfg.framework.projector_output_scale
        emb = emb + self.encodings.action(B, H, A.device, emb.dtype)
        emb = self._add_iteration(emb, iteration_progress, "action")
        emb = emb * tok_mask.unsqueeze(-1).to(emb.dtype)
        return TokenGroup(embeds=emb, mask=tok_mask, kind="action")

    def build_energy_tokens(self, batch: int, device: torch.device, dtype: torch.dtype) -> TokenGroup:
        emb = self.energy_token.to(device=device, dtype=dtype).expand(batch, 1, -1)
        # The Energy token is ALWAYS valid — padding may only suppress keys.
        mask = torch.ones(batch, 1, device=device, dtype=torch.bool)
        return TokenGroup(embeds=emb, mask=mask, kind="energy")

    def build_future_placeholder_tokens(
        self, batch: int, offsets: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype
    ) -> TokenGroup:
        """Learned placeholder standing in for an absent future modality.

        Used by ``action_only`` mode.  The token count matches a real future
        block so the geometry is unchanged, but the mask is **False** everywhere
        so the placeholder can never be read as evidence of a real future.
        """
        F, V, P = self.num_future_frames, self.num_cameras, self.future_token_patches
        N = F * V * P
        base = self.future_placeholder.to(device=device, dtype=dtype).expand(batch, N, -1)
        base = base + self.encodings.placeholder_future(batch, N, device, dtype)
        # All keys masked: the placeholder is structural only.
        mask = torch.zeros(batch, N, device=device, dtype=torch.bool)
        return TokenGroup(embeds=base, mask=mask, kind="future_placeholder")

    def build_current_dino_tokens(
        self, z_cur: torch.Tensor, *, dino_mask: Optional[torch.Tensor] = None
    ) -> TokenGroup:
        """z_cur: [B,V,P,Ddino] -> V*P/factor² condition tokens."""
        if z_cur.dim() == 3:
            z_cur = z_cur.unsqueeze(1)
        B, V, P, D = z_cur.shape
        if self.patch_pack_factor > 1:
            value_mask = _to_bool_mask(dino_mask, z_cur, value_rank=4, batch=B)
            z_cur = pack_patches(_apply_value_mask(z_cur, value_mask), self.patch_pack_factor)
            packed_mask = pack_patches(value_mask, self.patch_pack_factor)
            dino_mask = _token_mask(packed_mask, trailing=1)
            P, D = z_cur.shape[-2:]
        emb = self.current_dino_projector(z_cur.reshape(B, V * P, D)) * self.cfg.framework.projector_output_scale
        emb = emb + self.encodings.dino_current(B, V, P, z_cur.device, emb.dtype)
        if dino_mask is None:
            mask = torch.ones(B, V * P, device=z_cur.device, dtype=torch.bool)
        else:
            m = dino_mask
            if m.dtype != torch.bool:
                m = m != 0
            mask = m.reshape(B, V * P)
        emb = emb * mask.unsqueeze(-1).to(emb.dtype)
        return TokenGroup(embeds=emb, mask=mask, kind="dino_cur")

    def build_state_tokens(self, q: torch.Tensor, *, state_mask: Optional[torch.Tensor] = None) -> TokenGroup:
        """q: [B, Dq] (or [B, T, Dq] with T=1) -> TokenGroup of length num_state_tokens."""
        if q.dim() == 3:
            if q.shape[1] != 1:
                raise ValueError(f"state time dim must be 1, got {q.shape[1]}")
            q = q[:, 0]
        B, Dq = q.shape
        if Dq != self.state_dim:
            raise ValueError(f"state dim {Dq} != configured state_dim {self.state_dim}")
        emb = self.state_projector(q).view(B, self.num_state_tokens, self.hidden_size) * self.cfg.framework.projector_output_scale
        emb = emb + self.encodings.state(B, self.num_state_tokens, q.device, emb.dtype)
        if state_mask is None:
            mask = torch.ones(B, self.num_state_tokens, device=q.device, dtype=torch.bool)
        else:
            m = state_mask
            if m.dtype != torch.bool:
                m = m != 0
            if m.dim() == 1:
                m = m[:, None].expand(B, self.num_state_tokens)
            mask = m
        emb = emb * mask.unsqueeze(-1).to(emb.dtype)
        return TokenGroup(embeds=emb, mask=mask, kind="state")

    # ------------------------------------------------------------------
    # energy forward
    # ------------------------------------------------------------------
    def _attach_extra_condition(self, cond: ConditionBundle) -> ConditionBundle:
        """Turn the backbone's raw ``extra_raw`` tensors into a TokenGroup.

        Doing this here (rather than in the backbone) keeps the projectors and
        the semantic encodings in one place **and** avoids the backbone holding
        a reference back to the core, which would create a submodule cycle.
        """
        raw = cond.native.get("extra_raw") if cond.native else None
        if not raw:
            return cond
        groups: List[TokenGroup] = []
        if "dino" in raw and raw["dino"] is not None:
            groups.append(self.build_current_dino_tokens(raw["dino"]))
        if "state" in raw and raw["state"] is not None:
            groups.append(self.build_state_tokens(raw["state"]))
        if not groups:
            return cond
        emb = torch.cat([g.embeds for g in groups], dim=1)
        mask = torch.cat([g.mask for g in groups], dim=1)
        cond.group = TokenGroup(embeds=emb, mask=mask, kind="cond_extra")
        return cond

    def energy_from_groups(
        self,
        cond: ConditionBundle,
        groups: List[TokenGroup],
    ) -> EnergyReadout:
        cond = self._attach_extra_condition(cond)
        hidden, e_index = self.backbone.run_sequence(cond, groups)
        B = hidden.shape[0]
        idx = e_index.view(B, 1, 1).expand(B, 1, hidden.shape[-1])
        h_e = hidden.gather(1, idx).squeeze(1)  # [B, D]
        e = self.energy_head(h_e)
        return EnergyReadout(energy=e, energy_token_index=e_index, hidden=h_e)

    def quadratic_prior_energy(self, cond, Z, A, *, action_mask=None, future_mask=None):
        """Known potential around zero action and current visual persistence.

        No future target enters the center; current raw DINO is standardized
        in the model's condition builder, with the train-only future stats.
        Precision anchors are checkpointed constants, not learned step sizes.
        """
        am = _to_bool_mask(action_mask,A,value_rank=3,batch=A.shape[0])
        az = _apply_value_mask(A,am).float().square().flatten(1).sum(1)
        current = ((cond.native or {}).get("extra_raw") or {}).get("future_prior_center")
        if current is None:
            raise ValueError("quadratic future prior needs the current-only standardized DINO center")
        center = current.to(Z).unsqueeze(1)
        zm = _to_bool_mask(future_mask,Z,value_rank=5,batch=Z.shape[0])
        dz = _apply_value_mask(Z-center,zm).float().square().flatten(1).sum(1)
        return .5*(self.cfg.framework.quadratic_action_weight*self.prior_action_precision*az
                   + self.cfg.framework.quadratic_future_weight*self.prior_future_precision*dz)

    def energy(
        self,
        cond: ConditionBundle,
        Z: torch.Tensor,
        A: torch.Tensor,
        *,
        future_mask: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        iteration_progress=None,
    ) -> torch.Tensor:
        """``energy(condition, future_candidate, action_candidate, masks) -> [B]``.

        No cross-sample normalisation or aggregation is performed anywhere: each
        sample's scalar depends only on that sample's tokens.
        """
        fut = self.build_future_tokens(Z, offsets=offsets, future_mask=future_mask, iteration_progress=iteration_progress)
        act = self.build_action_tokens(A, action_mask=action_mask, iteration_progress=iteration_progress)
        eng = self.build_energy_tokens(Z.shape[0], Z.device, Z.dtype)
        energy = self.energy_from_groups(cond, [fut, act, eng]).energy
        if self.cfg.framework.quadratic_candidate_prior:
            energy = energy + self.quadratic_prior_energy(cond,Z,A,action_mask=action_mask,future_mask=future_mask)
        return energy

    def energy_action_only(
        self,
        cond: ConditionBundle,
        A: torch.Tensor,
        *,
        action_mask: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        iteration_progress=None,
    ) -> torch.Tensor:
        """Independent ``E(C, A)`` path.

        The future block is replaced by a learned placeholder whose tokens are
        fully masked (see :meth:`build_future_placeholder_tokens`), so this path
        cannot read any future ground truth.
        """
        B = A.shape[0]
        if self.cfg.framework.use_learned_future_placeholder:
            fut = self.build_future_placeholder_tokens(B, offsets, A.device, A.dtype)
        else:
            F, V, P = self.num_future_frames, self.num_cameras, self.future_patches
            zeros = torch.zeros(
                B, F, V, P, self.future_dim, device=A.device, dtype=torch.float32
            )
            fut = self.build_future_tokens(zeros, offsets=offsets, iteration_progress=iteration_progress)
            fut = TokenGroup(embeds=fut.embeds * 0, mask=torch.zeros_like(fut.mask), kind="future_placeholder")
        act = self.build_action_tokens(A, action_mask=action_mask, iteration_progress=iteration_progress)
        eng = self.build_energy_tokens(B, A.device, A.dtype)
        energy = self.energy_from_groups(cond, [fut, act, eng]).energy
        if self.cfg.framework.quadratic_candidate_prior:
            mask = _to_bool_mask(action_mask,A,value_rank=3,batch=A.shape[0])
            energy = energy + .5*self.cfg.framework.quadratic_action_weight*self.prior_action_precision*_apply_value_mask(A,mask).float().square().flatten(1).sum(1)
        return energy
