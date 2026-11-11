"""Masked losses for action / future endpoints.

The engineering requirements are explicit: ``J_A`` and ``J_Z`` are mean squared
errors **normalised by the number of valid elements**.  A 200k-dim latent error
must not be summed and added to a 16x12 action error.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def broadcast_bool_mask(mask: Optional[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """Return a bool mask broadcast to ``target``'s shape.

    ``mask=None`` means "everything valid".  Reduced masks are **batch-first**
    (the EnergyWAM convention):

    * ``[B, H, Da]``     for action (same shape as target)
    * ``[B, H]``         per-step action mask
    * ``[B, F, V, P]``   for future latents of shape ``[B, F, V, P, D]``
    * ``[B, F, V]`` / ``[B, F]``  per-camera / per-frame future mask

    In addition, any shape that broadcasts to ``target`` is accepted: standard
    right-aligned broadcast shapes (``[B, 1, 1, 1, 1]``, ``[1, F, V, P, 1]``,
    ...) and batch-shared reduced forms (``[F]``, ``[F, V]``, ``[F, V, P]``,
    ...).  A mask that cannot be broadcast raises ``ValueError``.
    """
    if mask is None:
        return torch.ones_like(target, dtype=torch.bool)
    m = mask.to(device=target.device)
    if m.dtype != torch.bool:
        m = m != 0
    if m.shape == target.shape:
        return m
    candidates = []
    if m.dim() < target.dim():
        # batch-first reduced: left-align by appending singleton dims
        left = m
        while left.dim() < target.dim():
            left = left.unsqueeze(-1)
        if m.shape[0] == target.shape[0]:
            candidates.append(left)
        # standard right-aligned broadcast (torch semantics)
        candidates.append(m)
        # batch-shared reduced form: missing batch dim + trailing singletons
        shared = m.unsqueeze(0)
        while shared.dim() < target.dim():
            shared = shared.unsqueeze(-1)
        candidates.append(shared)
        if m.shape[0] != target.shape[0]:
            candidates.append(left)  # left-align last, only if nothing else fits
    else:
        candidates.append(m)
    for cand in candidates:
        try:
            return torch.broadcast_to(cand, target.shape)
        except RuntimeError:
            continue
    raise ValueError(f"mask shape {tuple(mask.shape)} not broadcastable to {tuple(target.shape)}")


# historical private name
_broadcast_mask = broadcast_bool_mask


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """``sum((pred-target)^2 * mask) / (num_valid)`` over valid elements.

    Invalid (``mask == False``) entries of ``pred``/``target`` are replaced by
    zero *before* the subtraction — never multiplied by zero afterwards — so a
    NaN/Inf sitting in a masked-out entry cannot poison the result via
    ``NaN * 0``.  The computation keeps the promoted dtype of the inputs
    (float64 is preserved for the numerical tests).

    Returns a scalar.  If the mask is entirely empty the result is a
    differentiable 0 (same dtype, still on the autograd graph).
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    m = broadcast_bool_mask(mask, target)
    common = torch.promote_types(pred.dtype, target.dtype)
    zero = torch.zeros((), dtype=common, device=pred.device)
    p = torch.where(m, pred.to(common), zero)
    t = torch.where(m, target.to(common), zero)
    diff2 = (p - t) ** 2
    num_valid = int(m.sum())
    if num_valid == 0:
        # differentiable zero, dtype-preserving (no detached new_zeros)
        return diff2.sum() * 0.0
    denom = m.sum().to(common).clamp_min(eps)
    return diff2.sum() / denom


def masked_cosine(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Optional cosine distance on the channel dim of future latents.

    Default weight is 0 — this is **not** part of the reference method.

    Partial channel masks have a defined meaning: masked channels of both
    ``pred`` and ``target`` are zero-filled before the dot product and the two
    norms, which is exactly the cosine similarity of the valid sub-vectors
    (zero coordinates contribute nothing).  A token (everything except the
    trailing channel dim) contributes iff **at least one** of its channels is
    valid; tokens with no valid channel are excluded.  Invalid entries are
    sanitised first so NaN/Inf in them cannot poison the reduction via
    ``NaN * 0``.  An empty mask yields a differentiable 0 (promoted dtype).
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    m = broadcast_bool_mask(mask, target)
    common = torch.promote_types(pred.dtype, target.dtype)
    zero = torch.zeros((), dtype=common, device=pred.device)
    p = torch.where(m, pred.to(common), zero)
    t = torch.where(m, target.to(common), zero)
    cos = F.cosine_similarity(p, t, dim=-1, eps=eps)  # [..., tokens]
    token_m = m.any(dim=-1)  # a token needs at least one valid channel
    num_valid = int(token_m.sum())
    if num_valid == 0:
        return cos.sum() * 0.0
    denom = token_m.sum().to(common).clamp_min(eps)
    return ((1.0 - cos) * token_m).sum() / denom


def valid_element_count(mask: Optional[torch.Tensor], target: torch.Tensor) -> int:
    m = broadcast_bool_mask(mask, target)
    return int(m.sum())


def action_endpoint_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    kind: str = "masked_mse",
) -> torch.Tensor:
    if kind == "masked_mse":
        return masked_mse(pred, target, mask)
    raise ValueError(f"unknown action loss {kind!r}")


def future_endpoint_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    kind: str = "masked_mse",
    cosine_weight: float = 0.0,
) -> torch.Tensor:
    if kind == "masked_mse":
        out = masked_mse(pred, target, mask)
    else:
        raise ValueError(f"unknown future loss {kind!r}")
    if cosine_weight and cosine_weight > 0:
        out = out + float(cosine_weight) * masked_cosine(pred, target, mask)
    return out
