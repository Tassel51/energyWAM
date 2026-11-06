"""Small helper ops for the energy solver.

Gradient behaviour is documented explicitly because the engineering requirements
ask for it: clamping affects the gradient and must not be a silent surprise.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def clip_candidate_grad(
    grad: Optional[torch.Tensor],
    max_abs: Optional[float],
) -> Optional[torch.Tensor]:
    """Per-element gradient clip: ``grad.clamp(-max_abs, max_abs)``.

    ``None`` disables clipping.  This is a *candidate-space* clip and is named
    separately from the outer optimiser's ``max_grad_norm``.
    """
    if grad is None or max_abs is None:
        return grad
    return grad.clamp(min=-float(max_abs), max=float(max_abs))


def apply_candidate_constraints(
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    clamp: Tuple[Optional[float], Optional[float]],
) -> torch.Tensor:
    """Clamp candidates into a box, then re-apply the mask.

    Gradient behaviour
    ------------------
    ``torch.clamp`` is differentiable and forwards the gradient for elements
    strictly inside ``(lo, hi)``; elements sitting on/outside a bound get zero
    gradient through the clamp.  This is the documented behaviour of the action
    constraint and is why ``action_clamp_min/max`` default to ``None`` (no
    clamp).  Masked elements are hard-zeroed (zero gradient) so padding never
    moves.

    Note: we deliberately do **not** clamp latent future features to the action
    bounds — only the action candidates get an action-space box, and only when
    explicitly configured.
    """
    lo, hi = clamp
    if lo is not None or hi is not None:
        x = x.clamp(min=lo, max=hi)
    if mask is not None:
        m = mask
        if m.dtype != torch.bool:
            m = m != 0
        while m.dim() < x.dim():
            m = m.unsqueeze(-1)
        m = torch.broadcast_to(m, x.shape)
        x = torch.where(m, x, torch.zeros_like(x))
    return x


def precondition_candidate_grad(grad, mask=None, mode='none', epsilon=1e-6):
    """Optional differentiable per-sample block-RMS preconditioner.

    g / sqrt(mean_valid(g**2) + epsilon**2). No detach: outer gradients
    differentiate through this scale as well as through the energy gradient.
    A positive scalar rescales each sample's descent direction. Alpha now
    controls approximate coordinate-RMS displacement rather than raw g units.
    """
    if mode == 'none':
        return grad
    if mode != 'rms':
        raise ValueError(f'unknown candidate gradient preconditioner {mode!r}')
    if epsilon <= 0:
        raise ValueError('gradient RMS epsilon must be positive')
    m = torch.ones_like(grad, dtype=torch.bool) if mask is None else torch.as_tensor(mask,device=grad.device,dtype=torch.bool)
    while m.dim()<grad.dim():m=m.unsqueeze(-1)
    m=torch.broadcast_to(m,grad.shape)
    g=torch.where(m,grad,torch.zeros_like(grad))
    work=g.double() if g.dtype==torch.float64 else g.float()
    dims=tuple(range(1,g.dim()))
    count=m.sum(dim=dims,keepdim=True).clamp_min(1)
    rms=(work.square().sum(dim=dims,keepdim=True)/count + epsilon**2).sqrt()
    return (work/rms).to(grad.dtype)
