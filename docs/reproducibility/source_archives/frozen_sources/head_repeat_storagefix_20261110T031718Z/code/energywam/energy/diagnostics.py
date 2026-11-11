"""Offline Action–Future coupling diagnostics (paper §4.9, §6.4.2–6.4.3).

Scope and honesty rules
-----------------------
These functions quantify **mathematical dependence and energy ordering only**:

* a non-zero cross-gradient sensitivity (Eq. 44) shows that the energy is not
  separable in ``(A, Z)`` — it does **not** show physical correctness;
* a high matched/mismatched ranking AUC (Eq. 45) shows ordering on the supplied
  pairs — it is **not** a calibrated confidence or a success probability;
* energy magnitudes themselves are unidentifiable from the endpoint loss
  (paper §3.2) and are never reported as probabilities.

Negative pairs MUST be supplied by the caller together with an explicit
validity flag.  This module never constructs, shuffles or re-pairs negatives:
an automatically shuffled ``Z`` from another sample is not guaranteed to be a
physically wrong transition, and claiming so would fabricate a label
(paper §4.9: “不同演示未来并不天然都是错误负样本”).

Nothing here feeds back into training: the module is evaluation-only.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .losses import masked_cosine, masked_mse


INTERPRETATION = (
    "Mathematical-dependence / ordering diagnostic ONLY. Non-zero sensitivity or "
    "high ranking AUC does NOT establish physical correctness, causal "
    "understanding, or calibrated confidence; energies are not probabilities."
)
NEGATIVE_POLICY = (
    "Negatives are caller-supplied with explicit validity flags; this tool never "
    "constructs or shuffles negatives and never labels an unverified pair wrong."
)


def _as_bool_mask(mask, ref: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    m = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask)
    m = m.to(device=ref.device)
    return m if m.dtype == torch.bool else m != 0


def _masked(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x
    m = mask
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    m = torch.broadcast_to(m, x.shape)
    return torch.where(m, x, torch.zeros_like(x))


def _norms(x: torch.Tensor) -> torch.Tensor:
    """Per-sample L2 norm: ``x [B, ...] -> [B]``."""
    return x.reshape(x.shape[0], -1).norm(dim=-1)


# ---------------------------------------------------------------------------
# (i) conditional gradient sensitivity — Eq. 44
# ---------------------------------------------------------------------------
@torch.inference_mode(False)
@torch.enable_grad()
def _cross_sensitivity(
    energy_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    fixed: torch.Tensor,
    other1: torch.Tensor,
    other2: torch.Tensor,
    *,
    fixed_name: str,
    other_name: str,
    fixed_mask: Optional[torch.Tensor] = None,
    other_mask: Optional[torch.Tensor] = None,
    eps: float,
) -> Dict[str, torch.Tensor]:
    """S_{fixed <- other} = E[ ||g(fixed|other1) - g(fixed|other2)|| / (||g(fixed|other1)|| + eps) ].

    Both gradients are taken at the **same** ``fixed`` candidate (Eq. 44 holds
    ``C, A`` fixed and swaps ``Z``; the symmetric direction holds ``C, Z``
    fixed and swaps ``A``).
    """
    if other1.shape != other2.shape:
        raise ValueError("the two alternative candidates must have the same shape")
    B = fixed.shape[0]

    f = fixed.detach().clone().requires_grad_(True)
    o1 = other1.detach().clone()
    e1 = energy_fn(f, o1) if fixed_name == "action" else energy_fn(o1, f)
    g1 = torch.autograd.grad(e1.sum(), [f], retain_graph=False, create_graph=False)[0]

    f2 = fixed.detach().clone().requires_grad_(True)
    o2 = other2.detach().clone()
    e2 = energy_fn(f2, o2) if fixed_name == "action" else energy_fn(o2, f2)
    g2 = torch.autograd.grad(e2.sum(), [f2], retain_graph=False, create_graph=False)[0]

    g1 = _masked(g1, _as_bool_mask(fixed_mask, g1))
    g2 = _masked(g2, _as_bool_mask(fixed_mask, g2))
    diff = _norms(g1 - g2)
    denom = _norms(g1) + float(eps)
    s = diff / denom
    return {
        "sensitivity_per_sample": s.detach(),
        "sensitivity_mean": s.mean().detach(),
        "sensitivity_max": s.max().detach(),
        "grad_norm_reference": _norms(g1).mean().detach(),
        "grad_norm_alternative": _norms(g2).mean().detach(),
        "num_samples": torch.tensor(float(B)),
    }


def action_gradient_future_sensitivity(
    energy_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    A: torch.Tensor,
    Z1: torch.Tensor,
    Z2: torch.Tensor,
    *,
    action_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Dict[str, object]:
    """Eq. 44: ``S_{A<-Z}`` — how much the action gradient changes with the future.

    ``energy_fn(A, Z) -> [B]``.  ``Z1`` / ``Z2`` are two future candidates for
    the **same** condition and the **same** ``A`` (typically the matched future
    and a caller-supplied alternative).  The result says the future candidate
    influences the action update direction; it does not say either future is
    physically correct or incorrect.
    """
    out = _cross_sensitivity(
        energy_fn, A, Z1, Z2,
        fixed_name="action", other_name="future",
        fixed_mask=action_mask, other_mask=None, eps=eps,
    )
    out["metric"] = "S_A_from_Z (Eq. 44)"
    out["interpretation"] = INTERPRETATION
    return out


def future_gradient_action_sensitivity(
    energy_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    Z: torch.Tensor,
    A1: torch.Tensor,
    A2: torch.Tensor,
    *,
    future_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Dict[str, object]:
    """Symmetric of Eq. 44: ``S_{Z<-A}`` — future gradient change under swapped actions."""
    out = _cross_sensitivity(
        energy_fn, Z, A1, A2,
        fixed_name="future", other_name="action",
        fixed_mask=future_mask, other_mask=None, eps=eps,
    )
    out["metric"] = "S_Z_from_A (Eq. 44, symmetric)"
    out["interpretation"] = INTERPRETATION
    return out


# ---------------------------------------------------------------------------
# condition-swap action trajectory (§6.4.3)
# ---------------------------------------------------------------------------
@torch.inference_mode(False)
@torch.enable_grad()
def condition_swap_action_trajectory(
    energy_fn_c1: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    energy_fn_c2: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    A0: torch.Tensor,
    Z_fixed: torch.Tensor,
    *,
    num_steps: int = 4,
    alpha_action: float = 0.05,
    action_mask: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """Action-only refinement under two conditions from the same ``A0``.

    Uses :func:`energywam.energy.solver.refine` (``action_given_future``,
    inference mode) so the update rule is exactly the deployed one; the
    per-step candidate is captured from inside the energy callback.  The
    reported distances compare the two trajectories step by step: a non-zero
    distance means the condition changes the action optimisation path, nothing
    more.
    """
    from .solver import refine

    def _run(energy_fn):
        seen: List[torch.Tensor] = []

        def wrapped(a, z):
            seen.append(a.detach().clone())
            return energy_fn(a, z)

        A_K, _, _ = refine(
            wrapped,
            action=A0.detach().clone().requires_grad_(True),
            future=Z_fixed.detach().clone(),
            action_mask=action_mask,
            future_mask=None,
            mode="action_given_future",
            num_steps=num_steps,
            alpha_action=alpha_action,
            alpha_future=0.0,
            training=False,
            keep_state=True,
        )
        seen.append(A_K.detach().clone())
        return seen

    traj1 = _run(energy_fn_c1)
    traj2 = _run(energy_fn_c2)
    n = min(len(traj1), len(traj2))
    per_step = torch.stack(
        [_norms(traj1[k] - traj2[k]) for k in range(n)]
    )  # [n, B]
    return {
        "metric": "condition-swap action trajectory distance",
        "per_step_l2_per_sample": per_step.detach(),
        "per_step_l2_mean": per_step.mean(dim=1).detach(),
        "endpoint_l2_mean": per_step[-1].mean().detach(),
        "num_steps": torch.tensor(float(n - 1)),
        "interpretation": INTERPRETATION,
    }


# ---------------------------------------------------------------------------
# (ii) transition-compatibility ranking — Eq. 45 / §6.4.3
# ---------------------------------------------------------------------------
def energy_ranking_metrics(
    e_matched: torch.Tensor,
    e_mismatched: torch.Tensor,
    negatives_valid: torch.Tensor,
    *,
    margin: float = 0.0,
) -> Dict[str, object]:
    """Rank matched vs caller-supplied mismatched pairs (lower energy = better).

    Parameters
    ----------
    e_matched :
        ``[N]`` energies of the matched ``(C, A*, Z*)`` pairs.
    e_mismatched :
        ``[N]`` paired negatives ``(C, A*, Z^-)`` or ``[N, M]`` several
        negatives per condition.  Must be provided by the caller — this
        function refuses to run without them and never builds its own.
    negatives_valid :
        ``[N]`` / ``[N, M]`` bool flags declaring which supplied negatives are
        *verified* mismatches (e.g. judged from a known state transition).
        Invalid entries are excluded from every statistic.  Required even if
        all-True: the label is the caller's claim, and it must be explicit.

    Returns paired ranking accuracy (``E_match + margin < E_mismatch``),
    conditional concordance (tie credit .5, no margin), and pooled ROC AUC
    comparing every valid positive to every valid negative (tie credit .5,
    independent of margin). Pooled AUC depends on offsets across conditions;
    conditional concordance does not. Neither is calibrated confidence.
    Also reports pairwise energy gaps and exact counts used.
    """
    if e_mismatched is None or (isinstance(e_mismatched, torch.Tensor) and e_mismatched.numel() == 0):
        raise ValueError(
            "energy_ranking_metrics requires caller-supplied mismatched pairs "
            "(paper §4.9: negatives must be constructed and validated by the "
            "caller; this tool never fabricates them)."
        )
    if negatives_valid is None:
        raise ValueError(
            "negatives_valid is required: the caller must declare which supplied "
            "negatives are verified mismatches."
        )
    ex = e_mismatched.detach()
    em = e_matched.detach().to(device=ex.device).reshape(-1)
    valid = negatives_valid if isinstance(negatives_valid, torch.Tensor) else torch.as_tensor(negatives_valid)
    valid = (valid != 0).to(device=ex.device)

    if ex.dim() == 1:
        if ex.shape[0] != em.shape[0] or valid.shape != ex.shape:
            raise ValueError("paired negatives must match e_matched in shape")
        pair_match = em.unsqueeze(1)  # [N,1]
        pair_mis = ex.unsqueeze(1)    # [N,1]
        pair_valid = valid.unsqueeze(1)
    elif ex.dim() == 2:
        if ex.shape[0] != em.shape[0] or valid.shape != ex.shape:
            raise ValueError("negative energies and validity flags must both be [N, M]")
        pair_match = em.unsqueeze(1).expand_as(ex)
        pair_mis = ex
        pair_valid = valid
    else:
        raise ValueError(f"e_mismatched must be [N] or [N, M], got {tuple(ex.shape)}")

    vm = pair_valid
    n_used = int(vm.sum())
    if n_used == 0:
        raise ValueError("no valid negative pairs after applying negatives_valid")
    match_flat = pair_match[vm]
    mis_flat = pair_mis[vm]
    wins = (match_flat + margin) < mis_flat
    ties = (match_flat + margin) == mis_flat
    accuracy = wins.float().mean()
    if not torch.isfinite(match_flat).all() or not torch.isfinite(mis_flat).all():
        raise ValueError("valid matched/mismatched energies must be finite")
    # Conditional paired concordance is not the pooled ROC AUC. Report both.
    conditional_concordance = (
        (match_flat < mis_flat).float() + 0.5 * (match_flat == mis_flat).float()
    ).mean()
    # AUC = P(E_positive < E_negative) + .5 P(tie), without a margin.
    # Keep one positive per valid condition, even when it has many negatives.
    positives = em[vm.any(dim=1)]
    negatives = mis_flat.sort().values.contiguous()
    common_dtype = torch.promote_types(positives.dtype, negatives.dtype)
    positives = positives.to(common_dtype).contiguous()
    negatives = negatives.to(common_dtype)
    left = torch.searchsorted(negatives, positives, right=False)
    right = torch.searchsorted(negatives, positives, right=True)
    auc = ((negatives.numel() - right).double() + 0.5 * (right - left).double()).mean() / negatives.numel()
    gap = (mis_flat - match_flat)
    return {
        "metric": "transition-compatibility energy ranking (Eq. 45)",
        "ranking_accuracy": accuracy.detach(),
        "auc": auc.detach(),
        "auc_scope": "pooled ROC AUC; sensitive to energy offsets across conditions",
        "conditional_concordance": conditional_concordance.detach(),
        "margin_mean": gap.mean().detach(),
        "margin_min": gap.min().detach(),
        "num_pairs_used": torch.tensor(float(n_used)),
        "num_negatives_provided": torch.tensor(float(pair_mis.numel())),
        "num_negatives_valid": torch.tensor(float(n_used)),
        "interpretation": INTERPRETATION,
        "negative_policy": NEGATIVE_POLICY,
    }


# ---------------------------------------------------------------------------
# future-latent metrics — Eq. 53 / §6.4.2
# ---------------------------------------------------------------------------
def future_latent_metrics(
    Z_pred: torch.Tensor,
    Z_target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    space: str,
) -> Dict[str, object]:
    """Masked future MSE (Eq. 53) + patch-wise cosine similarity.

    ``space`` is **required**: paper §6.4.2 asks whether the metric is computed
    in the normalised feature space or the raw DINO space, and the report must
    say which.  Cosine is reported as a similarity (1 - the internal cosine
    distance) on the channel dim of each patch token.
    """
    if not space:
        raise ValueError("space= is required (e.g. 'raw_dino' or 'normalized_Z'): the "
                         "metric is meaningless without knowing its coordinate system")
    if Z_pred.numel() == 0 or (mask is not None and not bool(torch.as_tensor(mask).any())):
        raise ValueError("future metrics require at least one valid supervised element")
    Zp = Z_pred.detach()
    Zt = Z_target.detach()
    mse = masked_mse(Zp, Zt, mask)
    cos_dist = masked_cosine(Zp, Zt, mask)
    return {
        "metric": "future latent MSE + cosine",
        "space": space,
        "mse": mse.detach(),
        "cosine_similarity": (1.0 - cos_dist).detach(),
        "interpretation": INTERPRETATION,
    }
