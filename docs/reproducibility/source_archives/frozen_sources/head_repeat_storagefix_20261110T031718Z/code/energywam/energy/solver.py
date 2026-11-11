"""Full-unroll energy-gradient solver.

Reference semantics (from the engineering requirements, section 6/7)::

    gA, gZ = autograd.grad(E.sum(), [A, Z], create_graph=training)
    A_next = A - alpha_A * gA
    Z_next = Z - alpha_Z * gZ

* Both gradients are taken at the **same** point ``(A_k, Z_k)`` and both updates
  are applied **simultaneously** to produce ``(A_{k+1}, Z_{k+1})``.
* ``E.sum()`` (not ``E.mean()``) so each sample's candidate gradient does not
  shrink with batch size; the outer loss does the batch averaging.
* During training ``create_graph=True`` and nothing on the candidate path is
  detached, so the outer loss back-propagates *through* the unroll into the
  energy parameters.  The inner loop never steps an optimiser.
* During inference candidates are re-created (``detach().requires_grad_(True)``)
  at the top of every step so the history graph is released; parameters are not
  updated and ``autograd.grad`` (not ``backward``) is used so no ``.grad`` is
  accumulated on parameters.

The three optimisation modes are:

``action_given_future``  Z is pinned to the ground-truth future target Z*,
                         only A is refined.
``future_given_action``  A is pinned to the ground-truth action target A*,
                         only Z is refined.
``joint``                both A and Z are refined together.

``unroll_mode='truncated'`` is an explicit, off-by-default *deviation* that
detaches the first ``K - truncation_length`` steps.  It is never substituted for
the full unroll silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from .solver_utils import (  # noqa: F401  (re-exported for convenience)
    precondition_candidate_grad,
    apply_candidate_constraints,
    clip_candidate_grad,
)


@dataclass
class SolverState:
    """One refinement trajectory (kept for diagnostics)."""

    iteration_conditioned: bool = False
    energies: List[torch.Tensor] = field(default_factory=list)  # [B] each
    action_grad_norms: List[torch.Tensor] = field(default_factory=list)
    future_grad_norms: List[torch.Tensor] = field(default_factory=list)
    action_update_norms: List[torch.Tensor] = field(default_factory=list)
    future_update_norms: List[torch.Tensor] = field(default_factory=list)

    def as_diagnostics(self) -> Dict[str, torch.Tensor]:
        """Compact scalar diagnostics for logging.

        Energy values are a *diagnostic only* — they are not a success
        probability or a calibrated confidence.
        """
        out: Dict[str, torch.Tensor] = {"energy_condition_changes": torch.tensor(float(self.iteration_conditioned))}
        if self.energies:
            e = torch.stack(self.energies)  # [K, B]
            out["energy_first"] = e[0].mean().detach()
            out["energy_last"] = e[-1].mean().detach()
            out["energy_delta"] = (e[-1] - e[0]).mean().detach()
            out["energy_min"] = e.min().detach()
            out["energy_max"] = e.max().detach()
            out["energy_nonfinite"] = (~torch.isfinite(e)).sum().float().detach()
            # full per-step trajectory, reduced to a fixed-size vector so the
            # jsonl stays parseable regardless of K
            traj = e.mean(dim=1).detach()
            out["energy_traj_len"] = torch.tensor(float(traj.numel()))
            for i, v in enumerate(traj.tolist()):
                out[f"energy_traj_{i}"] = torch.tensor(float(v))
        if self.action_grad_norms:
            g = torch.stack([t.mean() for t in self.action_grad_norms])
            out["action_grad_norm_first"] = g[0].detach()
            out["action_grad_norm_last"] = g[-1].detach()
            out["action_grad_norm_mean"] = g.mean().detach()
        if self.future_grad_norms:
            g = torch.stack([t.mean() for t in self.future_grad_norms])
            out["future_grad_norm_first"] = g[0].detach()
            out["future_grad_norm_last"] = g[-1].detach()
            out["future_grad_norm_mean"] = g.mean().detach()
        if self.action_update_norms:
            u = torch.stack([t.mean() for t in self.action_update_norms])
            out["action_update_norm_mean"] = u.mean().detach()
            out["action_update_norm_max"] = u.max().detach()
        if self.future_update_norms:
            u = torch.stack([t.mean() for t in self.future_update_norms])
            out["future_update_norm_mean"] = u.mean().detach()
            out["future_update_norm_max"] = u.max().detach()
        out["solver_steps"] = torch.tensor(float(len(self.energies)))
        return out


def _masked(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x
    m = mask
    if m.dtype != torch.bool:
        m = m != 0
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    m = torch.broadcast_to(m, x.shape)
    return torch.where(m, x, torch.zeros_like(x))


def refine(
    energy_fn: Callable[..., torch.Tensor],
    *,
    action: torch.Tensor,
    future: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    future_mask: Optional[torch.Tensor] = None,
    mode: str = "joint",
    num_steps: int = 4,
    alpha_action: float = 0.05,
    alpha_future: float = 0.05,
    training: bool = True,
    clip_action_grad: Optional[float] = None,
    clip_future_grad: Optional[float] = None,
    action_clamp: Tuple[Optional[float], Optional[float]] = (None, None),
    future_clamp: Tuple[Optional[float], Optional[float]] = (None, None),
    unroll_mode: str = "full",
    truncation_length: int = 2,
    joint_update_mode: str = "simultaneous",
    langevin_noise_std: float = 0.0,
    keep_state: bool = True,
    iteration_aware: bool = False,
    candidate_gradient_normalization: str = "none",
    gradient_rms_epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, SolverState]:
    """Run ``num_steps`` energy-gradient refinements.

    Parameters
    ----------
    energy_fn :
        ``energy_fn(action, future) -> Tensor[B]``.  The condition is closed
        over.  Must be differentiable w.r.t. both candidates.
    action, future :
        Candidate tensors.  Shapes: ``[B, H, Da]`` and ``[B, F, V, P, D]``.
        During training these must already require grad and be float32.
    mode :
        ``joint`` | ``action_given_future`` | ``future_given_action``.

    Returns
    -------
    (action_final, future_final, state)
    """
    if candidate_gradient_normalization not in ("none", "rms"):
        raise ValueError("unknown candidate gradient normalization")
    if gradient_rms_epsilon <= 0:
        raise ValueError("gradient_rms_epsilon must be positive")
    if mode not in ("joint", "action_given_future", "future_given_action"):
        raise ValueError(f"unknown refine mode {mode!r}")
    if joint_update_mode not in ("simultaneous", "alternating"):
        raise ValueError(f"unknown joint_update_mode {joint_update_mode!r}")
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")

    update_action = mode in ("joint", "action_given_future")
    update_future = mode in ("joint", "future_given_action")

    A = action
    Z = future
    state = SolverState(iteration_conditioned=iteration_aware)

    # Which steps keep the graph?  ``full`` keeps all of them.
    if unroll_mode == "full":
        live_from = 0
    elif unroll_mode == "truncated":
        # This optional approximation is NOT part of the reference method and
        # has not been validated.  A previous version ran the leading steps
        # under ``torch.no_grad`` and then called ``autograd.grad``, which is
        # guaranteed to fail; rather than ship a broken option we fail fast and
        # tell the caller to use ``unroll_mode='full'``.
        raise NotImplementedError(
            "unroll_mode='truncated' is an unverified deviation from the full "
            "unroll and is not enabled in this build.  Use unroll_mode='full'."
        )
    else:
        raise ValueError(f"unknown unroll_mode {unroll_mode!r}")

    from functools import partial
    for k in range(num_steps):
        # Bind by value: checkpoint recomputation must not see a later k.
        step_energy = (partial(energy_fn, iteration_progress=k / num_steps)
                       if iteration_aware else energy_fn)
        if training:
            if k < live_from:
                # truncated-unroll deviation: run these steps without graph
                with torch.no_grad():
                    A, Z = _step(
                        step_energy,
                        A,
                        Z,
                        action_mask,
                        future_mask,
                        update_action,
                        update_future,
                        alpha_action,
                        alpha_future,
                        clip_action_grad,
                        clip_future_grad,
                        action_clamp,
                        future_clamp,
                        joint_update_mode,
                        langevin_noise_std,
                        create_graph=False,
                        state=state if keep_state else None,
                        candidate_gradient_normalization=candidate_gradient_normalization,
                        gradient_rms_epsilon=gradient_rms_epsilon,
                    )
                A = A.detach().requires_grad_(update_action)
                Z = Z.detach().requires_grad_(update_future)
                continue
            A, Z = _step(
                step_energy,
                A,
                Z,
                action_mask,
                future_mask,
                update_action,
                update_future,
                alpha_action,
                alpha_future,
                clip_action_grad,
                clip_future_grad,
                action_clamp,
                future_clamp,
                joint_update_mode,
                langevin_noise_std,
                create_graph=True,
                state=state if keep_state else None,
                        candidate_gradient_normalization=candidate_gradient_normalization,
                        gradient_rms_epsilon=gradient_rms_epsilon,
            )
        else:
            # Inference: release the history graph every step.  Parameters are
            # never touched — autograd.grad only, so no .grad accumulates.
            with torch.enable_grad():
                A = A.detach().requires_grad_(update_action)
                Z = Z.detach().requires_grad_(update_future)
                A, Z = _step(
                    step_energy,
                    A,
                    Z,
                    action_mask,
                    future_mask,
                    update_action,
                    update_future,
                    alpha_action,
                    alpha_future,
                    clip_action_grad,
                    clip_future_grad,
                    action_clamp,
                    future_clamp,
                    joint_update_mode,
                    langevin_noise_std,
                    create_graph=False,
                    state=state if keep_state else None,
                        candidate_gradient_normalization=candidate_gradient_normalization,
                        gradient_rms_epsilon=gradient_rms_epsilon,
                )

    return A, Z, state


def _step(
    energy_fn,
    A: torch.Tensor,
    Z: torch.Tensor,
    action_mask,
    future_mask,
    update_action: bool,
    update_future: bool,
    alpha_action: float,
    alpha_future: float,
    clip_action_grad,
    clip_future_grad,
    action_clamp,
    future_clamp,
    joint_update_mode: str,
    langevin_noise_std: float,
    create_graph: bool,
    state: Optional[SolverState],
    candidate_gradient_normalization="none",
    gradient_rms_epsilon=1e-6,
):
    want: List[torch.Tensor] = []
    flags: List[bool] = []
    if update_action:
        want.append(A)
        flags.append(True)
    if update_future:
        want.append(Z)
        flags.append(True)

    E = energy_fn(A, Z)  # [B]
    if state is not None:
        state.energies.append(E.detach())

    grads = torch.autograd.grad(
        E.sum(),
        want,
        create_graph=create_graph,
        retain_graph=create_graph,
        # allow_unused=False: if the energy really does not depend on a free
        # candidate, that is a broken model and must surface as an error rather
        # than being papered over with a zero gradient.
        allow_unused=False,
    )

    gA = gZ = None
    gi = 0
    if update_action:
        gA = grads[gi]
        gi += 1
    if update_future:
        gZ = grads[gi]

    # Mask + optional per-element gradient clipping (before the update).
    if gA is not None:
        gA = _masked(gA, action_mask)
        gA = clip_candidate_grad(gA, clip_action_grad)
        if state is not None:
            state.action_grad_norms.append(gA.detach().reshape(gA.shape[0], -1).norm(dim=-1))
    if gZ is not None:
        gZ = _masked(gZ, future_mask)
        gZ = clip_candidate_grad(gZ, clip_future_grad)
        if state is not None:
            state.future_grad_norms.append(gZ.detach().reshape(gZ.shape[0], -1).norm(dim=-1))

    if joint_update_mode == "simultaneous":
        # Both updates are computed from the *same* (A_k, Z_k).
        A_next = A
        Z_next = Z
        if update_action:
            dA = alpha_action * precondition_candidate_grad(gA, action_mask, candidate_gradient_normalization, gradient_rms_epsilon)
            A_next = A - dA
            if state is not None:
                state.action_update_norms.append(dA.detach().reshape(dA.shape[0], -1).norm(dim=-1))
        if update_future:
            dZ = alpha_future * precondition_candidate_grad(gZ, future_mask, candidate_gradient_normalization, gradient_rms_epsilon)
            Z_next = Z - dZ
            if state is not None:
                state.future_update_norms.append(dZ.detach().reshape(dZ.shape[0], -1).norm(dim=-1))
    else:  # alternating: A uses the pre-update Z, then Z is updated with the new A
        A_next = A
        Z_next = Z
        if update_action:
            dA = alpha_action * precondition_candidate_grad(gA, action_mask, candidate_gradient_normalization, gradient_rms_epsilon)
            A_next = A - dA
            if state is not None:
                state.action_update_norms.append(dA.detach().reshape(dA.shape[0], -1).norm(dim=-1))
        if update_future:
            # NOTE: deliberately a *different* configuration from simultaneous.
            E2 = energy_fn(A_next, Z)
            gZ2 = torch.autograd.grad(
                E2.sum(),
                [Z],
                create_graph=create_graph,
                retain_graph=create_graph,
                allow_unused=False,
            )[0]
            gZ2 = clip_candidate_grad(_masked(gZ2, future_mask), clip_future_grad)
            dZ = alpha_future * precondition_candidate_grad(gZ2, future_mask, candidate_gradient_normalization, gradient_rms_epsilon)
            Z_next = Z - dZ
            if state is not None:
                state.future_grad_norms[-1] if False else None
                state.future_update_norms.append(dZ.detach().reshape(dZ.shape[0], -1).norm(dim=-1))

    if langevin_noise_std and float(langevin_noise_std) > 0:
        # Explicitly NOT part of the reference method; off by default.
        if update_action:
            A_next = A_next + float(langevin_noise_std) * torch.randn_like(A_next) * (create_graph is False)
        if update_future:
            Z_next = Z_next + float(langevin_noise_std) * torch.randn_like(Z_next) * (create_graph is False)

    # Only the branches being updated are constrained.  A pinned ground-truth
    # A*/Z* must not be clamped or masked — that would silently alter the
    # supervision/condition the caller passed in.
    if update_action:
        A_next = apply_candidate_constraints(A_next, action_mask, action_clamp)
    if update_future:
        Z_next = apply_candidate_constraints(Z_next, future_mask, future_clamp)
    return A_next, Z_next
