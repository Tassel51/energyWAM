"""Finite-difference gradient checks in float64.

Covers
------
1. ``d E.sum() / d A`` and ``d E.sum() / d Z`` (candidate gradients)
2. the **outer** loss gradient w.r.t. network parameters *through the inner
   unroll* (the second-order path the full unroll exists for)

Method: central differences `` (f(x+eps) - f(x-eps)) / (2 eps) `` with
``eps = 1e-6`` in float64.  Reported tolerance is an absolute tolerance on the
gradient value; the maximum observed absolute and relative deviations are
printed so the run log records the actual numbers.
"""

from __future__ import annotations

import torch

from .common import tiny_batch, tiny_cfg, tiny_model
from .harness import TestFailure, assert_true

EPS = 1e-6
TOL_CAND = 1e-5
TOL_PARAM = 2e-4


def _fd_scalar(f, x, idx, eps=EPS):
    orig = x[idx].item()
    with torch.no_grad():
        x[idx] = orig + eps
    fp = f().item()
    with torch.no_grad():
        x[idx] = orig - eps
    fm = f().item()
    with torch.no_grad():
        x[idx] = orig
    return (fp - fm) / (2 * eps)


def _double_model(seed=0):
    cfg = tiny_cfg(**{"trainer.candidate_dtype": "float64"})
    m, cfg = tiny_model(cfg, seed=seed)
    m = m.double()
    return m, cfg


def test_fd_candidate_gradients():
    m, cfg = _double_model(seed=31)
    batch = tiny_batch(cfg, batch_size=2, seed=41)
    prepared = m._prepare_batch(batch)
    cond = m._condition(batch, prepared)
    a_mask = prepared.get("action_mask")
    z_mask = prepared.get("future_mask")
    offsets = prepared.get("future_offsets")

    A = prepared["action_target"].detach().clone().double().requires_grad_(True)
    Z = prepared["future_target"].detach().clone().double().requires_grad_(True)

    def energy():
        return m.core.energy(cond, Z, A, future_mask=z_mask, action_mask=a_mask, offsets=offsets)

    E = energy()
    gA, gZ = torch.autograd.grad(E.sum(), [A, Z])

    max_dev = 0.0
    # A: check a handful of interior elements
    for idx in [(0, 0, 0), (0, 3, 1), (1, 5, 2), (1, 7, 0)]:
        if a_mask is not None and not bool(a_mask[idx]):
            continue
        fd = _fd_scalar(lambda: energy().sum(), A, idx)
        ad = gA[idx].item()
        dev = abs(fd - ad)
        max_dev = max(max_dev, dev)
        if dev > TOL_CAND:
            raise TestFailure(f"dE/dA{idx}: fd={fd:.8e} ad={ad:.8e} dev={dev:.3e}")
    # Z
    for idx in [(0, 0, 0, 0, 0), (0, 2, 0, 3, 5), (1, 1, 0, 1, 7)]:
        if z_mask is not None:
            if not bool(z_mask[idx[:4]]):
                continue
        fd = _fd_scalar(lambda: energy().sum(), Z, idx)
        ad = gZ[idx].item()
        dev = abs(fd - ad)
        max_dev = max(max_dev, dev)
        if dev > TOL_CAND:
            raise TestFailure(f"dE/dZ{idx}: fd={fd:.8e} ad={ad:.8e} dev={dev:.3e}")
    print(f"  [fd candidates] max abs dev = {max_dev:.3e} (tol {TOL_CAND:.1e})")


def test_fd_outer_through_inner_parameters():
    """K=2 full unroll: compare autograd vs finite differences for the outer loss
    w.r.t. energy-head and projector parameters."""
    m, cfg = _double_model(seed=32)
    cfg.solver.train_inner_steps = 2
    batch = tiny_batch(cfg, batch_size=2, seed=42)
    prepared = m._prepare_batch(batch)
    cond = m._condition(batch, prepared)
    a_mask = prepared.get("action_mask")
    z_mask = prepared.get("future_mask")
    offsets = prepared.get("future_offsets")
    A_star = prepared["action_target"].detach().clone().double()
    Z_star = prepared["future_target"].detach().clone().double()

    from energywam.energy.losses import masked_mse
    from energywam.energy.solver import refine

    def outer_loss():
        A0 = torch.zeros_like(A_star, requires_grad=True)
        Z0 = torch.zeros_like(Z_star, requires_grad=True)
        energy_fn = lambda a, z: m.core.energy(
            cond, z, a, future_mask=z_mask, action_mask=a_mask, offsets=offsets
        )
        A_K, Z_K, _ = refine(
            energy_fn, action=A0, future=Z0, action_mask=a_mask, future_mask=z_mask,
            mode="joint", num_steps=2, alpha_action=0.05, alpha_future=0.05, training=True,
        )
        return masked_mse(A_K, A_star, a_mask)

    loss = outer_loss()
    loss.backward()

    targets = [
        ("core.energy_head.net.4.weight", (0, 0)),
        ("core.energy_head.net.1.bias", (0,)),
        ("core.action_projector.0.weight", (0, 0)),
        ("core.future_projector.0.weight", (1, 2)),
        ("backbone.transformer.blocks.0.attn.qkv.weight", (0, 1)),
    ]
    max_dev = 0.0
    for name, idx in targets:
        p = dict(m.named_parameters())[name]
        if p.grad is None:
            raise TestFailure(f"no outer gradient for {name} through the unroll")
        ad = p.grad[idx].item()
        fd = _fd_scalar(outer_loss, p, idx)
        dev = abs(fd - ad)
        rel = dev / max(abs(fd), abs(ad), 1e-12)
        max_dev = max(max_dev, dev)
        print(f"  [fd param] {name}{idx}: fd={fd:.8e} ad={ad:.8e} abs_dev={dev:.3e} rel={rel:.3e}")
        if dev > TOL_PARAM:
            raise TestFailure(f"{name}{idx}: fd={fd:.8e} ad={ad:.8e} dev={dev:.3e} > {TOL_PARAM}")
    print(f"  [fd outer params] max abs dev = {max_dev:.3e} (tol {TOL_PARAM:.1e})")

    # The energy head's FINAL-layer bias is a pure gauge freedom: E = W4 h + b4
    # shifts the energy by a constant without changing its gradient field, and
    # the refinement only ever uses grad(E).  Its outer gradient must therefore
    # be identically zero / absent — this is a correctness property, not a bug.
    bias = dict(m.named_parameters())["core.energy_head.net.4.bias"]
    if bias.grad is not None and float(bias.grad.abs().max()) > 1e-12:
        raise TestFailure("energy-head final bias should be a gauge freedom (zero outer gradient)")
    fd_bias = _fd_scalar(outer_loss, bias, (0,))
    if abs(fd_bias) > 1e-8:
        raise TestFailure(f"finite-difference says the gauge bias is NOT free: fd={fd_bias}")
    print("  [gauge] energy-head final bias: autograd=0, fd=%.3e (gauge freedom confirmed)" % fd_bias)


def test_fd_solver_analytic_matches_autograd():
    """A pure quadratic energy has an exactly known gradient; verify autograd."""
    B = 2
    A = torch.randn(B, 3, 2, dtype=torch.float64, requires_grad=True)
    Z = torch.randn(B, 2, 1, 4, 2, dtype=torch.float64, requires_grad=True)
    a0 = torch.randn_like(A)
    z0 = torch.randn_like(Z)
    coupling = 0.3

    def sums(t):
        return t.reshape(B, -1).sum(dim=1)

    def E_fn(a, z):
        quad = 0.5 * ((a - a0) ** 2).reshape(B, -1).sum(dim=1) + 0.25 * ((z - z0) ** 2).reshape(B, -1).sum(dim=1)
        cross = coupling * (sums(a) * sums(z))
        return quad + cross

    E = E_fn(A, Z)
    gA, gZ = torch.autograd.grad(E.sum(), [A, Z])
    gA_ref = (A - a0) + coupling * sums(Z).view(B, 1, 1).expand_as(A)
    gZ_ref = 0.5 * (Z - z0) + coupling * sums(A).view(B, 1, 1, 1, 1).expand_as(Z)
    if float((gA - gA_ref).abs().max()) > 1e-12:
        raise TestFailure(f"analytic gA mismatch {(gA - gA_ref).abs().max().item()}")
    if float((gZ - gZ_ref).abs().max()) > 1e-12:
        raise TestFailure(f"analytic gZ mismatch {(gZ - gZ_ref).abs().max().item()}")
    for idx in [(0, 1, 0), (1, 2, 1)]:
        fd = _fd_scalar(lambda: E_fn(A, Z).sum(), A, idx)
        if abs(fd - gA[idx].item()) > 1e-8:
            raise TestFailure(f"fd/analytic mismatch at A{idx}: {fd} vs {gA[idx].item()}")
    for idx in [(0, 0, 0, 2, 1), (1, 1, 0, 0, 0)]:
        fd = _fd_scalar(lambda: E_fn(A, Z).sum(), Z, idx)
        if abs(fd - gZ[idx].item()) > 1e-8:
            raise TestFailure(f"fd/analytic mismatch at Z{idx}: {fd} vs {gZ[idx].item()}")


TESTS = [
    ("fd_candidate_gradients", test_fd_candidate_gradients),
    ("fd_outer_through_inner_parameters", test_fd_outer_through_inner_parameters),
    ("fd_solver_analytic_matches_autograd", test_fd_solver_analytic_matches_autograd),
]
