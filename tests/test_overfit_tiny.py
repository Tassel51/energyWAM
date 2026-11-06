"""Tiny-mode overfit: the endpoint error must drop clearly in a few dozen steps.

This is a sanity check that the full unroll actually optimises the endpoint
objective end-to-end.  It is NOT a claim about real-data performance.

Why a reduced future geometry here
----------------------------------
The overfit probe uses ``tiny_future_patches=2, tiny_future_dim=4`` so the future
latent is 3*1*2*4 = 24 dims, the same size as the action chunk (8*3 = 24).  With
the default 4x8 geometry the future target is 96 dims and a 150-step CPU run
only moves the error by ~20%, which is too weak to demonstrate the optimisation
clearly.  Everything else (solver, losses, stage logic, backprop through the
unroll) is identical — only the tensor size changes.

Stability note: ``alpha*K`` around 2.4 (alpha=0.3, K=8) converges; alpha=0.4,
K=10 diverges on this toy.  That is exactly the kind of scale check the
engineering requirements ask for when choosing K / alpha.
"""

from __future__ import annotations

import torch

from .common import tiny_batch, tiny_cfg, tiny_model
from .harness import TestFailure, assert_true

ALPHA = 0.3
K = 8
LR = 3e-3
STEPS = 300


def _overfit_cfg(stage: str):
    cfg = tiny_cfg(
        **{
            "stage.stage": stage,
            "trainer.learning_rate": LR,
            "backbone.tiny_future_patches": 2,
            "backbone.tiny_future_dim": 4,
            "solver.alpha_action": ALPHA,
            "solver.alpha_future": ALPHA,
            "solver.train_inner_steps": K,
            "solver.eval_inner_steps": K,
        }
    )
    return cfg


def _endpoint_error(model, cfg, batch, mode="joint"):
    """Endpoint MSE after refinement.

    ``mode`` selects the solver problem the model is actually trained on:
    ``joint`` (both candidates free) or a single-direction conditional solve.
    For ``bi`` we evaluate both conditional directions separately, because that
    is what the stage optimises — scoring a ``joint`` unroll after ``bi``-only
    training would be measuring a different problem.
    """
    prepared = model._prepare_batch(batch)
    cond = model._condition(batch, prepared)
    from energywam.energy.losses import masked_mse
    from energywam.energy.solver import refine

    a_mask = prepared.get("action_mask")
    z_mask = prepared.get("future_mask")
    offsets = prepared.get("future_offsets")
    A_star = prepared["action_target"]
    Z_star = prepared["future_target"]
    kwargs = dict(
        num_steps=cfg.solver.eval_inner_steps,
        alpha_action=cfg.solver.alpha_action,
        alpha_future=cfg.solver.alpha_future,
        training=False,
    )
    ef = lambda a, z: model.core.energy(cond, z, a, future_mask=z_mask, action_mask=a_mask, offsets=offsets)

    def joint():
        A0 = torch.zeros_like(A_star, requires_grad=True)
        Z0 = torch.zeros_like(Z_star, requires_grad=True)
        A_K, Z_K, _ = refine(ef, action=A0, future=Z0, action_mask=a_mask,
                             future_mask=z_mask, mode="joint", **kwargs)
        return float(masked_mse(A_K, A_star, a_mask)), float(masked_mse(Z_K, Z_star, z_mask))

    def action_dir():
        A0 = torch.zeros_like(A_star, requires_grad=True)
        A_K, _, _ = refine(ef, action=A0, future=Z_star.detach().float(),
                           action_mask=a_mask, future_mask=z_mask,
                           mode="action_given_future", **kwargs)
        return float(masked_mse(A_K, A_star, a_mask)), float("nan")

    def future_dir():
        Z0 = torch.zeros_like(Z_star, requires_grad=True)
        _, Z_K, _ = refine(ef, action=A_star.detach().float(), future=Z0,
                           action_mask=a_mask, future_mask=z_mask,
                           mode="future_given_action", **kwargs)
        return float("nan"), float(masked_mse(Z_K, Z_star, z_mask))

    if mode == "joint":
        return joint()
    if mode == "action_given_future":
        return action_dir()
    if mode == "future_given_action":
        return future_dir()
    if mode == "bi":
        a1, _ = action_dir()
        _, z1 = future_dir()
        return a1, z1
    raise ValueError(f"unknown mode {mode!r}")


def _run(model, cfg, data, probe, mode="joint"):
    a0, z0 = _endpoint_error(model, cfg, probe, mode=mode)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.trainer.learning_rate)
    history = []
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        out = model(data[step % len(data)])
        out["action_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            model.eval()
            history.append((step,) + _endpoint_error(model, cfg, probe, mode=mode))
            model.train()
    model.eval()
    a1, z1 = _endpoint_error(model, cfg, probe, mode=mode)
    return (a0, z0), (a1, z1), history


def test_tiny_overfit_bi_endpoint_error_drops():
    cfg = _overfit_cfg("bi")
    m, cfg = tiny_model(cfg, seed=301)
    data = [tiny_batch(cfg, batch_size=4, seed=400)]
    (a0, z0), (a1, z1), hist = _run(m, cfg, data, data[0], mode="bi")
    print(f"  bi overfit: action {a0:.4f} -> {a1:.4f}, future {z0:.4f} -> {z1:.4f}")
    print(f"  history: {hist}")
    assert_true(a1 < a0 * 0.5, f"action endpoint error did not drop clearly: {a0:.4f} -> {a1:.4f}")
    assert_true(z1 < z0 * 0.5, f"future endpoint error did not drop clearly: {z0:.4f} -> {z1:.4f}")


def test_tiny_overfit_joint_endpoint_error_drops():
    cfg = _overfit_cfg("joint")
    m, cfg = tiny_model(cfg, seed=302)
    data = [tiny_batch(cfg, batch_size=4, seed=500)]
    (a0, z0), (a1, z1), hist = _run(m, cfg, data, data[0], mode="joint")
    print(f"  joint overfit: action {a0:.4f} -> {a1:.4f}, future {z0:.4f} -> {z1:.4f}")
    print(f"  history: {hist}")
    assert_true(a1 < a0 * 0.5, f"joint action error did not drop clearly: {a0:.4f} -> {a1:.4f}")
    assert_true(z1 < z0 * 0.5, f"joint future error did not drop clearly: {z0:.4f} -> {z1:.4f}")


def test_tiny_overfit_action_only_endpoint_error_drops():
    cfg = _overfit_cfg("action_only")
    m, cfg = tiny_model(cfg, seed=303)
    data = [tiny_batch(cfg, batch_size=4, seed=600)]
    (a0, _), (a1, _), hist = _run(m, cfg, data, data[0], mode="action_given_future")
    print(f"  action_only overfit: action {a0:.4f} -> {a1:.4f}")
    print(f"  history: {hist}")
    assert_true(a1 < a0 * 0.5, f"action_only error did not drop clearly: {a0:.4f} -> {a1:.4f}")


TESTS = [
    ("tiny_overfit_bi_endpoint_error_drops", test_tiny_overfit_bi_endpoint_error_drops),
    ("tiny_overfit_joint_endpoint_error_drops", test_tiny_overfit_joint_endpoint_error_drops),
    ("tiny_overfit_action_only_endpoint_error_drops", test_tiny_overfit_action_only_endpoint_error_drops),
]
