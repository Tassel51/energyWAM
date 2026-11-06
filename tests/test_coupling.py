"""Bidirectional influence: A and Z must actually couple through the energy.

A shared scalar does not automatically produce coupling — if the network
degenerates to ``E(C,A) + E(C,Z)`` the cross-gradients vanish.  These tests
check the cross-partial on a non-degenerate randomly-initialised tiny model.

They do **not** prove the model has learned physical regularities.
"""

from __future__ import annotations

import torch

from .common import tiny_batch, tiny_cfg, tiny_model
from .harness import TestFailure, assert_true


def _energy(m, batch):
    prepared = m._prepare_batch(batch)
    cond = m._condition(batch, prepared)
    a_mask = prepared.get("action_mask")
    z_mask = prepared.get("future_mask")
    offsets = prepared.get("future_offsets")

    def E(a, z):
        return m.core.energy(cond, z, a, future_mask=z_mask, action_mask=a_mask, offsets=offsets)

    return E, prepared


def test_action_grad_changes_with_future():
    """Fix C and A, change Z -> the action gradient must change."""
    cfg = tiny_cfg()
    m, cfg = tiny_model(cfg, seed=71)
    batch = tiny_batch(cfg, batch_size=2, seed=81)
    E, prepared = _energy(m, batch)
    a_mask = prepared.get("action_mask")
    z_mask = prepared.get("future_mask")

    A = prepared["action_target"].detach().float().requires_grad_(True)
    Z1 = prepared["future_target"].detach().float()
    Z2 = Z1 + 1.5

    gA1 = torch.autograd.grad(E(A, Z1).sum(), [A], retain_graph=False)[0]
    A2 = A.detach().clone().requires_grad_(True)
    gA2 = torch.autograd.grad(E(A2, Z2).sum(), [A2])[0]
    diff = float((gA1 - gA2).abs().max())
    print(f"  max |gA(Z1) - gA(Z2)| = {diff:.6e}")
    if diff < 1e-8:
        raise TestFailure("action gradient is independent of the future candidate — no A<-Z coupling")


def test_future_grad_changes_with_action():
    """Fix C and Z, change A -> the future gradient must change."""
    cfg = tiny_cfg()
    m, cfg = tiny_model(cfg, seed=72)
    batch = tiny_batch(cfg, batch_size=2, seed=82)
    E, prepared = _energy(m, batch)

    Z = prepared["future_target"].detach().float().requires_grad_(True)
    A1 = prepared["action_target"].detach().float()
    A2 = A1 + 1.5

    gZ1 = torch.autograd.grad(E(A1, Z).sum(), [Z])[0]
    Z2 = Z.detach().clone().requires_grad_(True)
    gZ2 = torch.autograd.grad(E(A2, Z2).sum(), [Z2])[0]
    diff = float((gZ1 - gZ2).abs().max())
    print(f"  max |gZ(A1) - gZ(A2)| = {diff:.6e}")
    if diff < 1e-8:
        raise TestFailure("future gradient is independent of the action candidate — no Z<-A coupling")


def test_joint_unroll_endpoints_respond_to_other_candidate():
    """After a joint unroll, changing only the initial Z changes the final A."""
    from energywam.energy.solver import refine

    cfg = tiny_cfg(**{"solver.train_inner_steps": 3})
    m, cfg = tiny_model(cfg, seed=73)
    batch = tiny_batch(cfg, batch_size=2, seed=83)
    prepared = m._prepare_batch(batch)
    cond = m._condition(batch, prepared)
    a_mask = prepared.get("action_mask")
    z_mask = prepared.get("future_mask")
    offsets = prepared.get("future_offsets")
    energy_fn = lambda a, z: m.core.energy(
        cond, z, a, future_mask=z_mask, action_mask=a_mask, offsets=offsets
    )

    g = m.geometry
    A0 = torch.zeros(2, g["action_horizon"], g["action_dim"], requires_grad=True)
    Z0 = torch.zeros(
        2, g["num_future_frames"], g["num_cameras"], g["future_patches"], g["future_dim"],
        requires_grad=True,
    )
    A_K1, Z_K1, _ = refine(
        energy_fn, action=A0, future=Z0, action_mask=a_mask, future_mask=z_mask,
        mode="joint", num_steps=3, alpha_action=0.1, alpha_future=0.1, training=False,
    )
    A0b = torch.zeros_like(A0, requires_grad=True)
    Z0b = (Z0.detach() + 2.0).requires_grad_(True)
    A_K2, Z_K2, _ = refine(
        energy_fn, action=A0b, future=Z0b, action_mask=a_mask, future_mask=z_mask,
        mode="joint", num_steps=3, alpha_action=0.1, alpha_future=0.1, training=False,
    )
    dA = float((A_K1 - A_K2).abs().max())
    dZ = float((Z_K1 - Z_K2).abs().max())
    print(f"  joint unroll: |dA_K| = {dA:.6e}, |dZ_K| = {dZ:.6e}")
    if dA < 1e-8:
        raise TestFailure("joint unroll: final A is insensitive to the initial Z")
    if dZ < 1e-8:
        raise TestFailure("joint unroll: final Z is insensitive to the initial A (A0 identical)")


TESTS = [
    ("action_grad_changes_with_future", test_action_grad_changes_with_future),
    ("future_grad_changes_with_action", test_future_grad_changes_with_action),
    ("joint_unroll_endpoints_respond_to_other_candidate", test_joint_unroll_endpoints_respond_to_other_candidate),
]
