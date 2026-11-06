"""Batch independence, mask behaviour and episode/未来-frame boundary safety."""

from __future__ import annotations

import torch

from .common import tiny_batch, tiny_cfg, tiny_model
from .harness import TestFailure, assert_close, assert_true


def _energy_fn(m, batch):
    prepared = m._prepare_batch(batch)
    cond = m._condition(batch, prepared)
    a_mask = prepared.get("action_mask")
    z_mask = prepared.get("future_mask")
    offsets = prepared.get("future_offsets")
    fn = lambda a, z: m.core.energy(
        cond, z, a, future_mask=z_mask, action_mask=a_mask, offsets=offsets
    )
    return fn, prepared


def test_batch_independence_energy_and_grad():
    """Sample i computed alone must match sample i computed inside a batch."""
    cfg = tiny_cfg()
    m, cfg = tiny_model(cfg, seed=91)

    full = tiny_batch(cfg, batch_size=3, seed=101)
    fn_full, prep_full = _energy_fn(m, full)
    A = prep_full["action_target"].detach().float().requires_grad_(True)
    Z = prep_full["future_target"].detach().float().requires_grad_(True)
    E_full = fn_full(A, Z)
    gA_full, gZ_full = torch.autograd.grad(E_full.sum(), [A, Z])

    for i in range(3):
        single = {k: (v[i : i + 1] if isinstance(v, torch.Tensor) and v.dim() > 0 else v) for k, v in full.items()}
        fn_s, prep_s = _energy_fn(m, single)
        A_s = prep_s["action_target"].detach().float().requires_grad_(True)
        Z_s = prep_s["future_target"].detach().float().requires_grad_(True)
        E_s = fn_s(A_s, Z_s)
        gA_s, gZ_s = torch.autograd.grad(E_s.sum(), [A_s, Z_s])
        assert_close(float(E_s[0]), float(E_full[i]), 1e-5, f"E sample {i} batch vs single")
        assert_close(gA_s[0], gA_full[i], 1e-5, f"gA sample {i} batch vs single")
        assert_close(gZ_s[0], gZ_full[i], 1e-5, f"gZ sample {i} batch vs single")
        # also check that running the sample in a *different* batch agrees
        other = tiny_batch(cfg, batch_size=4, seed=200 + i)
        other["action_target"] = other["action_target"].clone()
        other["action_target"][1] = full["action_target"][i]
        other["future_target"] = other["future_target"].clone()
        other["future_target"][1] = full["future_target"][i]
        other["action_mask"] = other["action_mask"].clone()
        other["action_mask"][1] = full["action_mask"][i]
        other["future_mask"] = other["future_mask"].clone()
        other["future_mask"][1] = full["future_mask"][i]
        other["cond_vec"] = other["cond_vec"].clone()
        other["cond_vec"][1] = full["cond_vec"][i]
        fn_o, prep_o = _energy_fn(m, other)
        A_o = prep_o["action_target"].detach().float().requires_grad_(True)
        Z_o = prep_o["future_target"].detach().float().requires_grad_(True)
        E_o = fn_o(A_o, Z_o)
        gA_o, gZ_o = torch.autograd.grad(E_o.sum(), [A_o, Z_o])
        assert_close(float(E_o[1]), float(E_full[i]), 1e-5, f"E sample {i} other-batch")
        assert_close(gA_o[1], gA_full[i], 1e-5, f"gA sample {i} other-batch")


def test_masked_target_values_do_not_change_loss():
    """Changing a masked-out target element must not change the loss."""
    cfg = tiny_cfg()
    m, cfg = tiny_model(cfg, seed=92)
    batch = tiny_batch(cfg, batch_size=2, seed=102, include_padding=True)
    prepared = m._prepare_batch(batch)
    cond = m._condition(batch, prepared)
    a_mask = prepared["action_mask"]
    z_mask = prepared["future_mask"]
    offsets = prepared.get("future_offsets")

    A_star = prepared["action_target"].detach().clone().float()
    Z_star = prepared["future_target"].detach().clone().float()

    from energywam.energy.losses import masked_mse

    L1 = masked_mse(A_star, A_star, a_mask) + masked_mse(Z_star, Z_star, z_mask)

    A2 = A_star.clone()
    Z2 = Z_star.clone()
    # perturb masked elements heavily
    def _b(m, ref):
        mm = ~m.bool()
        while mm.dim() < ref.dim():
            mm = mm.unsqueeze(-1)
        return mm.expand_as(ref).float()

    A2 = A2 + _b(a_mask, A_star) * 1000.0
    Z2 = Z2 + _b(z_mask, Z_star) * 1000.0
    L2 = masked_mse(A2, A_star, a_mask) + masked_mse(Z2, Z_star, z_mask)
    assert_close(float(L1), float(L2), 1e-6, "masked target values changed the loss")

    # and the energy must not depend on masked candidate elements
    def E(a, z):
        return m.core.energy(cond, z, a, future_mask=z_mask, action_mask=a_mask, offsets=offsets)

    A_c = A_star.clone().requires_grad_(True)
    Z_c = Z_star.clone().requires_grad_(True)
    e1 = E(A_c, Z_c)
    A_c2 = (A_star + _b(a_mask, A_star) * 50.0).requires_grad_(True)
    Z_c2 = (Z_star + _b(z_mask, Z_star) * 50.0).requires_grad_(True)
    e2 = E(A_c2, Z_c2)
    assert_close(e1, e2, 1e-5, "energy depends on masked candidate elements")


def test_masked_candidates_do_not_move_during_refinement():
    from energywam.energy.solver import refine

    cfg = tiny_cfg()
    m, cfg = tiny_model(cfg, seed=93)
    batch = tiny_batch(cfg, batch_size=2, seed=103, include_padding=True)
    fn, prepared = _energy_fn(m, batch)
    a_mask = prepared["action_mask"]
    z_mask = prepared["future_mask"]
    g = m.geometry
    A0 = torch.zeros(2, g["action_horizon"], g["action_dim"], requires_grad=True)
    Z0 = torch.zeros(
        2, g["num_future_frames"], g["num_cameras"], g["future_patches"], g["future_dim"],
        requires_grad=True,
    )
    A_K, Z_K, _ = refine(
        fn, action=A0, future=Z0, action_mask=a_mask, future_mask=z_mask,
        mode="joint", num_steps=3, alpha_action=0.3, alpha_future=0.3, training=False,
    )
    inv_a = ~a_mask.bool()
    inv_z = ~z_mask.bool()
    assert_true(float(A_K[inv_a].abs().max()) == 0.0, "masked action elements moved during refinement")
    assert_true(float(Z_K[inv_z].abs().max()) == 0.0, "masked future elements moved during refinement")


def test_future_offsets_valid_monotonic_and_bounded():
    from energywam.config import resolve_future_offsets

    cfg = tiny_cfg()
    offs = resolve_future_offsets(cfg.data)
    assert_true(len(offs) == 3, f"expected 3 offsets, got {offs}")
    assert_true(all(o > 0 for o in offs), f"offsets must be positive: {offs}")
    assert_true(list(offs) == sorted(set(offs)), f"offsets must be strictly increasing: {offs}")
    assert_true(max(offs) <= cfg.data.action_horizon, f"offsets exceed horizon: {offs} vs {cfg.data.action_horizon}")


def test_episode_boundary_rejects_out_of_range_offsets():
    """The data-side guard must refuse future frames that leave the episode."""
    from energywam.data.future_offsets import plan_future_frames

    L = 12  # episode length
    H = 8
    offsets = [3, 5, 8]
    ok = plan_future_frames(start_index=2, episode_length=L, action_horizon=H, future_offsets=offsets)
    assert_true(ok.valid, "start=2 with L=12, H=8, offsets=[3,5,8] should be valid")
    assert_true(max(ok.abs_frames) < L, f"future frame left the episode: {ok.as_dict()}")

    bad = plan_future_frames(start_index=6, episode_length=L, action_horizon=H, future_offsets=offsets)
    assert_true(not bad.valid, "start=6 must be invalid (tail crosses the episode end)")

    bad2 = plan_future_frames(start_index=4, episode_length=L, action_horizon=H, future_offsets=offsets)
    assert_true(not bad2.valid, "start=4 leaves the action horizon inside the episode but the future tail crosses")


TESTS = [
    ("batch_independence_energy_and_grad", test_batch_independence_energy_and_grad),
    ("masked_target_values_do_not_change_loss", test_masked_target_values_do_not_change_loss),
    ("masked_candidates_do_not_move_during_refinement", test_masked_candidates_do_not_move_during_refinement),
    ("future_offsets_valid_monotonic_and_bounded", test_future_offsets_valid_monotonic_and_bounded),
    ("episode_boundary_rejects_out_of_range_offsets", test_episode_boundary_rejects_out_of_range_offsets),
]
