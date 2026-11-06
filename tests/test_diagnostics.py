"""Offline coupling diagnostics (paper §4.9 / §6.4.2–6.4.3).

Verifies the *metrics themselves* against closed-form values and enforces the
honesty contract:

* every report carries the math-only interpretation label;
* the ranking metric refuses to run without caller-supplied, validity-flagged
  negatives and never invents any;
* the future metrics report their coordinate system (``space``).

These are diagnostics — they are not added to any training objective.
"""

from __future__ import annotations

import torch

from .common import tiny_batch, tiny_cfg, tiny_model
from .harness import TestFailure, assert_close, assert_true
from energywam.energy.diagnostics import (
    INTERPRETATION,
    action_gradient_future_sensitivity,
    condition_swap_action_trajectory,
    energy_ranking_metrics,
    future_gradient_action_sensitivity,
    future_latent_metrics,
)


# ---------------------------------------------------------------------------
# Eq. 44 against a closed form
# ---------------------------------------------------------------------------
def test_eq44_sensitivity_matches_closed_form():
    """E(A,Z) = sum(A * Z) per sample  =>  grad_A E = Z, so
    S_{A<-Z} = ||Z1 - Z2|| / ||Z1|| exactly (per sample, mean-reduced)."""
    B = 3
    A = torch.randn(B, 2, 2)
    Z1 = torch.randn(B, 1, 1, 2, 2)
    Z2 = torch.randn(B, 1, 1, 2, 2)

    def energy_fn(a, z):
        return (a.reshape(a.shape[0], -1) * z.reshape(z.shape[0], -1)).sum(dim=-1)

    out = action_gradient_future_sensitivity(energy_fn, A, Z1, Z2, eps=0.0)
    z1 = Z1.reshape(B, -1)
    z2 = Z2.reshape(B, -1)
    expected = (z1 - z2).norm(dim=-1) / z1.norm(dim=-1)
    assert_close(out["sensitivity_per_sample"], expected, 1e-5,
                 "Eq. 44 does not match the closed form")
    assert_close(out["sensitivity_mean"], expected.mean(), 1e-5, "mean reduction")
    assert_true(out["interpretation"] == INTERPRETATION, "interpretation label missing")
    assert_true(out["metric"] == "S_A_from_Z (Eq. 44)", f"metric label {out['metric']}")


def test_eq44_separable_energy_gives_zero_sensitivity():
    """E(A,Z) = f(A) + g(Z) (paper Eq. 42) must give EXACTLY zero cross
    sensitivity — the degenerate case the diagnostic exists to detect."""
    B = 2
    A = torch.randn(B, 2, 2)
    Z1 = torch.randn(B, 1, 1, 2, 2)
    Z2 = torch.randn(B, 1, 1, 2, 2) + 3.0

    def separable(a, z):
        return a.reshape(a.shape[0], -1).pow(2).sum(-1) + z.reshape(z.shape[0], -1).pow(2).sum(-1)

    out = action_gradient_future_sensitivity(separable, A, Z1, Z2, eps=1e-12)
    assert_close(out["sensitivity_mean"], 0.0, 1e-9,
                 "separable energy produced non-zero S_A<-Z")
    out_z = future_gradient_action_sensitivity(separable, Z1, A, A + 3.0, eps=1e-12)
    assert_close(out_z["sensitivity_mean"], 0.0, 1e-9,
                 "separable energy produced non-zero S_Z<-A")
    assert_true(out_z["metric"].startswith("S_Z_from_A"), out_z["metric"])


def test_symmetric_direction_matches_closed_form():
    """E(A,Z) = sum(A * Z) => grad_Z E = A, S_{Z<-A} = ||A1 - A2|| / ||A1||."""
    B = 4
    Z = torch.randn(B, 1, 1, 2, 2)
    A1 = torch.randn(B, 2, 2)
    A2 = torch.randn(B, 2, 2)

    def energy_fn(a, z):
        return (a.reshape(a.shape[0], -1) * z.reshape(z.shape[0], -1)).sum(dim=-1)

    out = future_gradient_action_sensitivity(energy_fn, Z, A1, A2, eps=0.0)
    a1 = A1.reshape(B, -1)
    a2 = A2.reshape(B, -1)
    expected = (a1 - a2).norm(dim=-1) / a1.norm(dim=-1)
    assert_close(out["sensitivity_per_sample"], expected, 1e-5, "symmetric Eq. 44")


# ---------------------------------------------------------------------------
# condition swap
# ---------------------------------------------------------------------------
def test_condition_swap_trajectory_distance():
    """Two different conditions must diverge the action trajectory; the same
    condition twice must give exactly zero distance at every step."""

    def make_energy(scale: float):
        def fn(a, z):
            return scale * a.reshape(a.shape[0], -1).pow(2).sum(-1)
        return fn

    A0 = torch.ones(2, 3, 2)
    Z = torch.zeros(2, 1, 1, 2, 2)
    out = condition_swap_action_trajectory(
        make_energy(1.0), make_energy(2.0), A0, Z,
        num_steps=3, alpha_action=0.1,
    )
    per_step = out["per_step_l2_mean"]
    assert_true(per_step.shape[0] == 4,
                f"expected num_steps+1 trajectory points, got {per_step.shape}")
    assert_true(float(per_step[-1]) > 0.0, "different conditions left the trajectory identical")
    assert_true(out["interpretation"] == INTERPRETATION, "interpretation label missing")

    same = condition_swap_action_trajectory(
        make_energy(1.0), make_energy(1.0), A0, Z,
        num_steps=3, alpha_action=0.1,
    )
    assert_close(same["per_step_l2_mean"], torch.zeros(4), 1e-7,
                 "identical conditions must give zero trajectory distance")


# ---------------------------------------------------------------------------
# Eq. 45 ranking — hand-computed values + negative policy
# ---------------------------------------------------------------------------
def test_ranking_metrics_hand_computed():
    e_m = torch.tensor([0.0, 0.0])
    e_x = torch.tensor([1.0, 2.0])
    valid = torch.tensor([True, True])
    out = energy_ranking_metrics(e_m, e_x, valid)
    assert_close(out["ranking_accuracy"], 1.0, 1e-6, "both pairs should win")
    assert_close(out["auc"], 1.0, 1e-6, "AUC")
    assert_close(out["margin_mean"], 1.5, 1e-6, "margin (mis - match)")
    assert_close(out["num_pairs_used"], 2.0, 0.0, "pair count")

    # one win one loss -> accuracy 0.5, AUC 0.5
    out2 = energy_ranking_metrics(torch.tensor([0.0, 0.0]), torch.tensor([1.0, -1.0]), valid)
    assert_close(out2["ranking_accuracy"], 0.5, 1e-6, "accuracy")
    assert_close(out2["auc"], 0.5, 1e-6, "AUC")

    # ties get half credit
    out3 = energy_ranking_metrics(torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([True]))
    assert_close(out3["auc"], 0.5, 1e-6, "tie AUC")
    assert_close(out3["ranking_accuracy"], 0.0, 1e-6, "tie is not a win")

    assert_true(out["interpretation"] == INTERPRETATION, "interpretation label missing")
    assert_true("negative" in out["negative_policy"].lower(), "negative policy missing")


def test_ranking_validity_flags_are_enforced():
    e_m = torch.tensor([0.0, 0.0, 0.0])
    e_x = torch.tensor([1.0, -5.0, 2.0])
    valid = torch.tensor([True, False, True])  # the middle negative is NOT verified
    out = energy_ranking_metrics(e_m, e_x, valid)
    assert_close(out["num_pairs_used"], 2.0, 0.0, "invalid negative must be excluded")
    assert_close(out["ranking_accuracy"], 1.0, 1e-6,
                 "only the verified negatives may count")

    # 2-D negatives with per-pair validity
    e_x2 = torch.tensor([[1.0, -9.0], [1.5, 1.0], [2.0, -9.0]])
    v2 = torch.tensor([[True, False], [True, True], [True, False]])
    out2 = energy_ranking_metrics(torch.tensor([0.0, 0.0, 0.0]), e_x2, v2)
    assert_close(out2["num_pairs_used"], 4.0, 0.0, "2-D validity mask")
    assert_close(out2["num_negatives_provided"], 6.0, 0.0, "provided count must not shrink")
    assert_close(out2["ranking_accuracy"], 1.0, 1e-6, "verified pairs all win")


def test_ranking_refuses_unsupplied_or_unflagged_negatives():
    e_m = torch.tensor([0.0, 0.0])
    try:
        energy_ranking_metrics(e_m, torch.zeros(0), torch.zeros(0, dtype=torch.bool))
    except ValueError as exc:
        assert_true("caller" in str(exc).lower() or "mismatched" in str(exc).lower(), str(exc))
    else:
        raise TestFailure("ranking ran without any negative pairs")
    try:
        energy_ranking_metrics(e_m, torch.tensor([1.0, 1.0]), None)
    except ValueError as exc:
        assert_true("valid" in str(exc).lower(), str(exc))
    else:
        raise TestFailure("ranking ran without negatives_valid")
    try:
        energy_ranking_metrics(e_m, torch.tensor([1.0, 1.0]),
                               torch.tensor([False, False]))
    except ValueError as exc:
        assert_true("valid" in str(exc).lower(), str(exc))
    else:
        raise TestFailure("ranking ran with zero valid negatives")


# ---------------------------------------------------------------------------
# §6.4.2 future metrics
# ---------------------------------------------------------------------------
def test_future_latent_metrics_hand_computed():
    Zt = torch.zeros(1, 1, 1, 2, 2)
    Zp = torch.zeros(1, 1, 1, 2, 2)
    Zp[0, 0, 0, 0] = torch.tensor([3.0, 4.0])  # |err| = 5 -> err^2 = 25
    mask = torch.ones(1, 1, 1, 2, dtype=torch.bool)
    out = future_latent_metrics(Zp, Zt, mask, space="raw_dino")
    # 25 over 4 valid elements
    assert_close(out["mse"], 6.25, 1e-5, "Eq. 53 normalisation")
    assert_true(out["space"] == "raw_dino", "space label not propagated")
    assert_true(out["interpretation"] == INTERPRETATION, "interpretation label missing")

    # perfectly aligned direction -> cosine similarity 1
    Zp2 = Zt + 1.0
    out2 = future_latent_metrics(Zp2, Zt + 2.0, mask, space="normalized_Z")
    assert_close(out2["cosine_similarity"], 1.0, 1e-5, "aligned patch directions")

    try:
        future_latent_metrics(Zp, Zt, mask, space="")
    except ValueError as exc:
        assert_true("space" in str(exc), str(exc))
    else:
        raise TestFailure("future metrics ran without declaring the coordinate system")


# ---------------------------------------------------------------------------
# end-to-end on the tiny model
# ---------------------------------------------------------------------------
def _tiny_energy():
    cfg = tiny_cfg(**{"solver.train_inner_steps": 2})
    m, cfg = tiny_model(cfg, seed=77)
    batch = tiny_batch(cfg, batch_size=2, seed=78)
    prepared = m._prepare_batch(batch)
    cond = m._condition(batch, prepared)
    a_mask = prepared.get("action_mask")
    z_mask = prepared.get("future_mask")
    offsets = prepared.get("future_offsets")

    def energy_fn(a, z):
        return m.core.energy(cond, z, a, future_mask=z_mask, action_mask=a_mask, offsets=offsets)

    return m, cfg, prepared, energy_fn


def test_diagnostics_end_to_end_on_tiny_model():
    m, cfg, prepared, energy_fn = _tiny_energy()
    A = prepared["action_target"].detach().float()
    Z1 = prepared["future_target"].detach().float()
    Z2 = Z1 + 0.75

    s_az = action_gradient_future_sensitivity(energy_fn, A, Z1, Z2)
    assert_true(torch.isfinite(s_az["sensitivity_mean"]).all(), "S_A<-Z not finite")
    assert_true(float(s_az["sensitivity_mean"]) >= 0.0, "negative sensitivity")

    s_za = future_gradient_action_sensitivity(energy_fn, Z1, A, A + 0.75)
    assert_true(torch.isfinite(s_za["sensitivity_mean"]).all(), "S_Z<-A not finite")

    swap = condition_swap_action_trajectory(
        lambda a, z: energy_fn(a, z),
        lambda a, z: energy_fn(a, z + 1.0),
        A, Z1, num_steps=2, alpha_action=0.05,
    )
    assert_true(torch.isfinite(swap["per_step_l2_mean"]).all(), "swap distances not finite")

    # ranking on caller-supplied pairs (matched vs explicitly provided negatives)
    e_match = energy_fn(A, Z1)
    e_mis = energy_fn(A, Z2)  # caller-declared alternative futures
    valid = torch.ones_like(e_mis, dtype=torch.bool)
    rank = energy_ranking_metrics(e_match, e_mis, valid)
    assert_true(0.0 <= float(rank["auc"]) <= 1.0, f"AUC out of range: {rank['auc']}")
    assert_true(rank["interpretation"] == INTERPRETATION, "interpretation label missing")

    fut = future_latent_metrics(Z2, Z1, prepared.get("future_mask"), space="raw_dino")
    assert_true(float(fut["mse"]) > 0.0, "MSE of a perturbed latent must be positive")


TESTS = [
    ("eq44_sensitivity_matches_closed_form", test_eq44_sensitivity_matches_closed_form),
    ("eq44_separable_energy_gives_zero_sensitivity", test_eq44_separable_energy_gives_zero_sensitivity),
    ("symmetric_direction_matches_closed_form", test_symmetric_direction_matches_closed_form),
    ("condition_swap_trajectory_distance", test_condition_swap_trajectory_distance),
    ("ranking_metrics_hand_computed", test_ranking_metrics_hand_computed),
    ("ranking_validity_flags_are_enforced", test_ranking_validity_flags_are_enforced),
    ("ranking_refuses_unsupplied_or_unflagged_negatives",
     test_ranking_refuses_unsupplied_or_unflagged_negatives),
    ("future_latent_metrics_hand_computed", test_future_latent_metrics_hand_computed),
    ("diagnostics_end_to_end_on_tiny_model", test_diagnostics_end_to_end_on_tiny_model),
]
