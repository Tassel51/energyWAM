"""Analytic toy solver validation.

The toy energy is a *coupled quadratic* with a known closed form, so the
simultaneous joint update can be checked against the exact arithmetic of one
gradient step and against the analytic optimum.

    E(A, Z) = 0.5 ||A - a*||^2 + 0.5 ||Z - z*||^2 + c * sum(A) * sum(Z)

For a single scalar pair (a, z) with c small enough the fixed point of
``a <- a - alpha * dE/da`` is a linear system we can solve exactly.

This validates **only** the solver update formula, synchrony and convergence
direction; it does not replace the real network tests.
"""

from __future__ import annotations

import torch

from .harness import TestFailure, assert_close, assert_true


def _quad_energy(a_star, z_star, c):
    def E(a, z):
        return 0.5 * ((a - a_star) ** 2).sum(dim=tuple(range(1, a.dim()))) + 0.5 * (
            (z - z_star) ** 2
        ).sum(dim=tuple(range(1, z.dim()))) + c * a.sum(dim=tuple(range(1, a.dim()))) * z.sum(
            dim=tuple(range(1, z.dim()))
        )

    return E


def test_simultaneous_joint_one_step_matches_analytic():
    """One simultaneous step must equal the exact gradient-descent arithmetic."""
    from energywam.energy.solver import refine

    torch.manual_seed(0)
    A = torch.randn(2, 3, 2, dtype=torch.float64)
    Z = torch.randn(2, 1, 1, 4, 2, dtype=torch.float64)
    a_star = torch.randn_like(A)
    z_star = torch.randn_like(Z)
    c = 0.05
    E = _quad_energy(a_star, z_star, c)
    alpha_A, alpha_Z = 0.1, 0.2

    A1, Z1, _ = refine(
        lambda a, z: E(a, z),
        action=A.clone().requires_grad_(True),
        future=Z.clone().requires_grad_(True),
        mode="joint", num_steps=1, alpha_action=alpha_A, alpha_future=alpha_Z,
        training=True, joint_update_mode="simultaneous",
    )

    # exact
    gA = (A - a_star) + c * Z.sum(dim=tuple(range(1, Z.dim()))).view(2, 1, 1).expand_as(A)
    gZ = (Z - z_star) + c * A.sum(dim=tuple(range(1, A.dim()))).view(2, 1, 1, 1, 1).expand_as(Z)
    A_ref = A - alpha_A * gA
    Z_ref = Z - alpha_Z * gZ
    assert_close(A1, A_ref, 1e-12, "simultaneous one-step A")
    assert_close(Z1, Z_ref, 1e-12, "simultaneous one-step Z")


def test_simultaneous_uses_same_point():
    """Simultaneous: both gradients are evaluated at (A_k, Z_k), so the Z update
    must NOT see the already-updated A."""
    from energywam.energy.solver import refine

    torch.manual_seed(1)
    A = torch.randn(1, 2, 2, dtype=torch.float64)
    Z = torch.randn(1, 1, 1, 2, 2, dtype=torch.float64)
    a_star = torch.zeros_like(A)
    z_star = torch.zeros_like(Z)
    c = 0.3
    E = _quad_energy(a_star, z_star, c)
    alpha = 0.1

    A1, Z1, _ = refine(
        lambda a, z: E(a, z),
        action=A.clone().requires_grad_(True),
        future=Z.clone().requires_grad_(True),
        mode="joint", num_steps=1, alpha_action=alpha, alpha_future=alpha,
        training=True, joint_update_mode="simultaneous",
    )
    # Alternating would give Z1_alt = Z - alpha * (Z + c*A1.sum)
    gA = A + c * Z.sum()
    A_after = A - alpha * gA
    gZ_same_point = Z + c * A.sum()  # uses the pre-update A
    gZ_alternating = Z + c * A_after.sum()
    Z_same = Z - alpha * gZ_same_point
    Z_alt = Z - alpha * gZ_alternating
    assert_close(Z1, Z_same, 1e-12, "simultaneous Z must use the same (A_k, Z_k)")
    if torch.allclose(Z1, Z_alt, atol=1e-12):
        raise TestFailure("simultaneous and alternating updates coincide — coupling c is too small to discriminate")


def test_alternating_is_a_separate_config_and_differs():
    """The alternating update is a different, explicitly-named configuration."""
    from energywam.energy.solver import refine

    torch.manual_seed(2)
    A = torch.randn(1, 2, 2, dtype=torch.float64)
    Z = torch.randn(1, 1, 1, 2, 2, dtype=torch.float64)
    E = _quad_energy(torch.zeros_like(A), torch.zeros_like(Z), 0.3)
    alpha = 0.1
    _, Z_sim, _ = refine(
        lambda a, z: E(a, z),
        action=A.clone().requires_grad_(True), future=Z.clone().requires_grad_(True),
        mode="joint", num_steps=2, alpha_action=alpha, alpha_future=alpha,
        training=True, joint_update_mode="simultaneous",
    )
    _, Z_alt, _ = refine(
        lambda a, z: E(a, z),
        action=A.clone().requires_grad_(True), future=Z.clone().requires_grad_(True),
        mode="joint", num_steps=2, alpha_action=alpha, alpha_future=alpha,
        training=True, joint_update_mode="alternating",
    )
    if torch.allclose(Z_sim, Z_alt, atol=1e-14):
        raise TestFailure("alternating and simultaneous produced identical trajectories")
    assert_true(torch.isfinite(Z_sim).all() and torch.isfinite(Z_alt).all(), "finite")


def test_action_only_and_future_only_directions():
    """action_given_future must not move Z; future_given_action must not move A."""
    from energywam.energy.solver import refine

    torch.manual_seed(3)
    A = torch.randn(2, 3, 2, dtype=torch.float64, requires_grad=True)
    Z = torch.randn(2, 1, 1, 4, 2, dtype=torch.float64)
    E = _quad_energy(torch.zeros_like(A), torch.zeros_like(Z), 0.05)
    A_K, Z_K, _ = refine(
        lambda a, z: E(a, z), action=A, future=Z,
        mode="action_given_future", num_steps=3, alpha_action=0.1, alpha_future=0.1,
        training=True,
    )
    assert_close(Z_K, Z, 0.0, "Z moved in action_given_future")
    if torch.allclose(A_K, A):
        raise TestFailure("A did not move in action_given_future")

    A2 = torch.randn(2, 3, 2, dtype=torch.float64)
    Z2 = torch.randn(2, 1, 1, 4, 2, dtype=torch.float64, requires_grad=True)
    A_K2, Z_K2, _ = refine(
        lambda a, z: E(a, z), action=A2, future=Z2,
        mode="future_given_action", num_steps=3, alpha_action=0.1, alpha_future=0.1,
        training=True,
    )
    assert_close(A_K2, A2, 0.0, "A moved in future_given_action")
    if torch.allclose(Z_K2, Z2):
        raise TestFailure("Z did not move in future_given_action")


def test_convergence_towards_analytic_optimum():
    """On the decoupled quadratic (c=0) the update converges to (a*, z*)."""
    from energywam.energy.solver import refine

    torch.manual_seed(4)
    A = torch.randn(2, 3, 2, dtype=torch.float64, requires_grad=True)
    Z = torch.randn(2, 1, 1, 4, 2, dtype=torch.float64, requires_grad=True)
    a_star = torch.randn_like(A)
    z_star = torch.randn_like(Z)
    E = _quad_energy(a_star, z_star, 0.0)
    A_K, Z_K, st = refine(
        lambda a, z: E(a, z), action=A, future=Z,
        mode="joint", num_steps=200, alpha_action=0.4, alpha_future=0.4,
        training=False,
    )
    assert_close(A_K, a_star, 1e-8, "A did not converge to the analytic optimum")
    assert_close(Z_K, z_star, 1e-8, "Z did not converge to the analytic optimum")
    # energy must decrease monotonically for this quadratic
    energies = [float(e.sum()) for e in st.energies]
    for i in range(1, len(energies)):
        if energies[i] > energies[i - 1] + 1e-9:
            raise TestFailure(f"energy increased at step {i}: {energies[i-1]} -> {energies[i]}")


def test_coupled_quadratic_joint_converges():
    """With a small coupling the joint update still converges to the true fixed
    point of the coupled linear system."""
    from energywam.energy.solver import refine

    torch.manual_seed(5)
    B = 1
    A = torch.zeros(B, 1, 1, dtype=torch.float64, requires_grad=True)
    Z = torch.zeros(B, 1, 1, 1, 1, dtype=torch.float64, requires_grad=True)
    a_star = torch.tensor([[[1.0]]], dtype=torch.float64)
    z_star = torch.tensor([[[[[2.0]]]]], dtype=torch.float64)
    c = 0.1
    E = _quad_energy(a_star, z_star, c)
    alpha = 0.2
    A_K, Z_K, _ = refine(
        lambda a, z: E(a, z), action=A, future=Z,
        mode="joint", num_steps=500, alpha_action=alpha, alpha_future=alpha,
        training=False,
    )
    # Fixed point:  a = a* - c z ,  z = z* - c a   (sum over 1 element)
    # => a = a* - c (z* - c a) => a (1 - c^2) = a* - c z*
    a_fp = (float(a_star) - c * float(z_star)) / (1 - c * c)
    z_fp = (float(z_star) - c * float(a_star)) / (1 - c * c)
    assert_close(float(A_K.reshape(-1)[0]), a_fp, 1e-6, "coupled fixed point A")
    assert_close(float(Z_K.reshape(-1)[0]), z_fp, 1e-6, "coupled fixed point Z")


TESTS = [
    ("simultaneous_joint_one_step_matches_analytic", test_simultaneous_joint_one_step_matches_analytic),
    ("simultaneous_uses_same_point", test_simultaneous_uses_same_point),
    ("alternating_is_a_separate_config_and_differs", test_alternating_is_a_separate_config_and_differs),
    ("action_only_and_future_only_directions", test_action_only_and_future_only_directions),
    ("convergence_towards_analytic_optimum", test_convergence_towards_analytic_optimum),
    ("coupled_quadratic_joint_converges", test_coupled_quadratic_joint_converges),
]
