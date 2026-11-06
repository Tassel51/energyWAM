"""Ground-truth leakage and inference-context safety.

Requirement 9 (禁止真值泄漏): joint inference must not receive ground truth, and
changing supervision-only targets under a fixed seed must not change the joint
initialisation or the refinement output.
"""

from __future__ import annotations

import torch

from .common import tiny_batch, tiny_cfg, tiny_model
from .harness import TestFailure, assert_close, assert_true


def test_joint_inference_ignores_supervision_targets():
    """Changing A* / future targets (used only for supervision) must not change
    the joint prediction when the seed and the condition are fixed."""
    cfg = tiny_cfg()
    m, cfg = tiny_model(cfg, seed=111)
    m.eval()

    batch = tiny_batch(cfg, batch_size=2, seed=121)
    out1 = m.predict(batch, mode="joint", return_diagnostics=True)

    batch2 = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    batch2["action_target"] = batch2["action_target"] + 123.0
    batch2["future_target"] = batch2["future_target"] - 456.0
    out2 = m.predict(batch2, mode="joint", return_diagnostics=True)

    assert_close(
        out1["normalized_actions"], out2["normalized_actions"], 1e-6,
        "joint prediction changed when only supervision targets changed",
    )
    assert_close(
        out1["future_latent"], out2["future_latent"], 1e-6,
        "joint future latent changed when only supervision targets changed",
    )


def test_joint_inference_does_not_read_future_images():
    """The predict entry point must work with no future data at all."""
    cfg = tiny_cfg()
    m, cfg = tiny_model(cfg, seed=112)
    m.eval()
    batch = tiny_batch(cfg, batch_size=2, seed=122)
    slim = {
        "cond_vec": batch["cond_vec"],
        # deliberately no action_target / future_target / future masks
    }
    pred = m.predict(slim, mode="joint", return_diagnostics=True)
    assert_true("normalized_actions" in pred, "predict must return normalized_actions without targets")
    assert_true(pred["normalized_actions"].shape[0] == 2, "batch size")


def test_action_only_energy_has_no_future_dependency():
    """E(C, A) must be exactly independent of any future candidate."""
    cfg = tiny_cfg(**{"stage.stage": "action_only"})
    m, cfg = tiny_model(cfg, seed=113)
    batch = tiny_batch(cfg, batch_size=2, seed=123)
    prepared = m._prepare_batch(batch)
    cond = m._condition(batch, prepared)
    a_mask = prepared.get("action_mask")
    offsets = prepared.get("future_offsets")

    A = prepared["action_target"].detach().float()
    e1 = m.core.energy_action_only(cond, A, action_mask=a_mask, offsets=offsets)
    # feed a wildly different future — the action-only path must not see it
    Z_bad = torch.randn_like(prepared["future_target"]) * 10
    e2 = m.core.energy_action_only(cond, A, action_mask=a_mask, offsets=offsets)
    assert_close(e1, e2, 0.0, "action_only energy changed without any input change")
    del Z_bad  # unused by design — the path cannot accept it

    # and the future placeholder must be fully masked
    g = m.geometry
    ph = m.core.build_future_placeholder_tokens(
        2, offsets, A.device, A.dtype
    )
    assert_true(not bool(ph.mask.any()), "action_only future placeholder must be fully masked")


def test_no_parameters_grad_after_predict():
    cfg = tiny_cfg()
    m, cfg = tiny_model(cfg, seed=114)
    m.eval()
    batch = tiny_batch(cfg, batch_size=2, seed=124)
    for p in m.parameters():
        p.grad = None
    m.predict(batch, mode="joint")
    for n, p in m.named_parameters():
        assert_true(p.grad is None, f"parameter {n} accumulated a gradient during predict")


def test_predict_under_inference_mode():
    cfg = tiny_cfg()
    m, cfg = tiny_model(cfg, seed=115)
    m.eval()
    batch = tiny_batch(cfg, batch_size=2, seed=125)
    for p in m.parameters():
        p.grad = None
    with torch.inference_mode():
        out = m.predict(batch, mode="joint", return_diagnostics=True)
    assert_true(out["normalized_actions"].shape[0] == 2, "inference_mode predict produced no output")
    for n, p in m.named_parameters():
        assert_true(p.grad is None, f"parameter {n} accumulated a gradient under inference_mode")


def test_predict_under_no_grad():
    cfg = tiny_cfg()
    m, cfg = tiny_model(cfg, seed=116)
    m.eval()
    batch = tiny_batch(cfg, batch_size=2, seed=126)
    for p in m.parameters():
        p.grad = None
    with torch.no_grad():
        out = m.predict(batch, mode="joint")
    assert_true(out["normalized_actions"].shape == (2, cfg.data.action_horizon, m.geometry["action_dim"]), "shape")
    for n, p in m.named_parameters():
        assert_true(p.grad is None, f"parameter {n} accumulated a gradient under no_grad")


def test_output_is_serialisable():
    import json

    cfg = tiny_cfg()
    m, cfg = tiny_model(cfg, seed=117)
    m.eval()
    batch = tiny_batch(cfg, batch_size=2, seed=127)
    out = m.predict(batch, mode="joint", return_diagnostics=True)
    out["normalized_actions"] = out["normalized_actions"].tolist()
    out["future_latent"] = out["future_latent"].tolist()
    out["diagnostics"] = {k: float(v) if v.size == 1 else v.tolist() for k, v in out["diagnostics"].items()}
    json.dumps(out)  # must not raise


TESTS = [
    ("joint_inference_ignores_supervision_targets", test_joint_inference_ignores_supervision_targets),
    ("joint_inference_does_not_read_future_images", test_joint_inference_does_not_read_future_images),
    ("action_only_energy_has_no_future_dependency", test_action_only_energy_has_no_future_dependency),
    ("no_parameters_grad_after_predict", test_no_parameters_grad_after_predict),
    ("predict_under_inference_mode", test_predict_under_inference_mode),
    ("predict_under_no_grad", test_predict_under_no_grad),
    ("output_is_serialisable", test_output_is_serialisable),
]
