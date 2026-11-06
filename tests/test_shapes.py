"""Shapes, variable-length inputs, padding, and Energy-token indexing."""

from __future__ import annotations

import torch

from .common import tiny_batch, tiny_cfg, tiny_model
from .harness import TestFailure, assert_true


def test_shapes_and_batch():
    cfg = tiny_cfg()
    m, cfg = tiny_model(cfg, seed=1)
    B = 4
    batch = tiny_batch(cfg, batch_size=B, seed=3)
    out = m(batch)
    assert_true(out["action_loss"].dim() == 0, f"loss must be scalar, got {tuple(out['action_loss'].shape)}")
    assert_true(torch.isfinite(out["action_loss"]), "loss not finite")

    g = m.geometry
    pred = m.predict(batch, mode="joint", return_diagnostics=True)
    A = pred["normalized_actions"]
    assert_true(A.shape == (B, g["action_horizon"], g["action_dim"]), f"A shape {A.shape}")
    Z = pred["future_latent"]
    assert_true(
        Z.shape == (B, g["num_future_frames"], g["num_cameras"], g["future_patches"], g["future_dim"]),
        f"Z shape {Z.shape}",
    )


def test_energy_token_index_is_last_and_valid():
    """The Energy token must be the final sequence position and must be a valid
    (non-padding) token even when the batch contains padded condition tokens."""
    cfg = tiny_cfg()
    m, cfg = tiny_model(cfg, seed=2)
    batch = tiny_batch(cfg, batch_size=3, seed=5, include_padding=True)
    prepared = m._prepare_batch(batch)
    cond = m._condition(batch, prepared)

    Z = prepared["future_target"].float()
    A = prepared["action_target"].float()
    fut = m.core.build_future_tokens(
        Z, offsets=prepared.get("future_offsets"), future_mask=prepared.get("future_mask")
    )
    act = m.core.build_action_tokens(A, action_mask=prepared.get("action_mask"))
    eng = m.core.build_energy_tokens(Z.shape[0], Z.device, Z.dtype)
    read = m.core.energy_from_groups(cond, [fut, act, eng])
    S = fut.length + act.length + eng.length + cond.group.length
    idx = read.energy_token_index
    assert_true(torch.all(idx == S - 1), f"energy index {idx.tolist()} != last position {S - 1}")
    # the energy token is always valid
    assert_true(bool(eng.mask.all()), "energy token mask must be all True")
    assert_true(read.energy.shape == (Z.shape[0],), f"energy shape {tuple(read.energy.shape)}")


def test_variable_condition_length_batch_independence_shape():
    """Different batch sizes give consistent per-sample energy shapes."""
    cfg = tiny_cfg()
    m, cfg = tiny_model(cfg, seed=3)
    for B in (1, 2, 5):
        batch = tiny_batch(cfg, batch_size=B, seed=10 + B)
        out = m(batch)
        assert_true(out["action_loss"].dim() == 0, "scalar loss")
        assert_true(torch.isfinite(out["action_loss"]), "finite loss")


def test_action_only_path_shape():
    cfg = tiny_cfg(**{"stage.stage": "action_only"})
    m, cfg = tiny_model(cfg, seed=4)
    batch = tiny_batch(cfg, batch_size=2, seed=8)
    out = m(batch)
    assert_true(torch.isfinite(out["action_loss"]), "action_only loss finite")
    pred = m.predict(batch, mode="action_only", return_diagnostics=True)
    assert_true(pred["normalized_actions"].shape[0] == 2, "batch dim")


TESTS = [
    ("shapes_and_batch", test_shapes_and_batch),
    ("energy_token_index_is_last_and_valid", test_energy_token_index_is_last_and_valid),
    ("variable_condition_length_batch_independence_shape", test_variable_condition_length_batch_independence_shape),
    ("action_only_path_shape", test_action_only_path_shape),
]
