"""Regression tests for mask handling in the token builders and masked losses.

These tests are written to fail against the pre-fix behaviour reproduced by
``coordination/codex_probe.py`` (see ``logs/codex-probes-round1.log``):

* changing the values of an ``action_mask=False`` action **dimension** changed
  the energy by ``1.0558988e-01`` — the masked dimension was projected before
  being masked, so it leaked into the token embedding;
* ``masked_mse`` returned NaN when a *masked-out* target element was NaN
  (``NaN * 0`` in the error reduction).

Covered here:

1. masked action dims (values and NaN) cannot move the energy or the token
   embeddings;
2. ``build_future_tokens`` accepts ``[B,F]`` / ``[B,F,V]`` / ``[B,F,V,P]`` /
   ``[B,F,V,P,D]`` and broadcastable masks, and masks the content *before* the
   projector;
3. ``masked_mse`` / ``masked_cosine`` ignore invalid entries (including NaN)
   and return a differentiable zero for an empty mask, preserving float64;
4. ``EnergyHead`` accepts a BF16 readout while the head itself stays FP32.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .common import tiny_batch, tiny_cfg, tiny_model
from .harness import TestFailure, assert_close, assert_true


def _setup(seed: int = 111, batch_seed: int = 124, batch_size: int = 2):
    cfg = tiny_cfg()
    m, cfg = tiny_model(cfg, seed=seed)
    m.eval()
    b = tiny_batch(cfg, batch_size=batch_size, seed=batch_seed)
    p = m._prepare_batch(b)
    c = m._condition(b, p)
    return m, cfg, b, p, c


# ---------------------------------------------------------------------------
# action tokens
# ---------------------------------------------------------------------------
def test_masked_action_dim_cannot_change_energy():
    """A masked action *dimension* must not influence E (codex probe case)."""
    m, cfg, b, p, c = _setup()
    a = p["action_target"].clone()
    z = p["future_target"].clone()
    mask = torch.ones_like(a, dtype=torch.bool)
    mask[..., 0] = False  # one invalid action dimension everywhere

    e_ref = m.core.energy(c, z, a, action_mask=mask)

    a_big = a.clone()
    a_big[..., 0] += 999.0  # pre-fix: energy moved by 1.0558988e-01
    e_big = m.core.energy(c, z, a_big, action_mask=mask)
    assert_close(e_big, e_ref, 1e-6, "masked action dimension changed the energy")

    a_nan = a.clone()
    a_nan[..., 0] = float("nan")  # pre-fix: NaN leaked through the projector
    e_nan = m.core.energy(c, z, a_nan, action_mask=mask)
    assert_true(torch.isfinite(e_nan).all(), "NaN in a masked action dimension poisoned the energy")
    assert_close(e_nan, e_ref, 1e-6, "NaN in a masked action dimension changed the energy")


def test_build_action_tokens_masks_before_projection():
    """Invalid dims are zeroed before the projector; token mask is ``any(-1)``."""
    m, cfg, b, p, c = _setup()
    core = m.core
    A = p["action_target"].detach().clone().float()
    B, H, Da = A.shape

    mask = torch.ones(B, H, Da, dtype=torch.bool)
    mask[0, 2:5, 1] = False  # some dims of some steps
    mask[1, 0, :] = False  # one fully invalid step

    g = core.build_action_tokens(A, action_mask=mask)
    expected_tok = mask.any(dim=-1)
    assert_true(torch.equal(g.mask, expected_tok), "token mask must be any() over the action dims")

    A2 = A.clone()
    A2[0, 2:5, 1] = 777.0
    A2[1, 0, :] = -1.0e6
    g2 = core.build_action_tokens(A2, action_mask=mask)
    assert_close(g2.embeds, g.embeds, 1e-6, "masked action dims leaked into the token embeddings")

    A3 = A.clone()
    A3[0, 2:5, 1] = float("nan")
    A3[1, 0, :] = float("nan")
    g3 = core.build_action_tokens(A3, action_mask=mask)
    assert_true(torch.isfinite(g3.embeds).all(), "NaN in masked action dims poisoned the embeddings")
    assert_close(g3.embeds, g.embeds, 1e-6, "NaN in masked action dims changed the embeddings")

    # fully invalid tokens stay exactly zero (no residual leak through biases)
    assert_true(float(g.embeds[1, 0].abs().max()) == 0.0, "fully masked action token is not zeroed")

    # reduced [B, H] masks and unmasked calls keep the old semantics
    tok_only = mask.all(dim=-1)
    g4 = core.build_action_tokens(A, action_mask=tok_only)
    assert_true(torch.equal(g4.mask, tok_only), "[B,H] mask must map to the same token mask")
    g5 = core.build_action_tokens(A, action_mask=None)
    assert_true(bool(g5.mask.all()), "no mask means every action token is valid")


# ---------------------------------------------------------------------------
# future tokens
# ---------------------------------------------------------------------------
def test_build_future_tokens_accepts_reduced_masks():
    """[B,F], [B,F,V], [B,F,V,P], [B,F,V,P,D] and broadcastable shapes."""
    m, cfg, b, p, c = _setup()
    core = m.core
    Z = p["future_target"].detach().clone().float()
    B, F, V, P, D = Z.shape
    N = F * V * P

    # [B, F] — per-frame
    fm = torch.ones(B, F, dtype=torch.bool)
    fm[:, -1] = False
    g = core.build_future_tokens(Z, future_mask=fm)
    expected = fm[:, :, None, None].expand(B, F, V, P).reshape(B, N)
    assert_true(torch.equal(g.mask, expected), f"[B,F] mask -> token mask mismatch")

    # [B, F, V] — per-frame/camera
    fv = torch.ones(B, F, V, dtype=torch.bool)
    fv[:, 0, :] = False
    g = core.build_future_tokens(Z, future_mask=fv)
    expected = fv[:, :, :, None].expand(B, F, V, P).reshape(B, N)
    assert_true(torch.equal(g.mask, expected), "[B,F,V] mask -> token mask mismatch")

    # [B, F, V, P] — per-token (the historical shape)
    fvp = torch.ones(B, F, V, P, dtype=torch.bool)
    fvp[0, 1, 0, 2] = False
    g = core.build_future_tokens(Z, future_mask=fvp)
    expected = fvp.reshape(B, N)
    assert_true(torch.equal(g.mask, expected), "[B,F,V,P] mask -> token mask mismatch")

    # full [B, F, V, P, D] — per-channel; token valid if ANY channel is valid
    full = torch.ones(B, F, V, P, D, dtype=torch.bool)
    full[..., 0] = False  # one channel invalid everywhere
    full[1, -1] = False  # one token fully invalid
    g = core.build_future_tokens(Z, future_mask=full)
    expected = full.any(dim=-1).reshape(B, N)
    assert_true(torch.equal(g.mask, expected), "[B,F,V,P,D] mask -> token mask (any over channels)")

    # broadcastable shapes
    g = core.build_future_tokens(Z, future_mask=fm[:, :, None, None, None])
    assert_true(torch.equal(g.mask, fm[:, :, None, None].expand(B, F, V, P).reshape(B, N)),
                "[B,F,1,1,1] mask -> token mask mismatch")
    shared = torch.ones(F, V, P, dtype=torch.bool)  # shared across the batch
    shared[1] = False
    g = core.build_future_tokens(Z, future_mask=shared)
    expected = shared[None].expand(B, F, V, P).reshape(B, N)
    assert_true(torch.equal(g.mask, expected), "[F,V,P] mask -> token mask mismatch")

    assert_true(core.build_future_tokens(Z, future_mask=None).mask.all(), "no mask = all valid")


def test_future_masked_values_do_not_change_tokens_or_energy():
    """Masked future content (values and NaN) cannot leak through the projector."""
    m, cfg, b, p, c = _setup()
    core = m.core
    Z = p["future_target"].detach().clone().float()
    B, F, V, P, D = Z.shape

    fvp = torch.ones(B, F, V, P, dtype=torch.bool)
    fvp[0, -1] = False
    fvp[1, 0, :, :2] = False

    g = core.build_future_tokens(Z, future_mask=fvp)
    Z2 = Z.clone()
    Z2[~fvp] = 999.0
    g2 = core.build_future_tokens(Z2, future_mask=fvp)
    assert_close(g2.embeds, g.embeds, 1e-6, "masked future values leaked into the token embeddings")

    Z3 = Z.clone()
    Z3[~fvp] = float("nan")  # pre-fix: NaN * 0 in the post-projection multiply
    g3 = core.build_future_tokens(Z3, future_mask=fvp)
    assert_true(torch.isfinite(g3.embeds).all(), "NaN in masked future entries poisoned the embeddings")
    assert_close(g3.embeds, g.embeds, 1e-6, "NaN in masked future entries changed the embeddings")
    assert_true(float(g.embeds[~fvp.reshape(B, F * V * P)].abs().max()) == 0.0,
                "fully masked future tokens are not zeroed")

    a = p["action_target"].clone()
    e_ref = m.core.energy(c, Z, a, future_mask=fvp)
    e_nan = m.core.energy(c, Z3, a, future_mask=fvp)
    assert_true(torch.isfinite(e_nan).all(), "NaN in masked future entries poisoned the energy")
    assert_close(e_nan, e_ref, 1e-6, "masked future values changed the energy")

    # the reduced per-frame form reaches the energy path as well
    fm = fvp.all(dim=(2, 3))
    e_frm = m.core.energy(c, Z3, a, future_mask=fm)
    assert_true(torch.isfinite(e_frm).all(), "[B,F] future_mask is not accepted / not sanitised")


# ---------------------------------------------------------------------------
# masked losses
# ---------------------------------------------------------------------------
def test_masked_mse_ignores_invalid_targets():
    """Invalid target entries (including NaN) must not reach the reduction."""
    from energywam.energy.losses import masked_mse

    # exact codex probe: invalid target element is NaN
    x = torch.ones(1, 2, requires_grad=True)
    target = torch.tensor([[1.0, float("nan")]])
    loss = masked_mse(x, target, torch.tensor([[1, 0]]))
    assert_true(torch.isfinite(loss).all(), "masked_mse returned NaN because a masked target was NaN")
    assert_close(float(loss), 0.0, 1e-12, "masked_mse over valid elements should be 0")

    # float64 must be preserved for the numerical tests
    pred = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    tgt = torch.tensor([[1.0, -7.0, 3.0]], dtype=torch.float64)
    mask = torch.tensor([[1, 0, 1]], dtype=torch.bool)
    loss = masked_mse(pred, tgt, mask)
    assert_true(loss.dtype == torch.float64, f"masked_mse dtype {loss.dtype} != float64")
    assert_close(float(loss), 0.0, 1e-12, "valid-only mse should be exactly 0")

    # non-trivial reference: mean over valid elements, computed in float64
    pred = torch.randn(2, 3, 4, dtype=torch.float64)
    tgt = torch.randn(2, 3, 4, dtype=torch.float64)
    mask = torch.rand(2, 3, 4) > 0.4
    ref = ((pred - tgt) ** 2)[mask].mean()
    loss = masked_mse(pred, tgt, mask)
    assert_close(float(loss), float(ref), 1e-12, "masked_mse != masked mean in float64")

    # heavy + NaN perturbations of invalid entries change nothing
    tgt2 = tgt.clone()
    tgt2[~mask] = 1.0e6
    pred2 = pred.clone()
    pred2[~mask] = float("nan")
    loss2 = masked_mse(pred2, tgt2, mask)
    assert_true(torch.isfinite(loss2).all(), "NaN in masked pred poisoned the loss")
    assert_close(float(loss2), float(ref), 1e-12, "masked pred/target values changed the loss")


def test_empty_mask_returns_differentiable_zero():
    """Empty mask -> a backprop-able zero in the promoted dtype (not NaN/ValueError)."""
    from energywam.energy.losses import masked_cosine, masked_mse

    for fn in (masked_mse, masked_cosine):
        pred = torch.zeros(2, 3, 4, dtype=torch.float64, requires_grad=True)
        target = torch.full((2, 3, 4), float("nan"), dtype=torch.float64)
        mask = torch.zeros(2, 3, 4, dtype=torch.bool)
        loss = fn(pred, target, mask)
        assert_true(torch.isfinite(loss).all(), f"{fn.__name__}: empty mask must not give NaN")
        assert_close(float(loss), 0.0, 0.0, f"{fn.__name__}: empty mask must give 0")
        assert_true(loss.dtype == torch.float64, f"{fn.__name__}: empty mask lost float64")
        assert_true(bool(loss.requires_grad), f"{fn.__name__}: empty mask zero is detached from the graph")
        loss.backward()
        assert_true(pred.grad is not None, f"{fn.__name__}: no gradient flowed through the empty-mask zero")
        assert_true(torch.isfinite(pred.grad).all(), f"{fn.__name__}: non-finite grad from empty mask")
        assert_true(float(pred.grad.abs().max()) == 0.0, f"{fn.__name__}: empty mask must give zero grad")


def test_masked_cosine_partial_channel_semantics():
    """Partial channel masks: cosine of the valid sub-vector, ``any`` token rule.

    Documented semantics: masked channels of both pred and target are
    zero-filled before the dot product and the norms (exactly the cosine of the
    valid sub-vector); a token contributes iff at least one channel is valid.
    """
    from energywam.energy.losses import masked_cosine

    torch.manual_seed(7)
    pred = torch.randn(2, 3, 4, dtype=torch.float64)
    tgt = torch.randn(2, 3, 4, dtype=torch.float64)

    mask = torch.ones(2, 3, 4, dtype=torch.bool)
    mask[:, :, 1] = False  # channel 1 invalid everywhere
    mask[0, 0, :] = False  # one token fully invalid -> excluded

    pred2 = pred.clone()
    tgt2 = tgt.clone()
    pred2[~mask] = float("nan")
    tgt2[~mask] = float("nan")

    loss = masked_cosine(pred2, tgt2, mask)
    assert_true(torch.isfinite(loss).all(), "NaN in masked channels poisoned the cosine loss")

    # reference: cosine over the valid channels of the valid tokens
    keep_ch = [0, 2, 3]
    keep_tok = mask.any(dim=-1)  # [2, 3]
    pv = pred[..., keep_ch][keep_tok]
    tv = tgt[..., keep_ch][keep_tok]
    ref = (1.0 - F.cosine_similarity(pv, tv, dim=-1)).mean()
    assert_close(float(loss), float(ref), 1e-10, "partial-channel cosine != cosine of the valid sub-vector")

    # a fully masked token must not contribute: flipping its values is a no-op
    pred3 = pred2.clone()
    pred3[0, 0] = 1.0e6
    loss3 = masked_cosine(pred3, tgt2, mask)
    assert_close(float(loss3), float(loss), 1e-12, "fully masked token contributed to the cosine loss")


# ---------------------------------------------------------------------------
# mixed-precision energy head
# ---------------------------------------------------------------------------
def test_energy_head_mixed_dtype_readout():
    """BF16 backbone readout into an FP32 head must work (head keeps its dtype)."""
    from energywam.models.energy_core import EnergyHead

    torch.manual_seed(3)
    head = EnergyHead(16, head_hidden=8, dropout=0.0).float().eval()
    h32 = torch.randn(4, 16, dtype=torch.float32)
    with torch.no_grad():
        e_ref = head(h32)
        e_bf16 = head(h32.to(torch.bfloat16))  # pre-fix: dtype mismatch inside nn.Linear
    assert_true(e_bf16.dtype == torch.float32, f"head output dtype {e_bf16.dtype} != float32")
    assert_true(torch.isfinite(e_bf16).all(), "mixed-dtype readout produced non-finite energy")
    assert_close(e_bf16, e_ref, 5e-2, "BF16 readout should track the FP32 head within BF16 tolerance")

    # and the other direction: BF16 head with an FP32 readout
    head16 = EnergyHead(16, head_hidden=8, dropout=0.0).bfloat16().eval()
    with torch.no_grad():
        e16 = head16(h32)
    assert_true(e16.dtype == torch.bfloat16, f"head output dtype {e16.dtype} != bfloat16")
    assert_true(torch.isfinite(e16.float()).all(), "BF16 head produced non-finite energy")


TESTS = [
    ("masked_action_dim_cannot_change_energy", test_masked_action_dim_cannot_change_energy),
    ("build_action_tokens_masks_before_projection", test_build_action_tokens_masks_before_projection),
    ("build_future_tokens_accepts_reduced_masks", test_build_future_tokens_accepts_reduced_masks),
    ("future_masked_values_do_not_change_tokens_or_energy", test_future_masked_values_do_not_change_tokens_or_energy),
    ("masked_mse_ignores_invalid_targets", test_masked_mse_ignores_invalid_targets),
    ("empty_mask_returns_differentiable_zero", test_empty_mask_returns_differentiable_zero),
    ("masked_cosine_partial_channel_semantics", test_masked_cosine_partial_channel_semantics),
    ("energy_head_mixed_dtype_readout", test_energy_head_mixed_dtype_readout),
]
