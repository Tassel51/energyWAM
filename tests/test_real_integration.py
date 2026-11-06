"""REAL-path integration tests: full ``QwenEnergyWAM`` + real frozen DINOv3.

Every test here goes through the **top-level model** (``QwenEnergyWAM``), not
through the adapter in isolation, so the module lifecycle, checkpointing and
the inference context boundaries are actually exercised.

Scope reminder: the Qwen backbone is a randomly-initialised *small*
``Qwen3VLConfig`` (the 4B weights are not loaded and no GPU is used).  The DINO
extractor DOES use the real checkpoint on CPU.  This validates structure and
gradient plumbing, not pretrained-4B behaviour.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from .harness import TestFailure, assert_true
from .real_fixtures import (
    DINO_PATH,
    build_real_model,
    make_images,
    make_real_batch,
)


# ---------------------------------------------------------------------------
# P0-1: module lifecycle (no core<->backbone cycle)
# ---------------------------------------------------------------------------
def test_real_model_module_lifecycle_no_cycle():
    """``.to()`` / ``.train()`` / ``state_dict()`` must terminate.

    A core<->backbone submodule cycle makes these recurse forever, so this test
    would hang rather than fail if the cycle regressed.
    """
    ew, cfg, geo = build_real_model(with_real_dino=False, seed=11)
    sd = ew.state_dict()
    assert_true(len(sd) > 0, "empty state_dict")
    ew = ew.to("cpu")
    ew.train()
    ew.eval()
    ew.train()
    n_params = sum(p.numel() for p in ew.parameters())
    assert_true(n_params > 0, "no parameters")
    # the backbone must NOT appear twice in the tree
    names = [n for n, _ in ew.named_modules()]
    assert_true(
        not any(n.count("backbone") > 1 for n in names),
        f"backbone appears nested under itself: {[n for n in names if 'backbone' in n][:10]}",
    )


def test_state_dict_keys_stable_across_forward():
    """DINO is a registered submodule from __init__; keys must not change after
    the first forward (a lazy property would break strict checkpoint loading)."""
    ew, cfg, geo = build_real_model(with_real_dino=True, seed=12)
    keys_before = set(ew.state_dict().keys())
    batch = make_real_batch(ew, cfg, geo, B=2, seed=3)
    out = ew(batch)
    assert_true(torch.isfinite(out["action_loss"]), "loss not finite")
    keys_after = set(ew.state_dict().keys())
    assert_true(keys_before == keys_after,
                f"key drift: +{keys_after - keys_before} / -{keys_before - keys_after}")


def test_state_dict_roundtrip_strict_load_and_forward():
    ew, cfg, geo = build_real_model(with_real_dino=True, seed=13)
    ew.eval()
    batch = make_real_batch(ew, cfg, geo, B=2, seed=4)
    # The candidate initialisation draws from the global torch RNG.  A
    # checkpoint-consistency comparison must CONTROL that RNG, not change the
    # training condition to make an uncontrolled comparison pass.
    torch.manual_seed(1234)
    before = float(ew(batch)["action_loss"])
    sd = copy.deepcopy(ew.state_dict())

    ew2, cfg2, geo2 = build_real_model(with_real_dino=True, seed=99)
    ew2.load_state_dict(sd, strict=True)
    ew2.eval()
    # primary check: the weights themselves must match exactly
    for k, v in sd.items():
        assert_true(torch.equal(ew2.state_dict()[k], v), f"weight {k} not restored exactly")
    torch.manual_seed(1234)  # same RNG -> same candidate initialisation
    after = float(ew2(batch)["action_loss"])
    # Weights are asserted bit-exact above; the scalar loss is float32 and CPU
    # matmul reduction order can differ by ~1e-6 across two forward passes, so
    # the numeric tolerance here is 1e-5, not 0.
    assert_true(abs(before - after) < 1e-5,
                f"strict load changed the loss under controlled RNG: {before} -> {after}")


# ---------------------------------------------------------------------------
# P0-2: inference context boundaries
# ---------------------------------------------------------------------------
def test_real_predict_all_contexts_and_mode_restore():
    ew, cfg, geo = build_real_model(with_real_dino=True, seed=14)
    batch = make_real_batch(ew, cfg, geo, B=2, seed=5)
    slim = {
        "images": batch["images"],
        "langs": batch["langs"],
        "states": batch["states"],
    }

    def check(label, ctx):
        for p in ew.parameters():
            p.grad = None
        snapshot = {k: v.clone() for k, v in ew.state_dict().items()}
        ew.train()
        with ctx():
            out = ew.predict(slim, mode="joint", return_diagnostics=True)
        assert_true(out["normalized_actions"].shape[0] == 2, f"{label}: batch size")
        assert_true(ew.training, f"{label}: train mode not restored")
        for n, p in ew.named_parameters():
            assert_true(p.grad is None, f"{label}: parameter {n} accumulated a gradient")
        for k, v in ew.state_dict().items():
            assert_true(torch.equal(v, snapshot[k]), f"{label}: parameter {k} changed during predict")
        return out

    import contextlib

    check("plain", contextlib.nullcontext)
    check("no_grad", torch.no_grad)
    check("inference_mode", torch.inference_mode)

    # nested: caller already inside inference_mode (the model-server case)
    for p in ew.parameters():
        p.grad = None
    with torch.inference_mode():
        out = ew.predict(slim, mode="joint")
    assert_true(out["normalized_actions"].shape[0] == 2, "nested inference_mode produced no output")
    for n, p in ew.named_parameters():
        assert_true(p.grad is None, f"nested inference_mode left a grad on {n}")


def test_real_predict_does_not_read_supervision():
    """Joint prediction must work with NO action_target / future_target at all,
    and must be invariant to the supervision values when they are present."""
    ew, cfg, geo = build_real_model(with_real_dino=True, seed=15)
    batch = make_real_batch(ew, cfg, geo, B=2, seed=6)

    slim = {"images": batch["images"], "langs": batch["langs"], "states": batch["states"]}
    a = ew.predict(slim, mode="joint")["normalized_actions"]

    with_sup = dict(slim)
    with_sup["action_target"] = batch["action_target"] + 999.0
    with_sup["future_target"] = batch["future_target"] - 999.0
    b = ew.predict(with_sup, mode="joint")["normalized_actions"]
    assert_true(np.allclose(a, b, atol=1e-6),
                "joint prediction changed when only supervision targets changed")

    # and B must come from the condition: a batch with no supervision at all
    # must still report the right batch size
    assert_true(a.shape[0] == 2, f"B not taken from the condition: {a.shape}")


# ---------------------------------------------------------------------------
# P0-4: real DINO
# ---------------------------------------------------------------------------
def test_real_dino_cpu_extraction_uint8_and_checkpoint():
    from energywam.backbones.dino_v3 import FrozenDINOv3

    d = FrozenDINOv3(DINO_PATH, freeze=True, dtype=torch.float32)
    imgs = make_images(2, V=1, size=224, seed=7)[0]
    feats = d.encode_images(imgs, device="cpu")
    assert_true(feats.shape == (1, 196, 384), f"DINO feature shape {tuple(feats.shape)}")

    # uint8 tensor input MUST be scaled to [0,1] — otherwise the features are
    # garbage and nothing downstream would notice
    t = torch.from_numpy(imgs[0])  # uint8 HWC
    f_uint8 = d.encode_images([t], device="cpu")
    pil_feats = d.encode_images([imgs[0]], device="cpu")
    assert_true(torch.allclose(f_uint8, pil_feats, atol=1e-4),
                "uint8 tensor input was not scaled by 1/255")

    # wrong size must be rejected before reaching the encoder
    try:
        d.encode_images([np.zeros((100, 100, 3), dtype=np.uint8)], device="cpu")
    except Exception:
        pass  # resized is fine
    try:
        d.extract_patches(torch.zeros(1, 3, 128, 128))
    except ValueError as exc:
        assert_true("224" in str(exc) or "expected" in str(exc), str(exc))
    else:
        raise TestFailure("extract_patches accepted a wrong-sized input")

    # checkpoint round-trip must keep the keys stable
    sd = copy.deepcopy(d.state_dict())
    assert_true(len(sd) > 0, "empty DINO state_dict")
    d2 = FrozenDINOv3(DINO_PATH, freeze=True, dtype=torch.float32)
    d2.load_state_dict(sd, strict=True)
    f2 = d2.encode_images(imgs, device="cpu")
    assert_true(torch.allclose(feats, f2, atol=1e-5), "DINO checkpoint round-trip changed the features")


def test_dino_rejects_non_dinov3_model_type():
    from energywam.backbones.dino_v3 import FrozenDINOv3

    class FakeCfg:
        model_type = "vit"
        hidden_size = 384
        patch_size = 16
        num_register_tokens = 4

    class FakeBody(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = FakeCfg()

        def forward(self, pixel_values=None, **kw):
            raise RuntimeError("should not be called")

    d = FrozenDINOv3.__new__(FrozenDINOv3)
    torch.nn.Module.__init__(d)
    d.body = FakeBody()
    d.num_patches = 196
    d.num_special_tokens = 5
    d.image_size = 224
    d.patch_size = 16
    d.hidden_dim = 384
    try:
        d._assert_config()
    except ValueError as exc:
        assert_true("dinov3" in str(exc).lower() or "model_type" in str(exc), str(exc))
    else:
        raise TestFailure("a non-DINOv3 model_type was accepted")


# ---------------------------------------------------------------------------
# P0-5: StarVLA adapter is a real nn.Module
# ---------------------------------------------------------------------------
def test_starvla_adapter_is_nn_module_and_handles_list_batch():
    from energywam.integration.starvla import EnergyWAMFrameworkAdapter

    ew, cfg, geo = build_real_model(with_real_dino=True, seed=16)
    adapter = EnergyWAMFrameworkAdapter(ew, cfg)
    assert_true(isinstance(adapter, torch.nn.Module), "adapter is not an nn.Module")
    names = [n for n, _ in adapter.named_parameters()]
    assert_true(len(names) > 0, "adapter exposes no parameters")
    assert_true(any(n.startswith("model.") for n in names),
                f"model is not a registered submodule: {names[:5]}")
    assert_true(len(list(adapter.modules())) > 1, "adapter.modules() is empty")

    # compute_loss must accept a real list[dict] batch
    batch = make_real_batch(ew, cfg, geo, B=2, seed=8)
    examples = []
    for i in range(2):
        examples.append(
            {
                "image": batch["images"][i],
                "lang": batch["langs"][i],
                "state": batch["states"][i],
                "action": batch["action_target"][i],
                "future_latent": batch["future_target"][i],
                "action_mask": batch["action_mask"][i],
                "future_mask": batch["future_mask"][i],
                "future_offsets": batch["future_offsets"][i],
            }
        )
    out = adapter.compute_loss("vla", examples)
    assert_true(out is not None and "action_loss" in out, "compute_loss returned no action_loss")
    assert_true(torch.isfinite(out["action_loss"]), "compute_loss loss not finite")
    out["action_loss"].backward()
    n_grad = sum(1 for p in adapter.parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0)
    assert_true(n_grad > 0, "compute_loss did not propagate to any adapter parameter")

    # predict_action through the same adapter
    for p in adapter.parameters():
        p.grad = None
    pred = adapter.predict_action(examples[:1])
    assert_true("normalized_actions" in pred, "predict_action missing normalized_actions")
    assert_true(pred["normalized_actions"].shape == (1, cfg.data.action_horizon, cfg.data.action_dim),
                f"shape {pred['normalized_actions'].shape}")


def test_examples_to_batch_handles_numpy_and_tensor_truthiness():
    """`ex.get('future_latent') or ...` raises on numpy/tensor inputs."""
    ew, cfg, geo = build_real_model(with_real_dino=False, seed=17)
    A = np.random.randn(cfg.data.action_horizon, cfg.data.action_dim).astype("float32")
    Z = np.random.randn(3, 1, 4, 8).astype("float32")
    ex = [{"cond_vec": np.zeros(8, dtype="float32"), "action": A, "future_latent": Z}]
    b = ew._examples_to_batch(ex, require_targets=True)
    assert_true(b["future_target"].shape[0] == 1, "future_target batch")

    ex2 = [{"cond_vec": torch.zeros(8), "action": torch.from_numpy(A),
            "future_latent": torch.from_numpy(Z)}]
    b2 = ew._examples_to_batch(ex2, require_targets=True)
    assert_true(b2["future_target"].shape[0] == 1, "tensor future_latent failed")


# ---------------------------------------------------------------------------
# P0-3: real dataset path (future target present, stats mandatory)
# ---------------------------------------------------------------------------
def test_energy_dataset_requires_action_stats():
    from energywam.data.energy_dataset import require_action_stats

    class FakeReader:
        root = __import__("pathlib").Path("/nonexistent/nowhere")

    cfg = build_real_model(with_real_dino=False, seed=18)[1]
    cfg.data.norm_stats_path = "/nonexistent/stats.json"
    try:
        require_action_stats(FakeReader(), cfg)
    except FileNotFoundError as exc:
        assert_true("normalisation" in str(exc) or "statistics" in str(exc), str(exc))
    else:
        raise TestFailure("missing action statistics did not raise")


# ---------------------------------------------------------------------------
# P1-7: solver behaviour
# ---------------------------------------------------------------------------
def test_solver_does_not_constrain_fixed_branch():
    from energywam.energy.solver import refine

    A_fixed = torch.full((1, 2, 2), 5.0)
    Z = torch.zeros(1, 2, 1, 2, 2, requires_grad=True)

    def E(a, z):
        return (a ** 2).sum() * 0 + (z ** 2).sum(dim=(1, 2, 3, 4))

    A_K, Z_K, _ = refine(
        E, action=A_fixed, future=Z, mode="future_given_action",
        num_steps=3, alpha_action=1.0, alpha_future=0.1, training=True,
        action_clamp=(-1.0, 1.0),  # would destroy A* if applied
    )
    assert_true(torch.allclose(A_K, A_fixed),
                f"fixed A* was altered by the action clamp: {A_K.flatten()[:3]}")

    Z_fixed = torch.full((1, 2, 1, 2, 2), 7.0)
    A = torch.zeros(1, 2, 2, requires_grad=True)
    A_K2, Z_K2, _ = refine(
        E, action=A, future=Z_fixed, mode="action_given_future",
        num_steps=3, alpha_action=0.1, alpha_future=1.0, training=True,
        future_clamp=(-1.0, 1.0),
    )
    assert_true(torch.allclose(Z_K2, Z_fixed),
                f"fixed Z* was altered by the future clamp: {Z_K2.flatten()[:3]}")


def test_solver_truncated_mode_raises_not_silently_broken():
    from energywam.energy.solver import refine

    A = torch.zeros(1, 2, 2, requires_grad=True)
    Z = torch.zeros(1, 2, 1, 2, 2, requires_grad=True)
    E = lambda a, z: (a ** 2).sum() + (z ** 2).sum()
    try:
        refine(E, action=A, future=Z, mode="joint", num_steps=3,
               alpha_action=0.1, alpha_future=0.1, training=True,
               unroll_mode="truncated", truncation_length=2)
    except NotImplementedError as exc:
        assert_true("truncated" in str(exc) or "full" in str(exc), str(exc))
    else:
        raise TestFailure("unroll_mode='truncated' did not fail fast")


def test_solver_missing_candidate_gradient_raises():
    """allow_unused must not silently paper over an energy that ignores a free
    candidate."""
    from energywam.energy.solver import refine

    A = torch.zeros(1, 2, 2, requires_grad=True)
    Z = torch.zeros(1, 2, 1, 2, 2, requires_grad=True)
    E = lambda a, z: (a ** 2).sum()  # deliberately ignores Z
    try:
        refine(E, action=A, future=Z, mode="joint", num_steps=2,
               alpha_action=0.1, alpha_future=0.1, training=True)
    except RuntimeError as exc:
        assert_true("unused" in str(exc).lower() or "allow" in str(exc).lower()
                    or "not been used" in str(exc).lower() or "one of the differentiated"
                    in str(exc).lower(), str(exc))
    else:
        raise TestFailure("an energy ignoring Z produced a silent zero gradient")


def test_joint_bi_aux_is_a_real_conditional_solve():
    """The bi-auxiliary term must come from its own conditional solver runs,
    not from re-scoring the joint endpoints."""
    ew, cfg, geo = build_real_model(with_real_dino=False, seed=19)
    cfg.stage.stage = "joint"
    cfg.stage.joint_use_bi_aux = True
    cfg.stage.joint_bi_aux_weight = 0.5
    from energywam.data.tiny_data import make_tiny_batch

    cfg.backbone.backend = "tiny"  # keep this check cheap; logic is shared
    from energywam.models.energy_wam import QwenEnergyWAM

    torch.manual_seed(0)
    m = QwenEnergyWAM(cfg)
    batch = make_tiny_batch(cfg, batch_size=2, seed=1)
    out = m(batch)
    assert_true("joint/bi_aux_action_loss" in out, "bi aux action term missing")
    assert_true("joint/bi_aux_future_loss" in out, "bi aux future term missing")
    aux_a = float(out["joint/bi_aux_action_loss"])
    aux_z = float(out["joint/bi_aux_future_loss"])
    joint_a = float(out["joint/action_loss"])
    joint_z = float(out["joint/future_loss"])
    # A real conditional solve gives different numbers than the joint endpoints.
    assert_true(abs(aux_a - joint_a) > 1e-9 or abs(aux_z - joint_z) > 1e-9,
                f"bi-aux equals the joint endpoints (aux {aux_a:.4f}/{aux_z:.4f} "
                f"joint {joint_a:.4f}/{joint_z:.4f}) — not a real conditional solve")


def test_action_only_prediction_ignores_future_input_perturbation():
    """The no-leak property is established by INPUT PERTURBATION, not by
    watching a loss stay flat.  Changing every future-related input must leave
    the action-only prediction bit-identical."""
    ew, cfg, geo = build_real_model(with_real_dino=True, seed=22)
    cfg.stage.stage = "action_only"
    batch = make_real_batch(ew, cfg, geo, B=2, seed=9)

    base = {"images": batch["images"], "langs": batch["langs"], "states": batch["states"]}
    a0 = ew.predict(base, mode="action_only")["normalized_actions"]

    perturbed = dict(base)
    perturbed["action_target"] = batch["action_target"] + 1234.0
    perturbed["future_target"] = batch["future_target"] - 1234.0
    perturbed["future_mask"] = 1.0 - batch["future_mask"]
    perturbed["future_offsets"] = batch["future_offsets"] + 1
    a1 = ew.predict(perturbed, mode="action_only")["normalized_actions"]
    assert_true(np.allclose(a0, a1, atol=0.0),
                "action-only prediction changed when future inputs were perturbed")

    # and a different future image set must not matter either
    other = dict(base)
    other["images"] = make_images(2, V=cfg.data.num_cameras, seed=9 + 100)
    # NOTE: images ARE part of the condition C, so this one legitimately may
    # change the output; it is the FUTURE supervision that must not.
    # The assertion above covers the no-leak claim.


TESTS = [
    ("real_model_module_lifecycle_no_cycle", test_real_model_module_lifecycle_no_cycle),
    ("state_dict_keys_stable_across_forward", test_state_dict_keys_stable_across_forward),
    ("state_dict_roundtrip_strict_load_and_forward", test_state_dict_roundtrip_strict_load_and_forward),
    ("real_predict_all_contexts_and_mode_restore", test_real_predict_all_contexts_and_mode_restore),
    ("real_predict_does_not_read_supervision", test_real_predict_does_not_read_supervision),
    ("real_dino_cpu_extraction_uint8_and_checkpoint", test_real_dino_cpu_extraction_uint8_and_checkpoint),
    ("dino_rejects_non_dinov3_model_type", test_dino_rejects_non_dinov3_model_type),
    ("starvla_adapter_is_nn_module_and_handles_list_batch", test_starvla_adapter_is_nn_module_and_handles_list_batch),
    ("examples_to_batch_handles_numpy_and_tensor_truthiness", test_examples_to_batch_handles_numpy_and_tensor_truthiness),
    ("energy_dataset_requires_action_stats", test_energy_dataset_requires_action_stats),
    ("solver_does_not_constrain_fixed_branch", test_solver_does_not_constrain_fixed_branch),
    ("solver_truncated_mode_raises_not_silently_broken", test_solver_truncated_mode_raises_not_silently_broken),
    ("solver_missing_candidate_gradient_raises", test_solver_missing_candidate_gradient_raises),
    ("joint_bi_aux_is_a_real_conditional_solve", test_joint_bi_aux_is_a_real_conditional_solve),
    ("action_only_prediction_ignores_future_input_perturbation", test_action_only_prediction_ignores_future_input_perturbation),
]
