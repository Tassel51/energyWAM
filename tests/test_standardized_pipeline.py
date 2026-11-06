"""Paper §4.8 standardized baseline: end-to-end pipeline checks.

Covers the approved method on the same core/solver as the legacy profile:

* bi / joint / direct_joint / single-direction / action_only train steps in the
  standardized coordinate;
* K>=2 full unroll with finite, non-zero gradients on the energy head, the
  candidate projectors and the backbone (second-order path);
* explicit evidence that the solver optimises ``tilde A / tilde Z`` directly
  (the future projector consumes the standardised target/candidate, the loss
  compares in the standardised space) — not a raw-space optimisation with a
  rescaled loss;
* the hand-built ``future_target`` coordinate contract (no double
  normalisation);
* checkpoint self-containment (external bundle removed), legacy <-> standardized
  refusal, bundle-mismatch refusal, exact raw-action inverse incl. epsilon;
* future output raw/normalised relation + labels;
* action_only never reads future ground truth;
* one real-path run: small random HF Qwen3-VL + real frozen DINOv3, paper
  profile, second-order backward (the pretrained 4B is NOT tested here).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from .harness import TestFailure, assert_close, assert_true

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def paper_cfg(**overrides):
    from energywam.config import load_config

    cfg = load_config(ROOT / "configs" / "tiny_paper_standardized.yaml")
    for k, v in overrides.items():
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = getattr(node, p)
        setattr(node, parts[-1], v)
    return cfg


def paper_model(cfg=None, seed=0, bundle=None):
    """Tiny paper-profile model.  With an explicit ``bundle`` the config's
    bundle path is cleared so the explicit one is authoritative."""
    from energywam.models.energy_wam import QwenEnergyWAM

    cfg = cfg or paper_cfg()
    if bundle is not None:
        cfg.data.norm_stats_bundle_path = None
    torch.manual_seed(seed)
    return QwenEnergyWAM(cfg, stats_bundle=bundle), cfg


def paper_batch(cfg, bundle, *, seed=1, batch_size=2, include_padding=False):
    from energywam.data.tiny_data import make_tiny_batch

    return make_tiny_batch(
        cfg, batch_size=batch_size, seed=seed, include_padding=include_padding,
        standardizer=bundle,
    )


# ---------------------------------------------------------------------------
# 1. train steps in the standardized coordinate (bi / joint / action_only)
# ---------------------------------------------------------------------------
def test_paper_profile_train_steps_all_stages():
    for stage in ("bi", "joint", "direct_joint", "action_given_future",
                  "future_given_action", "action_only"):
        cfg = paper_cfg(**{"stage.stage": stage})
        model, cfg = paper_model(cfg)
        batch = paper_batch(cfg, model.stats_bundle, seed=3)
        out = model(batch)
        loss = out["action_loss"]
        assert_true(torch.isfinite(loss), f"stage {stage}: non-finite loss {loss}")
        loss.backward()
        n_grad = sum(1 for p in model.parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0)
        assert_true(n_grad > 0, f"stage {stage}: no parameter received gradient")
        model.zero_grad(set_to_none=True)


def test_bi_k2_unroll_gradients_finite_nonzero():
    """K>=2 full unroll: head / projectors / backbone all get finite, non-zero grads."""
    cfg = paper_cfg(**{"solver.train_inner_steps": 2})
    model, cfg = paper_model(cfg)
    batch = paper_batch(cfg, model.stats_bundle, seed=5)
    out = model(batch)
    out["action_loss"].backward()

    groups = {
        "energy_head": list(model.core.energy_head.parameters()),
        "future_projector": list(model.core.future_projector.parameters()),
        "action_projector": list(model.core.action_projector.parameters()),
        "backbone": list(model.backbone.parameters()),
    }
    for name, params in groups.items():
        grads = [p.grad for p in params if p.grad is not None]
        assert_true(len(grads) > 0, f"{name} got no gradient")
        total = sum(float(g.abs().sum()) for g in grads)
        finite = all(bool(torch.isfinite(g).all()) for g in grads)
        assert_true(finite, f"{name} has non-finite gradients")
        assert_true(total > 0, f"{name} gradients are all zero")
    assert_true("diag/steps" in out, f"missing diagnostics: {sorted(out)[:8]}")


# ---------------------------------------------------------------------------
# 2. the solver optimises tilde A / tilde Z directly
# ---------------------------------------------------------------------------
def test_future_projector_consumes_standardized_target_and_candidate():
    """The pinned Z* fed to the energy is the standardised target (tilde Z*).

    ``future_projector`` is called with the standardised values — proving the
    optimisation variable is tilde Z (Eq. 39), not raw Z with a rescaled loss.
    """
    from energywam.data.tiny_data import make_tiny_raw_batch

    cfg = paper_cfg()
    model, cfg = paper_model(cfg)
    bundle = model.stats_bundle

    raw = make_tiny_raw_batch(cfg, batch_size=2, seed=9, include_padding=True)
    Z_raw = raw["raw_future_target"].to(torch.float32)
    Z_star = bundle.future.normalize(Z_raw)
    Z_star = torch.where(raw["future_mask"].unsqueeze(-1), Z_star, torch.zeros_like(Z_star))
    A_star = bundle.action.normalize(raw["raw_action_target"].to(torch.float32))
    A_star = torch.where(raw["action_mask"], A_star, torch.zeros_like(A_star))

    batch = {
        "cond_vec": raw["cond_vec"],
        "action_target": A_star,
        "future_target": Z_raw,
        "action_mask": raw["action_mask"].float(),
        "future_mask": raw["future_mask"].float(),
        "future_offsets": raw["future_offsets"],
        "future_target_space": "raw_dino",
        "action_target_space": "standardized",
    }

    seen = []
    handle = model.core.future_projector.register_forward_hook(
        lambda m, inp, out: seen.append(inp[0].detach().clone())
    )
    try:
        model(batch)
    finally:
        handle.remove()
    assert_true(len(seen) > 0, "future_projector was never called")

    # the pinned Z* appears (as tokens [B, F*V*P, D]) among the projector inputs
    want = Z_star.reshape(Z_star.shape[0], -1, Z_star.shape[-1])
    matched = any(
        s.shape == want.shape and bool(torch.allclose(s, want, atol=1e-6)) for s in seen
    )
    assert_true(matched, "future_projector never saw the STANDARDISED Z* target")

    # the standardised target is a z-scored quantity while the raw one is not
    assert_true(abs(float(Z_raw.mean())) > 0.1,
                "test setup: raw future features should be off-centre")
    assert_true(abs(float(Z_star.mean())) < 0.5, "standardised target must be centred")


def test_no_double_normalization_markers():
    """Hand-built future targets must declare their coordinate."""
    from energywam.config import load_config
    from energywam.data.standardization import (
        FUTURE_SPACE_RAW,
        FUTURE_SPACE_STANDARDIZED,
        StatsError,
    )
    from energywam.data.tiny_data import make_tiny_raw_batch
    from energywam.models.energy_wam import QwenEnergyWAM as M

    cfg = paper_cfg()
    model, cfg = paper_model(cfg)
    bundle = model.stats_bundle
    raw = make_tiny_raw_batch(cfg, batch_size=1, seed=4)
    Z_raw = raw["raw_future_target"].to(torch.float32)
    base = {
        "cond_vec": raw["cond_vec"],
        "action_target": bundle.action.normalize(raw["raw_action_target"].to(torch.float32)),
        "action_mask": raw["action_mask"].float(),
        "future_mask": raw["future_mask"].float(),
        "future_offsets": raw["future_offsets"],
        "action_target_space": "standardized",
    }

    def run(space, target):
        b = dict(base)
        b["future_target"] = target
        if space is not None:
            b["future_target_space"] = space
        torch.manual_seed(0)
        return model(b)

    # marker missing -> refuse to guess
    try:
        run(None, Z_raw)
        raise TestFailure("missing future_target_space marker was accepted")
    except TestFailure:
        raise
    except StatsError as exc:
        assert_true("future_target_space" in str(exc), str(exc))

    # raw and pre-standardised inputs must give the SAME loss (one standardisation)
    out_raw = run(FUTURE_SPACE_RAW, Z_raw)
    Z_pre = bundle.future.normalize(Z_raw)
    out_std = run(FUTURE_SPACE_STANDARDIZED, Z_pre)
    assert_close(float(out_raw["action_loss"]), float(out_std["action_loss"]), 1e-5,
                 "raw_dino vs standardized markers must agree")

    # passing RAW values as 'standardized' is double-normalisation: different loss
    out_wrong = run(FUTURE_SPACE_STANDARDIZED, Z_raw)
    assert_true(
        abs(float(out_wrong["action_loss"]) - float(out_raw["action_loss"])) > 1e-6,
        "a mislabelled coordinate changed nothing — the contract is not enforced",
    )

    # legacy profile keeps raw-DINO semantics and rejects the new marker
    lcfg = load_config(ROOT / "configs" / "tiny_cpu.yaml")
    lm = M(lcfg)
    b = dict(base)
    b.pop("action_target_space", None)
    b["future_target"] = Z_raw
    b["future_target_space"] = FUTURE_SPACE_STANDARDIZED
    try:
        lm(b)
        raise TestFailure("legacy profile accepted future_target_space='standardized'")
    except TestFailure:
        raise
    except StatsError as exc:
        assert_true("legacy" in str(exc), str(exc))


# ---------------------------------------------------------------------------
# 3. checkpoint / restore contract
# ---------------------------------------------------------------------------
def test_checkpoint_self_contained_and_cross_refusals():
    from energywam.config import load_config
    from energywam.data.standardization import StatsError
    from energywam.data.tiny_data import synthetic_train_bundle
    from energywam.training.checkpoint import bundle_from_payload, load_checkpoint
    from energywam.training.trainer import EnergyWAMTrainer

    base = load_config(ROOT / "configs" / "tiny_cpu.yaml")
    bundle = synthetic_train_bundle(base, n_windows=32, seed=21)
    with tempfile.TemporaryDirectory() as d:
        cfg = paper_cfg()
        cfg.data.norm_stats_bundle_path = None
        cfg.trainer.output_dir = d
        model, cfg = paper_model(cfg, bundle=bundle)
        tr = EnergyWAMTrainer(cfg, model)
        tr.train_step(paper_batch(cfg, bundle, seed=2))
        path = tr.save("paper.pt")

        payload = load_checkpoint(path)
        assert_true(payload["meta"]["normalization_profile"] == "paper_standardized_v1",
                    "checkpoint must record the coordinate profile")
        emb = bundle_from_payload(payload)
        assert_true(emb is not None and emb.fingerprint == bundle.fingerprint,
                    "checkpoint must embed the full statistics bundle")

        # (a) restore with the EXTERNAL stats file gone: the embedded bundle wins
        cfg_nostats = paper_cfg()
        cfg_nostats.data.norm_stats_bundle_path = str(Path(d) / "does_not_exist.json")
        cfg_nostats.trainer.output_dir = d
        m2, cfg_nostats = paper_model(cfg_nostats, bundle=emb)
        tr2 = EnergyWAMTrainer(cfg_nostats, m2)
        tr2.load(path, mode="resume")

        # predictions are identical after the restore (weights + statistics)
        b = paper_batch(cfg, bundle, seed=8, batch_size=1)
        cond = {"cond_vec": b["cond_vec"]}
        p1 = model.predict(cond, seed=123)["normalized_actions"]
        p2 = m2.predict(cond, seed=123)["normalized_actions"]
        assert_close(p1, p2, 1e-6, "restored model must predict identically")

        # (b) legacy <-> standardized refusals (no silent migration)
        lcfg = load_config(ROOT / "configs" / "tiny_cpu.yaml")
        lcfg.trainer.output_dir = d
        ltr = EnergyWAMTrainer(lcfg, QwenEnergyWAM_lcfg(lcfg))
        lpath = ltr.save("legacy.pt")

        m_p, cfg_p = paper_model(paper_cfg(), bundle=emb)
        tr_p = EnergyWAMTrainer(cfg_p, m_p)
        try:
            tr_p.load(lpath, mode="resume")
            raise TestFailure("legacy->paper restore was not refused")
        except TestFailure:
            raise
        except (ValueError, StatsError) as exc:
            assert_true("normalization coordinate mismatch" in str(exc), str(exc))

        m_l = QwenEnergyWAM_lcfg(lcfg)
        tr_l = EnergyWAMTrainer(lcfg, m_l)
        try:
            tr_l.load(path, mode="resume")
            raise TestFailure("paper->legacy restore was not refused")
        except TestFailure:
            raise
        except (ValueError, StatsError) as exc:
            assert_true("normalization coordinate mismatch" in str(exc), str(exc))

        # (c) different statistics -> refused (resume requires the same stats)
        other = synthetic_train_bundle(base, n_windows=32, seed=99)
        cfg_o = paper_cfg()
        cfg_o.data.norm_stats_bundle_path = None
        cfg_o.trainer.output_dir = d
        m3, cfg_o = paper_model(cfg_o, bundle=other)
        tr3 = EnergyWAMTrainer(cfg_o, m3)
        try:
            tr3.load(path, mode="resume")
            raise TestFailure("resume with different statistics was not refused")
        except TestFailure:
            raise
        except (ValueError, StatsError) as exc:
            assert_true("fingerprint" in str(exc) or "statistics" in str(exc), str(exc))

        # (d) bi -> joint stage switch keeps the SAME statistics and is checked
        cfg_j = paper_cfg(**{"stage.stage": "joint"})
        cfg_j.data.norm_stats_bundle_path = None
        cfg_j.trainer.output_dir = d
        m4, cfg_j = paper_model(cfg_j, bundle=bundle)
        tr4 = EnergyWAMTrainer(cfg_j, m4)
        meta = tr4.switch_stage("joint", path, optimizer_policy="reset", scheduler_policy="reset")
        assert_true(meta.stage == "bi", f"switch source stage {meta.stage}")
        assert_true(tr4.global_step == 0, "stage switch must restart the step counter")


def QwenEnergyWAM_lcfg(cfg):
    from energywam.models.energy_wam import QwenEnergyWAM

    return QwenEnergyWAM(cfg)


def test_raw_action_inverse_includes_epsilon():
    from energywam.data.normalization import normalize, unnormalize

    cfg = paper_cfg()
    model, cfg = paper_model(cfg)
    stats = model.stats_bundle.action.to_norm_stats()
    x = np.asarray([[1.5, -2.0, 0.25]], dtype="float32")
    z = normalize(x, stats)
    want = (x.astype("float64") - model.stats_bundle.action.mean) / (
        model.stats_bundle.action.std + 1e-6
    )
    assert_close(z, want, 1e-5, "zscore forward formula")
    back = unnormalize(z, stats)
    assert_close(back, x, 1e-5, "raw action recovery")
    assert_close(unnormalize(np.zeros_like(x), stats),
                 np.broadcast_to(model.stats_bundle.action.mean.astype("float32"), x.shape),
                 1e-5, "inverse of 0 is mean")


# ---------------------------------------------------------------------------
# 4. inference contract
# ---------------------------------------------------------------------------
def test_predict_outputs_and_no_future_gt():
    cfg = paper_cfg()
    model, cfg = paper_model(cfg)
    bundle = model.stats_bundle
    b = paper_batch(cfg, bundle, seed=11, batch_size=1)

    # predict only needs the condition — no future images / Z* / A*
    cond = {"cond_vec": b["cond_vec"]}
    out = model.predict(cond, return_diagnostics=True, seed=7)
    assert_true(set(out) >= {"normalized_actions", "future_latent",
                             "normalized_future_latent", "future_latent_space"},
                f"predict keys {sorted(out)}")
    assert_true(out["future_latent_space"] == "raw_dino", out["future_latent_space"])
    assert_true("paper_standardized_v1" in out["normalized_future_latent_space"],
                out["normalized_future_latent_space"])
    back = bundle.future.denormalize(out["normalized_future_latent"])
    assert_close(back, out["future_latent"], 1e-5,
                 "future_latent must be the raw DINO inverse of the standardised one")

    # determinism: same condition + same seed -> same output (local generator)
    torch.manual_seed(999)
    o1 = model.predict(cond, seed=7)["normalized_actions"]
    torch.manual_seed(1)
    o2 = model.predict(cond, seed=7)["normalized_actions"]
    assert_close(o1, o2, 0.0, "predict must be seed-deterministic")
    o3 = model.predict(cond, seed=8)["normalized_actions"]
    assert_true(float(np.abs(o3 - o1).max()) > 0, "different seed must change the init")


def test_action_only_never_reads_future_gt():
    from energywam.data.energy_dataset import require_future_for_stage

    assert_true(require_future_for_stage("action_only") is False,
                "action_only must not require future ground truth")
    cfg = paper_cfg(**{"stage.stage": "action_only"})
    model, cfg = paper_model(cfg)
    bundle = model.stats_bundle
    b = paper_batch(cfg, bundle, seed=12)
    torch.manual_seed(0)
    out1 = model(b)

    # NaN future ground truth must not influence the action_only path at all
    b2 = dict(b)
    b2["future_target"] = torch.full_like(b["future_target"], float("nan"))
    torch.manual_seed(0)
    out2 = model(b2)
    assert_true(torch.isfinite(out1["action_loss"]), "action_only loss not finite")
    assert_close(float(out1["action_loss"]), float(out2["action_loss"]), 1e-7,
                 "action_only must ignore the future ground truth")

    # and a batch with NO future target at all also works
    b3 = {k: v for k, v in b.items()
          if k not in ("future_target", "future_mask", "future_target_space")}
    torch.manual_seed(0)
    out3 = model(b3)
    assert_close(float(out1["action_loss"]), float(out3["action_loss"]), 1e-7,
                 "action_only must run without any future tensor")


def test_current_dino_noise_init_through_training_forward():
    """The optional init is reachable end to end and stays in the standardised space."""
    cfg = paper_cfg(**{"init.initialization_future": "current_dino_noise",
                       "init.init_future_noise_std": 0.0})
    model, cfg = paper_model(cfg)
    bundle = model.stats_bundle
    b = paper_batch(cfg, bundle, seed=13)
    # raw current-DINO block at the FUTURE raw mean -> standardised base 0
    cur = torch.zeros(2, 1, cfg.backbone.tiny_future_patches, cfg.backbone.tiny_future_dim)
    cur[...] = torch.from_numpy(bundle.future.mean.astype("float32"))
    b2 = dict(b)
    b2["current_dino"] = cur
    out = model(b2)
    assert_true(torch.isfinite(out["action_loss"]), "current_dino_noise forward not finite")

    cfg2 = paper_cfg(**{"init.initialization_future": "current_dino_noise"})
    m2, cfg2 = paper_model(cfg2)
    bb = paper_batch(cfg2, m2.stats_bundle, seed=13)
    bb2 = dict(bb)
    bb2["current_dino"] = cur
    out2 = m2(bb2)
    assert_true(torch.isfinite(out2["action_loss"]), "current_dino_noise with noise not finite")


# ---------------------------------------------------------------------------
# 5. real path: small HF Qwen3-VL + REAL frozen DINOv3 (4B NOT tested)
# ---------------------------------------------------------------------------
def _real_paper_bundle(cfg, geometry, seed=5):
    """A synthetic-standardized bundle matching the real-path geometry."""
    from energywam.data.standardization import BundleFitter, select_cameras

    Da, D = geometry["action_dim"], geometry["future_dim"]
    b = cfg.backbone
    rng = np.random.RandomState(seed)
    f = BundleFitter(
        action_dim=Da, future_dim=D, epsilon=float(cfg.data.standardization_epsilon),
        provenance={
            "synthetic": True,
            "dataset_identity": "synthetic_real_path/train",
            "split": "train",
            "camera_order": list(cfg.data.camera_order),
            "cameras_used": select_cameras(cfg.data.camera_order,
                                           cfg.data.future_camera_index,
                                           cfg.data.num_cameras),
            "camera_index": int(cfg.data.future_camera_index),
            "num_cameras": int(cfg.data.num_cameras),
            "action_horizon": cfg.data.action_horizon,
            "future_offsets": [int(x) for x in cfg.data.future_offsets],
            "num_future_frames": cfg.data.num_future_frames,
            "dino_preprocess_version": cfg.data.dino_preprocess_version,
            "dino": {
                "model_path": cfg.backbone.dino_path,
                "config_hash": "real",
                "hidden_dim": int(b.dino_hidden_dim),
                "num_patches": int(b.dino_num_patches),
                "num_special_tokens": int(b.dino_num_special_tokens),
                "image_size": int(b.dino_image_size),
                "patch_size": int(b.dino_patch_size),
            },
            "dims": {"action_dim": Da, "future_dim": D},
        },
    )
    f.update_action(rng.randn(6, cfg.data.action_horizon, Da) * 2.0 + 0.5)
    f.update_future(rng.randn(6, cfg.data.num_future_frames, 1, geometry["future_patches"], D))
    return f.finalize(coverage={"kind": "synthetic_train_split"})


def test_real_small_qwen_real_dino_paper_second_order():
    """K>=2 full unroll + backward over the REAL frozen DINO path, paper profile.

    Scope: a small randomly-initialised HF Qwen3-VL (structure/gradient
    plumbing) with the real DINOv3 weights on CPU.  The pretrained 4B weights
    are NOT loaded and are NOT validated by this test.

    The statistics here are synthetic; that is allowed ONLY via the explicit
    ``data.allow_synthetic_stats`` switch — the documented verification boundary
    for an injected non-production backbone.  Real runs never set it.
    """
    from energywam.models.energy_wam import QwenEnergyWAM

    from .real_fixtures import build_real_model, make_real_batch

    ew, cfg, geo = build_real_model(with_real_dino=True, seed=17,
                                    **{"solver.train_inner_steps": 2})
    # switch the shared cfg to the approved paper coordinate
    cfg.data.normalization_profile = "paper_standardized_v1"
    cfg.data.action_normalization = "zscore"
    cfg.init.init_action_std = 1.0
    cfg.init.init_future_std = 1.0
    cfg.data.allow_synthetic_stats = True  # explicit: injected small random Qwen
    bundle = _real_paper_bundle(cfg, geo)
    model = QwenEnergyWAM(cfg, backbone=ew.backbone, stats_bundle=bundle).to("cpu")

    batch = make_real_batch(model, cfg, geo, B=2, seed=2, standardizer=bundle)
    assert_true(batch.get("future_target_space") == "standardized", "batch must carry markers")

    out = model(batch)
    loss = out["action_loss"]
    assert_true(torch.isfinite(loss), f"real-path paper loss not finite: {loss}")
    loss.backward()

    for name, params in (
        ("energy_head", list(model.core.energy_head.parameters())),
        ("future_projector", list(model.core.future_projector.parameters())),
        ("backbone", list(model.backbone.parameters())),
    ):
        grads = [p.grad for p in params if p.grad is not None and p.requires_grad]
        assert_true(len(grads) > 0, f"real path: {name} got no gradient")
        assert_true(all(bool(torch.isfinite(g).all()) for g in grads),
                    f"real path: {name} has non-finite gradients")
        assert_true(sum(float(g.abs().sum()) for g in grads) > 0,
                    f"real path: {name} gradients are all zero")


def test_legacy_predict_output_schema_unchanged():
    """The legacy profile keeps the historical predict output schema exactly."""
    from energywam.config import load_config
    from energywam.data.tiny_data import make_tiny_batch
    from energywam.models.energy_wam import QwenEnergyWAM

    cfg = load_config(ROOT / "configs" / "tiny_cpu.yaml")
    torch.manual_seed(0)
    m = QwenEnergyWAM(cfg)
    b = make_tiny_batch(cfg, batch_size=1, seed=31)
    out = m.predict({"cond_vec": b["cond_vec"]}, return_diagnostics=True, seed=3)
    assert_true(set(out) == {"normalized_actions", "future_latent", "diagnostics", "mode"},
                f"legacy predict keys changed: {sorted(out)}")
    # and the full output stays JSON-serialisable after the historical tolist() pass
    import json

    out["normalized_actions"] = out["normalized_actions"].tolist()
    out["future_latent"] = out["future_latent"].tolist()
    out["diagnostics"] = {k: float(v) if v.size == 1 else v.tolist()
                          for k, v in out["diagnostics"].items()}
    json.dumps(out)


def test_dataset_items_are_standardized_with_markers():
    """The real dataset emits tilde A* / tilde Z* plus space markers."""
    from energywam.data.energy_dataset import EnergyWAMEpisodeDataset, collate_energy_batch
    from energywam.data.standardization import BundleFitter

    from .real_fixtures import StubDINO

    cfg = paper_cfg()
    cfg.data.state_normalization = "none"
    cfg.data.action_horizon = 8
    cfg.data.future_offsets = [3, 5, 8]
    cfg.data.camera_order = ("cam0",)
    Da, D, P = cfg.backbone.tiny_action_dim, cfg.backbone.tiny_future_dim, cfg.backbone.tiny_future_patches

    class FakeReader:
        dataset_identity = "fake/standardized"
        camera_keys = ["cam0"]
        episodes = [{"episode_index": 0, "length": 20, "tasks": ["t"]}]

        def __init__(self):
            self.actions = np.stack(
                [np.asarray([1.0 * t + d for d in range(Da)], dtype="float32") for t in range(20)]
            )

        def episode_length(self, ep):
            return 20

        def read_actions(self, ep):
            return self.actions.copy()

        def read_states(self, ep):
            return np.zeros((20, 4), dtype="float32")

        def read_language(self, ep):
            return "t"

        def read_frame(self, ep, cam, idx):
            rng = np.random.RandomState(100 + idx)
            return rng.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)

    f = BundleFitter(action_dim=Da, future_dim=D, epsilon=1e-6,
                     provenance={"synthetic": True, "dataset_identity": "fake/standardized",
                                 "camera_order": ["cam0"], "action_horizon": 8,
                                 "future_offsets": [3, 5, 8]})
    f.update_action(FakeReader().actions.reshape(-1, Da))
    f.update_future(np.random.RandomState(0).randn(4, 3, 1, P, D))
    bundle = f.finalize(coverage={"kind": "synthetic_train_split"})

    dino = StubDINO(patches=P, dim=D)
    ds = EnergyWAMEpisodeDataset(
        cfg, reader=FakeReader(), action_stats=bundle.action.to_norm_stats(),
        state_stats=None, dino=dino, max_samples=3,
        require_future=True, future_standardizer=bundle.future,
    )
    item = ds[0]
    assert_true(item["action_target_space"] == "standardized", item["action_target_space"])
    assert_true(item["future_target_space"] == "standardized", item["future_target_space"])

    # manual recomputation from the same raw sample
    reader = ds.reader
    raw_a = reader.read_actions(0)[: cfg.data.action_horizon]
    want_a = bundle.action.normalize(raw_a.astype("float64")).astype("float32")
    assert_close(item["action_target"], want_a, 1e-5, "dataset action target must be tilde A*")

    from energywam.data.future_offsets import plan_future_frames

    plan = plan_future_frames(0, 20, 8, [3, 5, 8], require_full_horizon=True)
    flat = [reader.read_frame(0, "cam0", abs_f) for abs_f in plan.abs_frames]
    with torch.no_grad():
        feats = dino.encode_images(flat, device="cpu").numpy()  # [F, P, D]
    want_z = bundle.future.normalize(feats.reshape(3, 1, P, D)).astype("float32")
    assert_close(item["future_target"], want_z, 1e-4, "dataset future target must be tilde Z*")
    assert_true(abs(float(feats.mean())) > 1e-3, "setup: raw DINO features should be off-centre")

    batch = collate_energy_batch([item, ds[1]])
    assert_true(batch["future_target_space"] == "standardized", "collate must pass markers")
    assert_true(batch["action_target_space"] == "standardized", "collate must pass markers")


TESTS = [
    ("paper_profile_train_steps_all_stages", test_paper_profile_train_steps_all_stages),
    ("bi_k2_unroll_gradients_finite_nonzero", test_bi_k2_unroll_gradients_finite_nonzero),
    ("future_projector_consumes_standardized_target_and_candidate", test_future_projector_consumes_standardized_target_and_candidate),
    ("no_double_normalization_markers", test_no_double_normalization_markers),
    ("checkpoint_self_contained_and_cross_refusals", test_checkpoint_self_contained_and_cross_refusals),
    ("raw_action_inverse_includes_epsilon", test_raw_action_inverse_includes_epsilon),
    ("predict_outputs_and_no_future_gt", test_predict_outputs_and_no_future_gt),
    ("action_only_never_reads_future_gt", test_action_only_never_reads_future_gt),
    ("current_dino_noise_init_through_training_forward", test_current_dino_noise_init_through_training_forward),
    ("real_small_qwen_real_dino_paper_second_order", test_real_small_qwen_real_dino_paper_second_order),
    ("legacy_predict_output_schema_unchanged", test_legacy_predict_output_schema_unchanged),
    ("dataset_items_are_standardized_with_markers", test_dataset_items_are_standardized_with_markers),
]
