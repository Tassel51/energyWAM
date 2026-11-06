"""Contract regressions from the 2026-10-29 independent review (P0-1..P0-4, P1).

Each test asserts the CORRECT contract; against the pre-fix code the key ones
fail (see ``logs/standardized_20261029_reviewfix_before.log``).  Coverage:

* P0-1 training split wiring: training samples come from the bundle's recorded
  TRAIN episodes only (identity / existence / empty / duplicate / bad index all
  rejected, no all-episode fallback); eval can read held-out episodes and the
  eval report must not claim held-out when nothing selected it;
* P0-2 camera selection: current and future frames come from the SAME selected
  camera(s); statistics fitting and training read the same cameras; bundle
  ``cameras_used`` / ``camera_index`` / ``num_cameras`` are validated against
  the run (a stale ``camera_order`` alone is not enough);
* P0-3 action-only bundles: with no future statistics and no future need the
  DINO identity check is skipped (the current-frame DINO still loads normally);
* P0-4 provenance protection: synthetic statistics are refused on the real
  backend by default (tiny allowed; explicit switch for injected-backbone
  verification); DINO config identity is checked beyond ``model_path``; bundle
  payloads without a fingerprint are refused;
* P1 checkpoint: the norm buffers inside the payload are checked against the
  embedded bundle before ``load_state_dict`` (wrapper prefixes supported).
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from .harness import TestFailure, assert_close, assert_true

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def tiny_paper_cfg(**overrides):
    from energywam.config import load_config

    cfg = load_config(ROOT / "configs" / "tiny_paper_standardized.yaml")
    cfg.data.norm_stats_bundle_path = None
    for k, v in overrides.items():
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = getattr(node, p)
        setattr(node, parts[-1], v)
    return cfg


def real_cfg(name: str = "bi", **overrides):
    """Paper-profile REAL (qwen3vl) config — no weights are ever loaded here."""
    from energywam.config import load_config

    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    cfg.data.norm_stats_bundle_path = None
    for k, v in overrides.items():
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = getattr(node, p)
        setattr(node, parts[-1], v)
    return cfg


def fit_bundle(cfg, *, action_rows, future_rows, synthetic=True, train_episodes=(0, 1),
               identity="fake/contract", cameras_used=None, camera_index=None,
               dino=None, include_future=True, extra=None):
    """Build a bundle with explicit provenance from raw arrays."""
    from energywam.config import geometry_from_cfg, resolve_future_offsets
    from energywam.data.standardization import BundleFitter

    geom = geometry_from_cfg(cfg)
    cams = list(cfg.data.camera_order) if cfg.data.camera_order else []
    idx = int(cfg.data.future_camera_index if camera_index is None else camera_index)
    sel = list(cameras_used) if cameras_used is not None else cams[idx: idx + cfg.data.num_cameras]
    prov = {
        "synthetic": bool(synthetic),
        "dataset_identity": identity,
        "split": "train",
        "train_episodes": {
            "episode_indices": [int(e) for e in train_episodes],
            "manifest_sha256": "x",
            "manifest_path": "",
            "num_episodes": len(list(train_episodes)),
        },
        "camera_order": cams,
        "cameras_used": sel,
        "camera_index": idx,
        "num_cameras": int(cfg.data.num_cameras),
        "action_horizon": int(cfg.data.action_horizon),
        "future_offsets": [int(x) for x in resolve_future_offsets(cfg.data)],
        "num_future_frames": int(cfg.data.num_future_frames),
        "dino_preprocess_version": cfg.data.dino_preprocess_version,
        "dino": dict(dino or {
            "model_path": "synthetic", "config_hash": "synthetic",
            "hidden_dim": geom["future_dim"], "num_patches": geom["future_patches"],
            "num_special_tokens": 5, "image_size": 224, "patch_size": 16,
        }),
        "dims": {"action_dim": geom["action_dim"], "future_dim": geom["future_dim"]},
    }
    prov.update(extra or {})
    f = BundleFitter(action_dim=geom["action_dim"],
                     future_dim=geom["future_dim"] if include_future else None,
                     epsilon=float(cfg.data.standardization_epsilon), provenance=prov)
    f.update_action(np.asarray(action_rows, dtype=np.float64))
    if include_future:
        f.update_future(np.asarray(future_rows, dtype=np.float64))
    return f.finalize(coverage={"kind": "synthetic_train_split"})


class SpyReader:
    """Fake LeRobot reader with 4 episodes (0,1 = train, 2 = val, 3 = test)."""

    dataset_identity = "fake/contract"
    camera_keys = ["cam_a", "cam_b"]
    root = "/tmp/fake_contract_corpus"

    def __init__(self, lengths=(20, 20, 20, 20), action_dim=3, state_dim=4):
        self.episodes = [{"episode_index": i, "length": int(L), "tasks": [f"t{i}"]}
                         for i, L in enumerate(lengths)]
        self.lengths = {i: int(L) for i, L in enumerate(lengths)}
        self.action_dim = action_dim
        self.read_calls = []  # (episode, camera, frame)

    def episode_length(self, ep):
        return self.lengths[int(ep)]

    def read_actions(self, ep):
        L = self.episode_length(ep)
        return np.stack([np.asarray([1.0 * t + 0.1 * d for d in range(self.action_dim)],
                                    dtype="float32") for t in range(L)])

    def read_states(self, ep):
        L = self.episode_length(ep)
        return np.zeros((L, 4), dtype="float32")

    def read_language(self, ep):
        return f"t{ep}"

    def read_frame(self, ep, cam, idx):
        self.read_calls.append((int(ep), str(cam), int(idx)))
        # numerically distinct per camera so a wrong camera is detectable
        base = 10.0 if cam == "cam_a" else 200.0
        arr = np.full((8, 8, 3), base + int(idx), dtype=np.float64)
        return arr.astype("uint8")


class MeanDino:
    """Deterministic stand-in: ``[P=4, D=8]`` features, row c = frame_mean + c."""

    model_path = "mean_dino"
    hidden_dim = 8
    num_patches = 4
    num_special_tokens = 0
    image_size = 8
    patch_size = 2

    def encode_images(self, images, device="cpu"):
        out = []
        for img in images:
            m = float(np.asarray(img, dtype=np.float64).mean() / 255.0)
            row = np.asarray([m + c for c in range(8)], dtype="float32")
            out.append(np.stack([row] * 4))  # [P=4, D=8]
        return torch.as_tensor(np.stack(out))


class StubBackbone(torch.nn.Module):
    """Construction-only stand-in so qwen3vl configs never load real weights."""

    hidden_size = 32

    def __init__(self):
        super().__init__()
        self.dino = None


# ---------------------------------------------------------------------------
# P0-1: training split wiring
# ---------------------------------------------------------------------------
def _contract_cfg():
    return tiny_paper_cfg(**{
        "data.camera_order": ("cam_a", "cam_b"),
        "data.future_camera_index": 1,
        "data.num_cameras": 1,
    })


def test_training_index_restricted_to_bundle_train_episodes():
    """Training reads ONLY the bundle's train episodes; eval can read holdout."""
    from energywam.data import energy_dataset as ed
    from energywam.data.energy_dataset import build_dataloader, build_eval_examples

    cfg = _contract_cfg()
    reader = SpyReader()
    action_rows = reader.read_actions(0).reshape(-1, cfg.backbone.tiny_action_dim)
    bundle = fit_bundle(cfg, action_rows=action_rows,
                        future_rows=np.random.RandomState(0).randn(4, 1, 1, 1, 8),
                        train_episodes=(0, 1), identity="fake/contract")

    orig_root, orig_reader = ed.resolve_dataset_root, ed.LeRobotEpisodeReader
    ed.resolve_dataset_root = lambda c: Path("/tmp/fake_root")
    ed.LeRobotEpisodeReader = lambda root: reader
    try:
        loader = build_dataloader(cfg, dino=MeanDino(), stats_bundle=bundle,
                                  max_samples=64, shuffle=False)
        eps = {int(e) for e, _ in loader.dataset._index}
        assert_true(eps == {0, 1}, f"training index covered episodes {sorted(eps)}, expected [0, 1]")
        assert_true(len(loader.dataset._index) > 0, "empty training index")

        # eval: unrestricted by default — the holdout episodes ARE reachable
        examples, _ = build_eval_examples(cfg, reader=reader, max_samples=64,
                                          stats_bundle=bundle)
        seen = set()
        ds = ed.EnergyWAMEpisodeDataset(
            cfg, reader=reader, action_stats=bundle.action.to_norm_stats(),
            state_stats=None, dino=None, require_future=False, max_samples=64,
        )
        for i in range(len(ds)):
            seen.add(int(ds._index[i][0]))
        assert_true({0, 1, 2, 3} <= seen,
                    f"eval dataset must be able to read holdout episodes, saw {sorted(seen)}")

        # explicit holdout selection is honoured for eval
        ds_h = ed.EnergyWAMEpisodeDataset(
            cfg, reader=reader, action_stats=bundle.action.to_norm_stats(),
            state_stats=None, dino=None, require_future=False, max_samples=64,
            episode_indices=[2],
        )
        seen_h = {int(e) for e, _ in ds_h._index}
        assert_true(seen_h == {2}, f"explicit eval selection gave {sorted(seen_h)}")
    finally:
        ed.resolve_dataset_root, ed.LeRobotEpisodeReader = orig_root, orig_reader


def test_training_binding_rejects_bad_identity_and_episodes():
    from energywam.data import energy_dataset as ed
    from energywam.data.energy_dataset import build_dataloader
    from energywam.data.standardization import StatsError

    cfg = _contract_cfg()
    reader = SpyReader()
    rows = reader.read_actions(0).reshape(-1, cfg.backbone.tiny_action_dim)

    orig_root, orig_reader = ed.resolve_dataset_root, ed.LeRobotEpisodeReader
    ed.resolve_dataset_root = lambda c: Path("/tmp/fake_root")
    ed.LeRobotEpisodeReader = lambda root: reader

    def expect_error(bundle, needle):
        try:
            build_dataloader(cfg, dino=MeanDino(), stats_bundle=bundle,
                             max_samples=8, shuffle=False)
        except (StatsError, ValueError, RuntimeError) as exc:
            assert_true(needle in str(exc).lower(),
                        f"expected {needle!r} in error, got: {exc}")
            return
        raise TestFailure(f"build_dataloader accepted a bad bundle ({needle})")

    try:
        # wrong dataset identity
        b = fit_bundle(cfg, action_rows=rows, future_rows=np.random.randn(2, 1, 1, 1, 8),
                       identity="other/dataset")
        expect_error(b, "dataset")
        # train episode not in the reader
        b = fit_bundle(cfg, action_rows=rows, future_rows=np.random.randn(2, 1, 1, 1, 8),
                       train_episodes=(0, 7))
        expect_error(b, "episode")
        # empty train list
        b = fit_bundle(cfg, action_rows=rows, future_rows=np.random.randn(2, 1, 1, 1, 8),
                       train_episodes=())
        expect_error(b, "episode")
        # duplicate indices
        b = fit_bundle(cfg, action_rows=rows, future_rows=np.random.randn(2, 1, 1, 1, 8),
                       train_episodes=(0, 0))
        expect_error(b, "duplicate")
        # no recorded train episodes -> NO fallback to all episodes
        b = fit_bundle(cfg, action_rows=rows, future_rows=np.random.randn(2, 1, 1, 1, 8))
        b.provenance.pop("train_episodes")
        expect_error(b, "train")
    finally:
        ed.resolve_dataset_root, ed.LeRobotEpisodeReader = orig_root, orig_reader


def test_eval_report_never_claims_heldout_without_selection():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "report.json"
        r = subprocess.run(
            [PY, str(ROOT / "scripts" / "evaluate.py"),
             "--config", "configs/tiny_paper_standardized.yaml",
             "--synthetic", "2", "--out", str(out)],
            capture_output=True, text=True, timeout=300,
            env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        assert_true(r.returncode == 0, r.stdout + r.stderr)
        report = json.loads(out.read_text())
        h = report.get("heldout") or {}
        assert_true(h.get("guaranteed_heldout") is False,
                    f"report must not claim held-out without selection: {h}")
        note = json.dumps(h, ensure_ascii=False) + json.dumps(report.get("subset", {}), ensure_ascii=False)
        assert_true("held" in note.lower() or "留出" in note,
                    f"report must state the held-out status explicitly: {note[:200]}")


# ---------------------------------------------------------------------------
# P0-2: camera selection
# ---------------------------------------------------------------------------
def test_build_sample_current_and_future_use_same_selected_camera():
    """camera_index=1 single camera: current AND future must both read cam_b."""
    from energywam.data.episode_dataset import DatasetSpec, build_sample

    reader = SpyReader()
    spec = DatasetSpec(action_horizon=4, future_offsets=(2, 3, 4), num_cameras=1,
                       camera_index=1, camera_keys=("cam_a", "cam_b"))
    s = build_sample(reader, 0, 0, spec, action_stats=None, state_stats=None)
    assert_true(s is not None, "sample not built")
    cams_used = {cam for (_, cam, _) in reader.read_calls}
    assert_true(cams_used == {"cam_b"},
                f"camera_index=1 must read ONLY cam_b (current+future), got {sorted(cams_used)}")
    # numeric check: future pixels come from the cam_b track (value 200 + idx)
    fut0 = np.asarray(s.future_images[0][0])
    idx = s.future_frames_abs[0]
    assert_close(float(fut0.mean()), 200.0 + float(idx), 0.5,
                 "future frame content is not the selected camera's")
    cur = np.asarray(s.images[0])
    assert_close(float(cur.mean()), 200.0 + 0.0, 0.5, "current frame content")

    # multi-camera selection is the ordered slice starting at camera_index
    reader2 = SpyReader()
    spec2 = DatasetSpec(action_horizon=4, future_offsets=(2, 3, 4), num_cameras=2,
                        camera_index=0, camera_keys=("cam_a", "cam_b"))
    build_sample(reader2, 0, 0, spec2, action_stats=None, state_stats=None)
    cams2 = {cam for (_, cam, _) in reader2.read_calls}
    assert_true(cams2 == {"cam_a", "cam_b"}, f"two-camera selection: {sorted(cams2)}")


def test_stats_fit_and_training_read_same_cameras_with_numbers():
    """Statistics fitting and the training dataset must read the same camera."""
    from energywam.data.energy_dataset import EnergyWAMEpisodeDataset
    from energywam.data.stats_fit import fit_bundle_from_reader

    cfg = _contract_cfg()  # camera_order=(cam_a,cam_b), future_camera_index=1 -> cam_b
    rows = None

    fit_reader = SpyReader()
    bundle = fit_bundle_from_reader(
        fit_reader, cfg, episodes=[0, 1], dino=MeanDino(),
        include_future=True,
        manifest={"manifest_sha256": "m", "manifest_path": ""},
    )
    assert_true(bundle.provenance["cameras_used"] == ["cam_b"],
                bundle.provenance["cameras_used"])
    assert_true(bundle.provenance["camera_index"] == 1, bundle.provenance["camera_index"])
    fit_cams = {cam for (_, cam, _) in fit_reader.read_calls}
    assert_true(fit_cams == {"cam_b"}, f"stats fit read cameras {sorted(fit_cams)}")

    # hand-computed future statistics from the cam_b tracks only
    from energywam.data.future_offsets import plan_future_frames

    vals = []
    for ep in (0, 1):
        L = fit_reader.episode_length(ep)
        for t in range(L):
            plan = plan_future_frames(t, L, 8, [3, 5, 8], require_full_horizon=True)
            if not plan.valid:
                continue
            for abs_f in plan.abs_frames:
                frame = SpyReader().read_frame(ep, "cam_b", int(abs_f))
                m = float(np.asarray(frame, dtype=np.float64).mean() / 255.0)
                vals.append([m + c for c in range(8)])
    v = np.asarray(vals, dtype=np.float64)
    assert_close(bundle.future.mean, v.mean(axis=0), 1e-6, "future mean from cam_b")
    assert_close(bundle.future.std, v.std(axis=0), 1e-6, "future std from cam_b")

    # training dataset (same cfg) reads the same camera keys
    ds_reader = SpyReader()
    ds = EnergyWAMEpisodeDataset(
        cfg, reader=ds_reader, action_stats=bundle.action.to_norm_stats(),
        state_stats=None, dino=MeanDino(), require_future=True, max_samples=8,
        future_standardizer=bundle.future,
    )
    it = ds[0]
    ds_cams = {cam for (_, cam, _) in ds_reader.read_calls}
    assert_true(ds_cams == {"cam_b"}, f"training dataset read cameras {sorted(ds_cams)}")
    assert_true(it["future_target"].shape[-1] == 8, it["future_target"].shape)


def test_bundle_camera_selection_mismatch_rejected():
    from energywam.data.standardization import StatsError

    cfg = _contract_cfg()
    rows = np.random.RandomState(1).randn(24, 3)
    b = fit_bundle(cfg, action_rows=rows,
                   future_rows=np.random.RandomState(2).randn(3, 1, 1, 1, 8))
    kw = dict(action_dim=3, future_dim=8, camera_order=["cam_a", "cam_b"],
              cameras_used=["cam_b"], camera_index=1, num_cameras=1,
              action_horizon=8, future_offsets=[3, 5, 8], require_future=True,
              dino_preprocess_version=cfg.data.dino_preprocess_version, epsilon=1e-6,
              allow_synthetic=True)
    b.validate_geometry(**kw)  # consistent -> ok
    for over in ({"cameras_used": ["cam_a"]}, {"camera_index": 0}, {"num_cameras": 2}):
        k = dict(kw)
        k.update(over)
        try:
            b.validate_geometry(**k)
            raise TestFailure(f"camera mismatch {over} accepted")
        except TestFailure:
            raise
        except StatsError as exc:
            assert_true("camera" in str(exc).lower(), str(exc))


# ---------------------------------------------------------------------------
# P0-3: action-only bundle without future statistics
# ---------------------------------------------------------------------------
def test_action_only_bundle_skips_future_dino_identity():
    from energywam.data.energy_dataset import resolve_stats_bundle
    from energywam.data.standardization import StatsError
    from energywam.models.energy_wam import QwenEnergyWAM

    cfg = real_cfg("action_only")  # qwen3vl backend, real dino_path configured
    cfg.data.allow_synthetic_stats = True  # explicit verification switch (stub backbone)
    action_rows = np.random.RandomState(3).randn(24, cfg.data.action_dim)
    # an action-only fit records no DINO identity at all (CLI --no-future)
    b = fit_bundle(cfg, action_rows=action_rows, future_rows=None,
                   include_future=False,
                   dino={"model_path": "", "config_hash": "", "hidden_dim": 384,
                         "num_patches": 196, "num_special_tokens": 5,
                         "image_size": 224, "patch_size": 16},
                   cameras_used=[str(cfg.data.camera_order[0])], camera_index=0)
    b.provenance.pop("dino")
    b.provenance["dino"] = {"model_path": ""}

    # validation entry used by the loader: must PASS (no future standardization)
    got = resolve_stats_bundle(cfg, stats_bundle=b, require_future=False)
    assert_true(got.future is None, "action-only bundle must stay future-less")

    # and the model constructor (fail-fast path) must accept it with a stub backbone
    m = QwenEnergyWAM(cfg, backbone=StubBackbone(), stats_bundle=b)
    assert_true(m.stats_bundle.future is None, "model kept a future-less bundle")

    # but a bundle WITH future statistics must still be DINO-checked
    b2 = fit_bundle(cfg, action_rows=action_rows,
                    future_rows=np.random.RandomState(4).randn(2, 1, 1, 196, 384),
                    dino={"model_path": "/other/dino", "config_hash": "x",
                          "hidden_dim": 384, "num_patches": 196,
                          "num_special_tokens": 5, "image_size": 224, "patch_size": 16},
                    cameras_used=[str(cfg.data.camera_order[0])], camera_index=0)
    try:
        resolve_stats_bundle(cfg, stats_bundle=b2, require_future=False)
        raise TestFailure("future-carrying bundle skipped the DINO identity check")
    except TestFailure:
        raise
    except StatsError as exc:
        assert_true("dino" in str(exc).lower(), str(exc))


# ---------------------------------------------------------------------------
# P0-4: synthetic refusal / DINO config identity / fingerprint
# ---------------------------------------------------------------------------
def test_synthetic_stats_refused_on_real_backend_by_default():
    from energywam.data.energy_dataset import resolve_stats_bundle
    from energywam.data.standardization import StatsError

    cfg = real_cfg("bi")
    rows = np.random.RandomState(5).randn(24, cfg.data.action_dim)
    b = fit_bundle(cfg, action_rows=rows,
                   future_rows=np.random.RandomState(6).randn(2, 1, 1, 196, 384),
                   dino={"model_path": cfg.backbone.dino_path, "config_hash": "x",
                         "hidden_dim": 384, "num_patches": 196,
                         "num_special_tokens": 5, "image_size": 224, "patch_size": 16},
                   cameras_used=[str(cfg.data.camera_order[0])], camera_index=0)

    try:
        resolve_stats_bundle(cfg, stats_bundle=b, require_future=True)
        raise TestFailure("synthetic statistics accepted on the real backend by default")
    except TestFailure:
        raise
    except StatsError as exc:
        assert_true("synthetic" in str(exc).lower(), str(exc))

    # explicit switch (documented verification boundary) accepts them
    cfg.data.allow_synthetic_stats = True
    resolve_stats_bundle(cfg, stats_bundle=b, require_future=True)

    # the tiny backend keeps working with synthetic statistics (test mode)
    tcfg = tiny_paper_cfg()
    trows = np.random.RandomState(7).randn(24, tcfg.backbone.tiny_action_dim)
    tb = fit_bundle(tcfg, action_rows=trows,
                    future_rows=np.random.RandomState(8).randn(2, 1, 1, 1, 8))
    resolve_stats_bundle(tcfg, stats_bundle=tb, require_future=True)


def test_dino_config_identity_checked_beyond_model_path():
    from energywam.data.energy_dataset import resolve_stats_bundle
    from energywam.data.standardization import StatsError

    cfg = real_cfg("bi")
    rows = np.random.RandomState(9).randn(24, cfg.data.action_dim)
    b = fit_bundle(cfg, action_rows=rows,
                   future_rows=np.random.RandomState(10).randn(2, 1, 1, 196, 384),
                   dino={"model_path": cfg.backbone.dino_path, "config_hash": "x",
                         "hidden_dim": 384, "num_patches": 196,
                         "num_special_tokens": 5, "image_size": 224, "patch_size": 16},
                   cameras_used=[str(cfg.data.camera_order[0])], camera_index=0)
    cfg.data.allow_synthetic_stats = True
    resolve_stats_bundle(cfg, stats_bundle=b, require_future=True)  # consistent

    for field, value in (("dino_num_patches", 144), ("dino_image_size", 192),
                         ("dino_patch_size", 8), ("dino_num_special_tokens", 4),
                         ("dino_hidden_dim", 768)):
        bad = real_cfg("bi")
        bad.data.allow_synthetic_stats = True
        setattr(bad.backbone, field, value)
        # keep the geometry dim consistent so the DINO check is what fires
        if field == "dino_hidden_dim":
            b2 = fit_bundle(cfg, action_rows=rows,
                            future_rows=np.random.RandomState(11).randn(2, 1, 1, 196, 384),
                            dino={"model_path": cfg.backbone.dino_path, "config_hash": "x",
                                  "hidden_dim": 384, "num_patches": 196,
                                  "num_special_tokens": 5, "image_size": 224, "patch_size": 16},
                            cameras_used=[str(cfg.data.camera_order[0])], camera_index=0)
            try:
                resolve_stats_bundle(bad, stats_bundle=b2, require_future=True)
                raise TestFailure(f"{field} change accepted")
            except TestFailure:
                raise
            except StatsError as exc:
                assert_true("dino" in str(exc).lower() or "future_dim" in str(exc).lower(), str(exc))
            continue
        try:
            resolve_stats_bundle(bad, stats_bundle=b, require_future=True)
            raise TestFailure(f"{field} change accepted (same model_path)")
        except TestFailure:
            raise
        except StatsError as exc:
            assert_true("dino" in str(exc).lower(), str(exc))


def test_missing_fingerprint_rejected_from_payload():
    from energywam.data.standardization import NormalizationBundle, StatsError

    cfg = tiny_paper_cfg()
    rows = np.random.RandomState(12).randn(24, cfg.backbone.tiny_action_dim)
    b = fit_bundle(cfg, action_rows=rows,
                   future_rows=np.random.RandomState(13).randn(2, 1, 1, 1, 8))
    payload = b.to_payload()
    assert_true("fingerprint" in payload, "payload must carry a fingerprint")

    stripped = {k: v for k, v in payload.items() if k != "fingerprint"}
    try:
        NormalizationBundle.from_payload(stripped)
        raise TestFailure("payload without fingerprint was accepted")
    except TestFailure:
        raise
    except StatsError as exc:
        assert_true("fingerprint" in str(exc).lower(), str(exc))

    # direct object construction (test API) is unaffected
    NormalizationBundle(profile=b.profile, epsilon=b.epsilon, action=b.action,
                        future=b.future, provenance=dict(b.provenance))


# ---------------------------------------------------------------------------
# P1: checkpoint payload buffers must match the embedded bundle
# ---------------------------------------------------------------------------
def test_checkpoint_payload_buffer_tamper_rejected():
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.training.checkpoint import restore
    from energywam.training.trainer import EnergyWAMTrainer

    cfg = tiny_paper_cfg()
    rows = np.random.RandomState(14).randn(24, cfg.backbone.tiny_action_dim)
    bundle = fit_bundle(cfg, action_rows=rows,
                        future_rows=np.random.RandomState(15).randn(2, 1, 1, 1, 8))
    with tempfile.TemporaryDirectory() as d:
        cfg.trainer.output_dir = d
        torch.manual_seed(0)
        model = QwenEnergyWAM(cfg, stats_bundle=bundle)
        tr = EnergyWAMTrainer(cfg, model)
        path = tr.save("ok.pt")
        payload = torch.load(path, map_location="cpu", weights_only=False)

        # sanity: the untampered payload restores into a fresh model
        torch.manual_seed(0)
        m2 = QwenEnergyWAM(cfg, stats_bundle=bundle)
        restore(copy.deepcopy(payload), model=m2, mode="resume", restore_rng=False)

        # tamper ONLY a payload norm buffer (meta fingerprint untouched)
        tampered = copy.deepcopy(payload)
        key = "norm_action_mean"
        assert_true(key in tampered["model"], f"missing {key} in payload model keys")
        tampered["model"][key] = tampered["model"][key] + 1.0
        torch.manual_seed(0)
        m3 = QwenEnergyWAM(cfg, stats_bundle=bundle)
        try:
            restore(tampered, model=m3, mode="resume", restore_rng=False)
            raise TestFailure("tampered payload norm buffer was accepted")
        except TestFailure:
            raise
        except (ValueError, RuntimeError) as exc:
            assert_true("buffer" in str(exc).lower() or "norm_" in str(exc), str(exc))

        # wrapper prefixes (e.g. StarVLA adapter) are matched by suffix
        from energywam.integration.starvla import EnergyWAMFrameworkAdapter

        wrapped = copy.deepcopy(payload)
        wrapped["model"] = {f"model.{k}": v for k, v in wrapped["model"].items()}
        torch.manual_seed(0)
        m4 = EnergyWAMFrameworkAdapter(QwenEnergyWAM(cfg, stats_bundle=bundle), cfg)
        restore(wrapped, model=m4, mode="resume", restore_rng=False)


TESTS = [
    ("training_index_restricted_to_bundle_train_episodes", test_training_index_restricted_to_bundle_train_episodes),
    ("training_binding_rejects_bad_identity_and_episodes", test_training_binding_rejects_bad_identity_and_episodes),
    ("eval_report_never_claims_heldout_without_selection", test_eval_report_never_claims_heldout_without_selection),
    ("build_sample_current_and_future_use_same_selected_camera", test_build_sample_current_and_future_use_same_selected_camera),
    ("stats_fit_and_training_read_same_cameras_with_numbers", test_stats_fit_and_training_read_same_cameras_with_numbers),
    ("bundle_camera_selection_mismatch_rejected", test_bundle_camera_selection_mismatch_rejected),
    ("action_only_bundle_skips_future_dino_identity", test_action_only_bundle_skips_future_dino_identity),
    ("synthetic_stats_refused_on_real_backend_by_default", test_synthetic_stats_refused_on_real_backend_by_default),
    ("dino_config_identity_checked_beyond_model_path", test_dino_config_identity_checked_beyond_model_path),
    ("missing_fingerprint_rejected_from_payload", test_missing_fingerprint_rejected_from_payload),
    ("checkpoint_payload_buffer_tamper_rejected", test_checkpoint_payload_buffer_tamper_rejected),
]
