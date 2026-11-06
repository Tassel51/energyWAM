"""``scripts/fit_normalization_stats.py`` — CLI behaviour and a REAL fit run.

The real-path test writes a small on-disk LeRobot dataset with real videos
(:mod:`tests.synthetic_lerobot`) and fits the bundle with the **real frozen
DINOv3** (no Qwen is ever loaded).  The resulting action statistics are
compared against hand-enumerated window statistics — the same train-only
counting rule the CLI documents.  Everything produced here is synthetic data,
labelled as such; it is not RoboCasa365.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .harness import TestFailure, assert_close, assert_true

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run_cli(args, expect_ok=True):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([PY, str(ROOT / "scripts" / "fit_normalization_stats.py"), *args],
                       capture_output=True, text=True, timeout=600, env=env, cwd=str(ROOT))
    if expect_ok and r.returncode != 0:
        raise TestFailure(f"CLI failed:\n{r.stdout}\n{r.stderr}")
    if not expect_ok and r.returncode == 0:
        raise TestFailure(f"CLI unexpectedly succeeded:\n{r.stdout}\n{r.stderr}")
    return r


# ---------------------------------------------------------------------------
# 1. guards
# ---------------------------------------------------------------------------
def test_cli_rejects_non_train_split():
    for split in ("val", "test", "VAL", "evaluation"):
        r = _run_cli([
            "--config", "configs/tiny_paper_standardized.yaml",
            "--split", split, "--synthetic-tiny", "--out", "/tmp/should_not_exist.json",
        ], expect_ok=False)
        assert_true("train" in (r.stdout + r.stderr).lower(),
                    f"split={split}: unhelpful message {r.stdout}{r.stderr}")


def test_cli_refuses_overwrite_without_flag():
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "bundle.json")
        _run_cli(["--config", "configs/tiny_paper_standardized.yaml", "--split", "train",
                  "--synthetic-tiny", "--out", out])
        r = _run_cli(["--config", "configs/tiny_paper_standardized.yaml", "--split", "train",
                      "--synthetic-tiny", "--out", out], expect_ok=False)
        assert_true("overwrite" in (r.stdout + r.stderr).lower(), r.stdout + r.stderr)
        _run_cli(["--config", "configs/tiny_paper_standardized.yaml", "--split", "train",
                  "--synthetic-tiny", "--out", out, "--overwrite"])


def test_cli_synthetic_tiny_marks_synthetic():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "bundle.json"
        _run_cli(["--config", "configs/tiny_paper_standardized.yaml", "--split", "train",
                  "--synthetic-tiny", "--out", str(out)])
        payload = json.loads(out.read_text())
        assert_true(payload["provenance"]["synthetic"] is True,
                    "synthetic bundle must be marked synthetic")
        assert_true(payload["provenance"]["coverage"]["kind"] == "synthetic_train_split",
                    payload["provenance"]["coverage"])
        assert_true(payload["fingerprint"] and payload["epsilon"] == 1e-6, "bundle fields")


# ---------------------------------------------------------------------------
# 2. real fit run on an on-disk synthetic LeRobot dataset (real frozen DINO)
# ---------------------------------------------------------------------------
def _hand_action_stats(root: Path, dataset_name: str, episodes, H: int, offs, Da: int):
    """Enumerate the same training windows and compute the expected statistics."""
    from tests.synthetic_lerobot import synthetic_action

    vals = []
    L = int(json.loads((root / dataset_name / "meta" / "episodes.jsonl").read_text().splitlines()[0])["length"])
    from energywam.data.future_offsets import plan_future_frames

    for ep in episodes:
        for t in range(L):
            plan = plan_future_frames(t, L, H, offs, require_full_horizon=True)
            if not plan.valid:
                continue
            for k in range(H):
                if k < len(plan.action_valid) and plan.action_valid[k]:
                    vals.append(synthetic_action(ep, t + k, Da))
    a = np.asarray(vals, dtype=np.float64)
    return a.mean(axis=0), a.std(axis=0), a.shape[0]  # population std


def test_cli_real_fit_on_synthetic_dataset():
    from energywam.config import load_config
    from energywam.data.standardization import load_bundle

    cfg = load_config(ROOT / "configs" / "legacy" / "real_smoke.yaml")
    Da = int(cfg.data.action_dim)
    H = int(cfg.data.action_horizon)
    offs = [int(x) for x in cfg.data.future_offsets]

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        from tests.synthetic_lerobot import write_manifest, write_synthetic_lerobot

        ds_root = write_synthetic_lerobot(d, num_episodes=3, episode_length=24,
                                          action_dim=Da, state_dim=16)
        identity = f"{ds_root.name}/SyntheticRobot"
        train_m = write_manifest(d / "train.json", split="train", episodes=[0],
                                 dataset_identity=identity)
        val_m = write_manifest(d / "val.json", split="val", episodes=[1],
                               dataset_identity=identity)

        set_args = [f"data.data_root_dir={d}", f"data.data_mix={ds_root.name}"]

        # --- subset fit (smoke): coverage must say 'subset' ---
        out_sub = d / "subset.json"
        _run_cli([
            "--config", "configs/legacy/real_smoke.yaml", "--split", "train",
            "--episodes-manifest", str(train_m), "--val-manifest", str(val_m),
            "--out", str(out_sub), "--max-samples", "6", "--set", *set_args,
        ])
        b = load_bundle(out_sub)
        assert_true(b.provenance["coverage"]["kind"] == "subset",
                    f"max_samples must mark subset coverage: {b.provenance['coverage']}")
        assert_true(b.action.dim == Da, f"action dim {b.action.dim} != {Da}")
        assert_true(b.future.dim == 384, f"future dim {b.future.dim} != 384 (DINOv3-ViT-S)")
        assert_true(b.provenance["dataset_identity"] == identity,
                    b.provenance["dataset_identity"])
        assert_true(b.provenance["train_episodes"]["manifest_sha256"], "manifest hash recorded")
        assert_true(b.provenance["camera_order"] == ["robot0_agentview_left"],
                    b.provenance["camera_order"])
        assert_true(b.provenance["future_offsets"] == offs, b.provenance["future_offsets"])
        assert_true(b.provenance["dino_preprocess_version"] == cfg.data.dino_preprocess_version,
                    b.provenance["dino_preprocess_version"])
        assert_true(b.provenance["dino"]["model_path"] == cfg.backbone.dino_path,
                    b.provenance["dino"])
        assert_true(b.provenance["split"] == "train", b.provenance["split"])
        assert_true(b.fingerprint == load_bundle(out_sub).fingerprint, "fingerprint stable")

        # --- full fit over episode 0: hand-computed action statistics ---
        out_full = d / "full.json"
        _run_cli([
            "--config", "configs/legacy/real_smoke.yaml", "--split", "train",
            "--episodes-manifest", str(train_m), "--val-manifest", str(val_m),
            "--out", str(out_full), "--set", *set_args,
        ])
        bf = load_bundle(out_full)
        assert_true(bf.provenance["coverage"]["kind"] == "full_train_split",
                    bf.provenance["coverage"])
        want_mean, want_std, n_rows = _hand_action_stats(d, ds_root.name, [0], H, offs, Da)
        assert_close(bf.action.mean, want_mean, 1e-9, "hand-computed action mean")
        assert_close(bf.action.std, want_std, 1e-9, "hand-computed action std (population)")
        assert_close(bf.action.count, [float(n_rows)] * Da, 0, "action valid-element counts")
        # future counts: windows * F * V * P per channel
        n_windows = bf.provenance["coverage"]["num_windows"]
        want_f = float(n_windows * cfg.data.num_future_frames * 1 * cfg.backbone.dino_num_patches)
        assert_close(bf.future.count, [want_f] * 384, 0, "future per-channel counts")

        # --- action-only fit: no future statistics at all ---
        out_a = d / "action_only.json"
        _run_cli([
            "--config", "configs/legacy/real_smoke.yaml", "--split", "train",
            "--episodes-manifest", str(train_m), "--out", str(out_a),
            "--no-future", "--set", *set_args,
        ])
        ba = load_bundle(out_a)
        assert_true(ba.future is None, "no-future fit must carry no future statistics")
        assert_close(ba.action.mean, want_mean, 1e-9, "action-only fit gives the same action stats")

        # --- guards on the real path ---
        r = _run_cli([
            "--config", "configs/legacy/real_smoke.yaml", "--split", "val",
            "--episodes-manifest", str(train_m), "--out", str(d / "x1.json"),
            "--set", *set_args,
        ], expect_ok=False)
        assert_true("train" in (r.stdout + r.stderr).lower(), r.stdout + r.stderr)

        bad_m = write_manifest(d / "train_bad.json", split="train", episodes=[0, 9],
                               dataset_identity=identity)
        r = _run_cli([
            "--config", "configs/legacy/real_smoke.yaml", "--split", "train",
            "--episodes-manifest", str(bad_m), "--out", str(d / "x2.json"),
            "--set", *set_args,
        ], expect_ok=False)
        assert_true("not in the dataset" in (r.stdout + r.stderr), r.stdout + r.stderr)

        r = _run_cli([
            "--config", "configs/legacy/real_smoke.yaml", "--split", "train",
            "--episodes-manifest", str(train_m), "--val-manifest", str(d / "val.json"),
            "--out", str(d / "x3.json"), "--max-samples", "2",
            "--set", *[f"data.data_root_dir={d}", f"data.data_mix={ds_root.name}"],
        ], expect_ok=True)  # episode 0 vs val episode 1 -> no overlap, must pass
        assert_true(r.returncode == 0, r.stdout + r.stderr)

        overlap_m = write_manifest(d / "train_ov.json", split="train", episodes=[0, 1],
                                   dataset_identity=identity)
        r = _run_cli([
            "--config", "configs/legacy/real_smoke.yaml", "--split", "train",
            "--episodes-manifest", str(overlap_m), "--val-manifest", str(val_m),
            "--out", str(d / "x4.json"), "--max-samples", "2", "--set", *set_args,
        ], expect_ok=False)
        assert_true("overlap" in (r.stdout + r.stderr), r.stdout + r.stderr)


def test_cli_bundle_rejected_on_geometry_mismatch():
    """A fitted bundle must be refused when the run configuration disagrees."""
    from energywam.config import load_config
    from energywam.data.energy_dataset import resolve_stats_bundle
    from energywam.data.standardization import StatsError
    from energywam.data.tiny_data import synthetic_train_bundle

    cfg = load_config(ROOT / "configs" / "tiny_paper_standardized.yaml")
    bundle = synthetic_train_bundle(cfg, n_windows=16, seed=3)
    cfg.data.norm_stats_bundle_path = None
    resolve_stats_bundle(cfg, stats_bundle=bundle, require_future=True)  # consistent: OK

    bad = load_config(ROOT / "configs" / "tiny_paper_standardized.yaml")
    bad.data.norm_stats_bundle_path = None
    bad.data.camera_order = ("another_camera",)
    try:
        resolve_stats_bundle(bad, stats_bundle=bundle, require_future=True)
        raise TestFailure("camera-order mismatch was accepted")
    except TestFailure:
        raise
    except StatsError as exc:
        assert_true("camera order mismatch" in str(exc), str(exc))

    # missing bundle file -> loud error, no fallback
    miss = load_config(ROOT / "configs" / "tiny_paper_standardized.yaml")
    miss.data.norm_stats_bundle_path = str(ROOT / "no_such_bundle.json")
    try:
        resolve_stats_bundle(miss, require_future=True)
        raise TestFailure("missing bundle file was accepted")
    except TestFailure:
        raise
    except StatsError as exc:
        assert_true("not found" in str(exc), str(exc))


def test_real_backend_missing_stats_fails_before_backbone():
    """Fail fast: missing statistics error BEFORE the 4B/DINO is constructed."""
    from energywam.config import load_config
    from energywam.data.standardization import StatsError
    from energywam.models.energy_wam import QwenEnergyWAM

    cfg = load_config(ROOT / "configs" / "bi.yaml")  # paper profile, real backend
    cfg.data.norm_stats_bundle_path = "/nonexistent/bundle.json"
    # if stats validation did not run first, construction would fail on the
    # MISSING WEIGHTS instead (FileNotFoundError mentioning Qwen3-VL)
    cfg.backbone.base_vlm = "/nonexistent/Qwen3-VL-4B-Instruct"
    cfg.backbone.dino_path = "/nonexistent/dino"
    try:
        QwenEnergyWAM(cfg)
        raise TestFailure("real backend without statistics did not raise")
    except TestFailure:
        raise
    except StatsError as exc:
        msg = str(exc)
        assert_true("not found" in msg, msg)
        assert_true("Qwen3-VL" not in msg and "weights" not in msg,
                    f"stats check ran AFTER the backbone load: {msg}")


def test_cli_camera_selection_recorded_and_used():
    """The statistics CLI must fit on the SAME camera the training dataset reads."""
    from energywam.config import load_config
    from energywam.data.standardization import load_bundle

    from .synthetic_lerobot import write_manifest, write_synthetic_lerobot

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ds_root = write_synthetic_lerobot(
            d, num_episodes=2, episode_length=24, action_dim=12, state_dim=16,
            cameras=["cam_a", "cam_b"], dataset_name="two_cams",
        )
        identity = f"{ds_root.name}/SyntheticRobot"
        train_m = write_manifest(d / "train.json", split="train", episodes=[0],
                                 dataset_identity=identity)
        out = d / "bundle.json"
        _run_cli([
            "--config", "configs/legacy/real_smoke.yaml", "--split", "train",
            "--episodes-manifest", str(train_m), "--out", str(out),
            "--max-samples", "2",
            "--set", f"data.data_root_dir={d}", f"data.data_mix={ds_root.name}",
            "data.camera_order=[cam_a,cam_b]", "data.future_camera_index=1",
        ])
        b = load_bundle(out)
        assert_true(b.provenance["cameras_used"] == ["cam_b"],
                    f"CLI fitted on {b.provenance['cameras_used']}, expected ['cam_b']")
        assert_true(b.provenance["camera_index"] == 1, b.provenance["camera_index"])
        assert_true(b.provenance["num_cameras"] == 1, b.provenance["num_cameras"])

        # the training dataset with the SAME config reads the same camera keys
        from energywam.data.energy_dataset import EnergyWAMEpisodeDataset
        from energywam.data.episode_dataset import LeRobotEpisodeReader

        cfg = load_config(
            ROOT / "configs" / "legacy" / "real_smoke.yaml",
            overrides=[f"data.data_root_dir={d}", f"data.data_mix={ds_root.name}",
                       "data.camera_order=[cam_a,cam_b]", "data.future_camera_index=1"],
        )
        reader = LeRobotEpisodeReader(ds_root)
        seen = []
        orig = reader.read_frame

        def spy(ep, cam, idx):
            seen.append(str(cam))
            return orig(ep, cam, idx)

        reader.read_frame = spy
        ds = EnergyWAMEpisodeDataset(
            cfg, reader=reader, action_stats=b.action.to_norm_stats(),
            state_stats=None, dino=None, require_future=False, max_samples=2,
            episode_indices=[0],
        )
        _ = ds[0]
        assert_true(set(seen) == {"cam_b"},
                    f"training dataset read cameras {sorted(set(seen))}, expected ['cam_b']")


TESTS = [
    ("cli_rejects_non_train_split", test_cli_rejects_non_train_split),
    ("cli_refuses_overwrite_without_flag", test_cli_refuses_overwrite_without_flag),
    ("cli_synthetic_tiny_marks_synthetic", test_cli_synthetic_tiny_marks_synthetic),
    ("cli_real_fit_on_synthetic_dataset", test_cli_real_fit_on_synthetic_dataset),
    ("cli_bundle_rejected_on_geometry_mismatch", test_cli_bundle_rejected_on_geometry_mismatch),
    ("real_backend_missing_stats_fails_before_backbone", test_real_backend_missing_stats_fails_before_backbone),
    ("cli_camera_selection_recorded_and_used", test_cli_camera_selection_recorded_and_used),
]
