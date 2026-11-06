"""Regression tests for the REVIEW_EXTRA / ROUND_2 items.

Each test is written so it fails against the pre-fix behaviour:

* ``future_offsets`` were absolute frame indices when the chunk started at
  ``t > 0`` (they must be relative; absolute frames go in a separate field);
* ``load_state_dict(strict=False)`` silently accepted a mismatched checkpoint;
* an optimiser ``reset`` policy left existing AdamW moments in place;
* RNG state restored through ``map_location`` could land on GPU;
* ``assert_close`` let NaN comparisons pass (``NaN > tol`` is False);
* declared config fields (``camera_index`` / ``camera_order``) were not wired.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import torch

from .harness import TestFailure, assert_close, assert_true
from .harness import _self_check_nan_must_fail

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# harness meta-test (must come first — everything else relies on it)
# ---------------------------------------------------------------------------
def test_harness_rejects_nan():
    assert_true(_self_check_nan_must_fail(),
                "assert_close does not reject NaN — comparisons can silently pass")


def test_harness_preserves_float64():
    a = np.array([1.0], dtype=np.float64)
    b = np.array([1.0 + 1e-13], dtype=np.float64)
    assert_close(a, b, 1e-12, "float64 pair should be distinguishable at 1e-12")
    try:
        assert_close(a, b, 1e-15, "float64 pair must be distinguishable at 1e-15")
    except TestFailure:
        pass
    else:
        raise TestFailure("assert_close lost float64 resolution")


# ---------------------------------------------------------------------------
# relative future offsets
# ---------------------------------------------------------------------------
def test_future_offsets_are_relative_for_nonzero_start():
    from energywam.data.episode_dataset import DatasetSpec, build_sample
    from energywam.data.normalization import NormStats

    spec = DatasetSpec(action_horizon=8, future_offsets=(3, 5, 8), num_cameras=1)

    class FakeReader:
        dataset_identity = "fake/ep"
        camera_keys = ["cam0"]
        episodes = [{"episode_index": 0, "length": 40, "tasks": ["t"]}]

        def episode_length(self, ep):
            return 40

        def read_actions(self, ep):
            return np.zeros((40, 3), dtype="float32")

        def read_states(self, ep):
            return np.zeros((40, 4), dtype="float32")

        def read_language(self, ep):
            return "t"

        def read_frame(self, ep, cam, idx):
            return np.zeros((8, 8, 3), dtype="uint8")

    start = 7  # non-zero episode start — the bug only shows here
    s = build_sample(FakeReader(), 0, start, spec,
                     action_stats=NormStats(mode="none"))
    assert_true(s is not None, "sample is None")
    assert_true(s.future_offsets == [3, 5, 8],
                f"future_offsets must be RELATIVE, got {s.future_offsets}")
    assert_true(s.future_frames_abs == [start + 3, start + 5, start + 8],
                f"future_frames_abs must be ABSOLUTE, got {s.future_frames_abs}")
    assert_true(s.future_frames_abs != s.future_offsets,
                "relative and absolute offsets coincide for a non-zero start")

    # and the encodings must consume the relative values
    enc = torch.nn.Module()
    from energywam.energy.encodings import CandidateEncoding

    ce = CandidateEncoding(8, {"action_horizon": 8})
    rel = torch.tensor([[3, 5, 8]], dtype=torch.long)
    e = ce.future(1, 3, 1, 4, torch.device("cpu"), torch.float32, offsets=rel)
    assert_true(e.shape == (1, 12, 8), f"encoding shape {tuple(e.shape)}")


# ---------------------------------------------------------------------------
# strict checkpoint restore
# ---------------------------------------------------------------------------
def test_checkpoint_restore_is_strict():
    from energywam.config import load_config
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.training.checkpoint import CheckpointMeta, load_checkpoint, restore, save_checkpoint

    cfg = load_config(ROOT / "configs" / "tiny_overfit.yaml")
    torch.manual_seed(0)
    m = QwenEnergyWAM(cfg)
    sd = copy.deepcopy(m.state_dict())

    with tempfile.TemporaryDirectory() as d:
        path = save_checkpoint(f"{d}/c.pt", model=m, optimizer=None, scheduler=None,
                               cfg=cfg, meta=CheckpointMeta(stage="bi", global_step=1))
        payload = load_checkpoint(path)
        # corrupt one key so the checkpoint no longer matches the network
        k = next(iter(payload["model"]))
        payload["model"][k] = payload["model"][k] + 1.0
        # (still the same shape, so this would pass a naive shape check but the
        #  point below is a MISSING key)
        del payload["model"][k]
        m2, _ = QwenEnergyWAM(cfg), cfg
        try:
            restore(payload, model=m2, optimizer=None, scheduler=None, mode="resume")
        except RuntimeError as exc:
            assert_true("Missing key" in str(exc) or "missing" in str(exc).lower()
                        or "Unexpected" in str(exc) or "size mismatch" in str(exc).lower(),
                        f"unhelpful strict-load error: {exc}")
        else:
            raise TestFailure("strict restore silently accepted a mismatched checkpoint")


def test_optimizer_reset_clears_existing_state():
    from energywam.config import load_config
    from energywam.data.tiny_data import make_tiny_batch
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.training.checkpoint import CheckpointMeta, load_checkpoint, restore, save_checkpoint
    from energywam.training.trainer import EnergyWAMTrainer

    cfg = load_config(ROOT / "configs" / "tiny_overfit.yaml")
    torch.manual_seed(0)
    m = QwenEnergyWAM(cfg)
    tr = EnergyWAMTrainer(cfg, m)
    for i in range(2):
        tr.train_step(make_tiny_batch(cfg, batch_size=2, seed=i))
    assert_true(len(tr.optimizer.state) > 0, "optimiser accumulated no state")

    with tempfile.TemporaryDirectory() as d:
        path = tr.save("c.pt")
        payload = load_checkpoint(path)
        meta = restore(payload, model=m, optimizer=tr.optimizer, scheduler=tr.scheduler,
                       mode="switch", optimizer_policy="reset", scheduler_policy="reset")
        assert_true(meta.optimizer_policy == "reset", "policy not honoured")
        assert_true(len(tr.optimizer.state) == 0,
                    f"optimizer reset left {len(tr.optimizer.state)} states behind")


def test_rng_state_restores_to_cpu():
    from energywam.config import load_config
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.training.checkpoint import (
        CheckpointMeta, _set_rng_state, load_checkpoint, restore, save_checkpoint,
    )

    cfg = load_config(ROOT / "configs" / "tiny_overfit.yaml")
    torch.manual_seed(0)
    m = QwenEnergyWAM(cfg)

    # Snapshot the stream at a known point, then advance it.
    torch.manual_seed(4242)
    with tempfile.TemporaryDirectory() as d:
        path = save_checkpoint(
            f"{d}/c.pt", model=m, optimizer=None, scheduler=None,
            cfg=cfg, meta=CheckpointMeta(stage="bi", global_step=0),
        )
        ref = torch.randn(4)          # the stream *after* the snapshot
        _ = torch.randn(11)           # advance further
        payload = load_checkpoint(path)

        st = payload["rng"]["torch"]
        assert_true(isinstance(st, torch.Tensor), "torch RNG state is not a tensor")
        assert_true(st.device.type == "cpu",
                    f"RNG state must be a CPU tensor, got {st.device}")
        assert_true(st.dtype == torch.uint8,
                    f"RNG state must be a ByteTensor, got {st.dtype}")

        # restore must put the stream back to the snapshot point
        restore(payload, model=m, optimizer=None, scheduler=None, mode="resume")
        out = torch.randn(4)
        assert_close(out, ref, 0.0, "RNG restore did not reproduce the stream")

        # _set_rng_state must force CPU even if map_location moved the tensor
        moved = st.clone()
        _set_rng_state({"torch": moved})
        out2 = torch.randn(4)
        assert_close(out2, ref, 0.0, "_set_rng_state did not restore the stream")


# ---------------------------------------------------------------------------
# config field wiring
# ---------------------------------------------------------------------------
def test_camera_index_and_order_are_wired():
    from energywam.data.episode_dataset import DatasetSpec, build_sample
    from energywam.data.energy_dataset import EnergyWAMEpisodeDataset
    from energywam.data.normalization import NormStats

    class FakeReader:
        dataset_identity = "fake/ep"
        camera_keys = ["cam_left", "cam_right", "cam_wrist"]
        episodes = [{"episode_index": 0, "length": 40, "tasks": ["t"]}]

        def episode_length(self, ep):
            return 40

        def read_actions(self, ep):
            return np.zeros((40, 3), dtype="float32")

        def read_states(self, ep):
            return np.zeros((40, 4), dtype="float32")

        def read_language(self, ep):
            return "t"

        def read_frame(self, ep, cam, idx):
            arr = np.zeros((8, 8, 3), dtype="uint8")
            arr[..., 0] = {"cam_left": 10, "cam_right": 20, "cam_wrist": 30}[cam]
            return arr

    # camera_index=1 selects cam_right as the single used camera
    spec = DatasetSpec(action_horizon=8, future_offsets=(3, 5, 8),
                       num_cameras=1, camera_index=1)
    s = build_sample(FakeReader(), 0, 0, spec, action_stats=NormStats(mode="none"))
    assert_true(s.meta["cameras"] == ["cam_right"],
                f"camera_index not honoured: {s.meta['cameras']}")
    assert_true(int(s.images[0][0, 0, 0]) == 20,
                "read_frame used the wrong camera")

    # explicit camera_keys order wins over the reader order
    spec2 = DatasetSpec(action_horizon=8, future_offsets=(3, 5, 8),
                        num_cameras=2, camera_index=0,
                        camera_keys=("cam_wrist", "cam_left"))
    s2 = build_sample(FakeReader(), 0, 0, spec2, action_stats=NormStats(mode="none"))
    assert_true(s2.meta["cameras"] == ["cam_wrist", "cam_left"],
                f"camera_keys order not honoured: {s2.meta['cameras']}")

    # out-of-range index must raise rather than silently wrap
    spec3 = DatasetSpec(action_horizon=8, future_offsets=(3, 5, 8),
                        num_cameras=1, camera_index=5)
    try:
        build_sample(FakeReader(), 0, 0, spec3, action_stats=NormStats(mode="none"))
    except ValueError as exc:
        assert_true("camera_index" in str(exc), str(exc))
    else:
        raise TestFailure("out-of-range camera_index did not raise")


def test_state_sincos_expansion_is_wired():
    from energywam.data.energy_dataset import sincos_state

    q = np.arange(16, dtype="float32") / 10.0
    out = sincos_state(q, 32)
    assert_true(out.shape == (32,), f"shape {out.shape}")
    assert_close(out[:16], np.sin(q), 1e-7, "sin block wrong")
    assert_close(out[16:32], np.cos(q), 1e-7, "cos block wrong")
    assert_close(sincos_state(np.zeros(16, dtype="float32"), 32),
                 np.concatenate([np.zeros(16), np.ones(16)]), 0.0, "zero-state semantics")
    try:
        sincos_state(q, 32 + 1)
    except ValueError:
        pass
    else:
        raise TestFailure("odd out_dim accepted for sin/cos expansion")


# ---------------------------------------------------------------------------
# finite assertions on the main path
# ---------------------------------------------------------------------------
def test_forward_rejects_nonfinite_loss_and_grad():
    from energywam.config import load_config
    from energywam.data.tiny_data import make_tiny_batch
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.training.trainer import EnergyWAMTrainer

    cfg = load_config(ROOT / "configs" / "tiny_overfit.yaml")
    torch.manual_seed(0)
    m = QwenEnergyWAM(cfg)
    tr = EnergyWAMTrainer(cfg, m)

    # a NaN parameter must be caught before it reaches the optimiser
    p = next(p for p in m.parameters() if p.requires_grad)
    saved = p.data.clone()
    p.data.fill_(float("nan"))
    try:
        tr.train_step(make_tiny_batch(cfg, batch_size=2, seed=0))
    except RuntimeError as exc:
        assert_true("non-finite" in str(exc), f"unhelpful error: {exc}")
    else:
        raise TestFailure("a NaN parameter produced no error")
    finally:
        p.data.copy_(saved)


def test_model_inputs_are_finite_after_prepare():
    from energywam.config import load_config
    from energywam.data.tiny_data import make_tiny_batch
    from energywam.models.energy_wam import QwenEnergyWAM

    cfg = load_config(ROOT / "configs" / "tiny_overfit.yaml")
    torch.manual_seed(0)
    m = QwenEnergyWAM(cfg)
    b = make_tiny_batch(cfg, batch_size=2, seed=3)
    prep = m._prepare_batch(b)
    for k, v in prep.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            assert_true(bool(torch.isfinite(v).all()), f"non-finite value in prepared[{k}]")
    # and the model's own weights
    for n, p in m.named_parameters():
        assert_true(bool(torch.isfinite(p).all()), f"non-finite weight {n}")


TESTS = [
    ("harness_rejects_nan", test_harness_rejects_nan),
    ("harness_preserves_float64", test_harness_preserves_float64),
    ("future_offsets_are_relative_for_nonzero_start", test_future_offsets_are_relative_for_nonzero_start),
    ("checkpoint_restore_is_strict", test_checkpoint_restore_is_strict),
    ("optimizer_reset_clears_existing_state", test_optimizer_reset_clears_existing_state),
    ("rng_state_restores_to_cpu", test_rng_state_restores_to_cpu),
    ("camera_index_and_order_are_wired", test_camera_index_and_order_are_wired),
    ("state_sincos_expansion_is_wired", test_state_sincos_expansion_is_wired),
    ("forward_rejects_nonfinite_loss_and_grad", test_forward_rejects_nonfinite_loss_and_grad),
    ("model_inputs_are_finite_after_prepare", test_model_inputs_are_finite_after_prepare),
]
