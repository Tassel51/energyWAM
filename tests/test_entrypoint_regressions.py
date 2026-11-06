"""Regression tests for the REAL CLI entry points (evaluate.py / serve.py) and
the trainer's scheduler wiring.

Each test fails against the pre-fix behaviour it pins down:

* ``scripts/evaluate.py`` fed ``predict_action`` example dicts with batch-style
  keys (``images``/``langs``/``states`` instead of ``image``/``lang``/``state``),
  left the demonstration target un-normalized, skipped the ``sincos`` state
  expansion, ignored ``data.data_mix`` and ``data.camera_order``, and decoded
  real future frames during offline evaluation;
* ``scripts/serve.py`` never passed action statistics to the predictor, so the
  server returned NORMALIZED actions while the protocol claimed raw units;
* ``trainer.warmup_steps`` was dead config (``scheduler=None`` by default), and
  scheduler resume/reset only shuffled counters — the optimiser's learning
  rate was never actually restored/reset.

None of this is a pretrained-4B validation: everything runs on the explicit
tiny backend (mock weights) on CPU.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from .harness import TestFailure, assert_close, assert_true
from .common import tiny_batch, tiny_cfg, tiny_model

ROOT = Path(__file__).resolve().parents[1]


def _load_serve_module():
    """Import ``scripts/serve.py`` by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("energywam_serve_entry", ROOT / "scripts" / "serve.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# fake LeRobot reader for the offline-eval example builder
# ---------------------------------------------------------------------------
class FakeReader:
    """Minimal reader with two named cameras and known action/state values.

    Records every ``read_frame`` call so a test can prove offline evaluation
    never asks for a future frame.
    """

    dataset_identity = "fake/eval"
    camera_keys = ["cam_left", "cam_right"]

    def __init__(self, length: int = 12, action_dim: int = 2, state_dim: int = 2):
        self.episodes = [{"episode_index": 0, "length": length, "tasks": ["open the drawer"]}]
        self.length = length
        self.read_calls = []
        self._actions = np.stack(
            [np.asarray([2.0 + 0.5 * t, 4.0 + 0.5 * t], dtype="float32") for t in range(length)]
        )
        self._states = np.tile(np.asarray([0.0, 0.5], dtype="float32"), (length, 1))

    def episode_length(self, ep):
        return self.length

    def read_actions(self, ep):
        return self._actions.copy()

    def read_states(self, ep):
        return self._states.copy()

    def read_language(self, ep):
        return "open the drawer"

    def read_frame(self, ep, cam, idx):
        self.read_calls.append((cam, int(idx)))
        arr = np.zeros((8, 8, 3), dtype="uint8")
        arr[..., 0] = {"cam_left": 10, "cam_right": 20}[cam]
        return arr


def _eval_cfg(stats_path: str):
    cfg = tiny_cfg(
        **{
            "data.state_normalization": "sincos",
            "data.state_dim": 4,
            "data.camera_order": ("cam_right", "cam_left"),
            "data.norm_stats_path": stats_path,
            "data.require_full_horizon": True,
        }
    )
    return cfg


def _write_action_stats(path: str):
    Path(path).write_text(json.dumps({"action": {"min": [2.0, 4.0], "max": [6.0, 8.0]}}))


# ---------------------------------------------------------------------------
# 1. offline-eval examples follow the predict_action contract
# ---------------------------------------------------------------------------
def test_eval_examples_follow_predict_contract():
    from energywam.data.energy_dataset import build_eval_examples

    with tempfile.TemporaryDirectory() as d:
        stats_file = str(Path(d) / "stats.json")
        _write_action_stats(stats_file)
        cfg = _eval_cfg(stats_file)
        reader = FakeReader()
        examples, action_stats = build_eval_examples(cfg, reader=reader, max_samples=2)

        assert_true(len(examples) == 2, f"max_samples not honoured: {len(examples)}")
        mn = np.asarray([2.0, 4.0], dtype="float32")
        mx = np.asarray([6.0, 8.0], dtype="float32")
        assert_close(action_stats.min, mn, 0.0, "action stats min not loaded")

        ex = examples[0]
        assert_true(
            set(ex) == {"image", "lang", "state", "action", "action_mask"},
            f"example keys must be the predict_action contract (image/lang/state), got {sorted(ex)}",
        )

        # camera order comes from data.camera_order ("cam_right" first), not
        # from the reader's default order
        assert_true(int(np.asarray(ex["image"][0])[0, 0, 0]) == 20,
                    "camera_order config not honoured: wrong camera frame selected")

        # state goes through the configured sincos expansion: [sin(q), cos(q)]
        q = np.asarray([0.0, 0.5], dtype="float32")
        expected_state = np.concatenate([np.sin(q), np.cos(q)]).astype("float32")
        assert_true(ex["state"].shape == (4,), f"state shape {ex['state'].shape}")
        assert_close(ex["state"], expected_state, 1e-6, "sincos state must be [sin(q), cos(q)]")

        # the demonstration target is normalised with the TRAINING statistics
        raw_chunk = reader._actions[:8]
        expected_action = 2.0 * (raw_chunk - mn) / (mx - mn) - 1.0
        assert_close(ex["action"], expected_action, 1e-6,
                     "eval target must be normalized with the training action stats")
        assert_true(not np.allclose(ex["action"], raw_chunk),
                    "eval target was left in raw action units")
        assert_close(ex["action_mask"], np.ones((8, 2), dtype="float32"), 0.0, "action mask")
        assert_true(ex["lang"] == "open the drawer", f"lang {ex['lang']!r}")

        # offline eval must NEVER decode a real future frame: the fake reader
        # only sees the two current-frame reads (t=0 and t=1)
        seen = sorted({idx for _, idx in reader.read_calls})
        assert_true(seen == [0, 1], f"future frames were read during eval: read_frame calls {seen}")

        # the example dicts must feed predict_action's _examples_to_batch
        # (image/lang/state singular).  The tiny model's cond_vec requirement is
        # disabled only so the real example-contract path can be checked.
        m, cfg = tiny_model(cfg)
        m.cfg.backbone.backend = "qwen3vl"
        batch = m._examples_to_batch(examples, require_targets=False)
        assert_true(batch["images"][0] is not None and len(batch["images"][0]) == 1,
                    f"images not carried through the example contract: {batch['images']}")
        assert_true(batch["langs"][0] == "open the drawer",
                    f"langs not carried through: {batch['langs']}")
        assert_close(batch["states"][0], expected_state, 1e-6,
                     "states not carried through the example contract")


def test_eval_examples_resolve_data_mix_root():
    from energywam.data.energy_dataset import resolve_dataset_root

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "robocasa365"
        mix = root / "robocasa365_open_drawer_target_human"
        (mix / "meta").mkdir(parents=True)
        (mix / "meta" / "info.json").write_text("{}")
        cfg = tiny_cfg(**{"data.data_root_dir": str(root),
                          "data.data_mix": "robocasa365_open_drawer_target_human"})
        assert_true(resolve_dataset_root(cfg) == mix,
                    "data_mix not resolved under data_root_dir")

        # without the mix directory the root itself is used (when it is a dataset)
        root2 = Path(d) / "plain_root"
        (root2 / "meta").mkdir(parents=True)
        (root2 / "meta" / "info.json").write_text("{}")
        cfg2 = tiny_cfg(**{"data.data_root_dir": str(root2), "data.data_mix": "missing_mix"})
        assert_true(resolve_dataset_root(cfg2) == root2, "data_root_dir fallback broken")


# ---------------------------------------------------------------------------
# 2. serve assembly really uses the checkpoint's meta.norm_stats
# ---------------------------------------------------------------------------
def test_serve_assembly_uses_checkpoint_norm_stats():
    from energywam.data.normalization import NormStats
    from energywam.inference.predictor import build_policy_server_adapter
    from energywam.training.checkpoint import load_checkpoint
    from energywam.training.trainer import EnergyWAMTrainer

    serve_mod = _load_serve_module()
    # action_dim=2 so the model output matches the [2,4]/[6,8] statistics below
    cfg = tiny_cfg(**{"data.action_dim": 2, "backbone.tiny_action_dim": 2})
    m, cfg = tiny_model(cfg, seed=1)
    m.norm_stats = {
        "task_a": NormStats(mode="min_max",
                            min=np.asarray([2.0, 4.0], dtype="float32"),
                            max=np.asarray([6.0, 8.0], dtype="float32"),
                            version="test_entrypoint_v1")
    }
    with tempfile.TemporaryDirectory() as d:
        cfg.trainer.output_dir = d
        tr = EnergyWAMTrainer(cfg, m)
        path = tr.save("c.pt")

        # --- entry assembly pulls the stats out of the CHECKPOINT ---
        predictor = serve_mod.build_serve_predictor(
            cfg, checkpoint_path=str(path), unnorm_key="task_a"
        )
        assert_true(predictor.action_stats is not None,
                    "serve assembly produced a predictor without action stats")
        assert_close(predictor.action_stats.min, [2.0, 4.0], 0.0, "stats min not restored from meta")
        assert_close(predictor.action_stats.max, [6.0, 8.0], 0.0, "stats max not restored from meta")
        assert_true(predictor.action_stats.mode == "min_max", f"mode {predictor.action_stats.mode}")
        assert_true(predictor.unnorm_key == "task_a", f"unnorm_key {predictor.unnorm_key!r}")
        assert_true(sorted(predictor.model.norm_stats) == ["task_a"], "model.norm_stats not restored")

        policy = build_policy_server_adapter(
            predictor, action_keys=["action.x"], state_keys=["state.x"],
            unnorm_key="task_a", obs_image_size=(224, 224),
        )
        meta = policy.metadata
        assert_true(meta["available_unnorm_keys"] == ["task_a"], f"metadata keys {meta}")
        assert_true(meta["default_unnorm_key"] == "task_a", f"metadata default {meta}")
        assert_true(meta["action_units"] == "raw_dataset_units",
                    f"metadata claims wrong units: {meta.get('action_units')}")
        assert_true(meta["physical_eval_valid"] is True, "raw actions must be marked physically evaluable")

        # --- min=[2,4], max=[6,8]: normalized 0 must map to raw [4,6] ---
        model = predictor.model
        model.predict_action = lambda examples, **kw: {
            "normalized_actions": np.zeros((1, 2, 2), dtype="float32")
        }
        reply = policy.predict_action(
            examples=[{"image": [], "lang": "", "state": np.zeros(4, dtype="float32")}]
        )
        assert_close(np.asarray(reply["actions"])[0, 0], [4.0, 6.0], 1e-6,
                     "normalized 0 must un-normalize to raw [4, 6] under min=[2,4], max=[6,8]")
        assert_true(reply["action_units"] == "raw_dataset_units", "reply must state its units")
        del model.predict_action  # back to the real tiny model

        # --- and the REAL model output is un-normalized with the CHECKPOINT stats ---
        payload = load_checkpoint(path)
        block = payload["meta"]["norm_stats"]["task_a"]
        mn = np.asarray(block["min"], dtype="float32")
        mx = np.asarray(block["max"], dtype="float32")
        from energywam.data.tiny_data import make_tiny_batch

        b = make_tiny_batch(cfg, batch_size=1, seed=5)
        reply2 = policy.predict_action(examples=[{"cond_vec": b["cond_vec"][0]}])
        norm = np.asarray(reply2["normalized_actions"], dtype="float32")
        expected = (norm + 1.0) / 2.0 * (mx - mn) + mn
        assert_close(np.asarray(reply2["actions"]), expected, 1e-6,
                     "served actions are not the checkpoint-stats un-normalization of the model output")
        assert_true(not np.allclose(np.asarray(reply2["actions"]), norm),
                    "server returned normalized actions unchanged (stats never applied)")


def test_serve_units_and_missing_stats_policy():
    from energywam.inference.predictor import build_policy_server_adapter
    from energywam.training.trainer import EnergyWAMTrainer

    serve_mod = _load_serve_module()

    # real backend without statistics must be refused at the entry
    cfg_real = tiny_cfg()
    cfg_real.backbone.backend = "qwen3vl"
    try:
        serve_mod.resolve_serve_action_stats(cfg_real, payload=None, unnorm_key=None)
    except RuntimeError as exc:
        assert_true("stat" in str(exc).lower(), f"unhelpful error: {exc}")
    else:
        raise TestFailure("real backend without action statistics was not refused")

    # ... including a checkpoint that carries no norm stats
    cfg = tiny_cfg()
    m, cfg = tiny_model(cfg, seed=2)
    with tempfile.TemporaryDirectory() as d:
        cfg.trainer.output_dir = d
        tr = EnergyWAMTrainer(cfg, m)
        path = tr.save("c.pt")
        cfg_real.trainer.output_dir = d
        try:
            serve_mod.build_serve_predictor(cfg_real, checkpoint_path=str(path))
        except RuntimeError as exc:
            assert_true("stat" in str(exc).lower(), f"unhelpful error: {exc}")
        else:
            raise TestFailure("real backend served a checkpoint with no norm stats")

        # tiny backend may serve without stats, but must say so explicitly
        cfg_real.backbone.backend = "tiny"
        predictor = serve_mod.build_serve_predictor(cfg_real, checkpoint_path=str(path))
        assert_true(predictor.action_stats is None, "unexpected stats")
        policy = build_policy_server_adapter(predictor, unnorm_key=None, obs_image_size=(224, 224))
        meta = policy.metadata
        assert_true(meta["action_units"] == "normalized",
                    f"missing stats must be marked normalized: {meta.get('action_units')}")
        assert_true(meta["physical_eval_valid"] is False,
                    "normalized outputs must not be marked physically evaluable")
        assert_true("NORMALIZED" in meta["action_units_note"],
                    f"units note must be explicit: {meta.get('action_units_note')}")

    # several keys and no explicit selection is an error, not a silent pick
    payload = {"meta": {"norm_stats": {
        "a": {"mode": "min_max", "min": [0.0], "max": [1.0]},
        "b": {"mode": "min_max", "min": [0.0], "max": [1.0]},
    }}}
    cfg2 = tiny_cfg()
    try:
        serve_mod.resolve_serve_action_stats(cfg2, payload=payload, unnorm_key=None)
    except ValueError as exc:
        assert_true("unnorm-key" in str(exc) or "unnorm_key" in str(exc), str(exc))
    else:
        raise TestFailure("ambiguous norm_stats key was silently resolved")


# ---------------------------------------------------------------------------
# 3. scheduler: warmup config wired, resume/reset really touch the LR
# ---------------------------------------------------------------------------
def test_default_scheduler_warms_up_then_constant():
    from energywam.training.trainer import EnergyWAMTrainer

    cfg = tiny_cfg(**{"trainer.warmup_steps": 4, "trainer.learning_rate": 1.0})
    m, cfg = tiny_model(cfg, seed=3)
    with tempfile.TemporaryDirectory() as d:
        cfg.trainer.output_dir = d
        tr = EnergyWAMTrainer(cfg, m)
        assert_true(tr.scheduler is not None, "no default scheduler built")
        lrs = []
        for i in range(5):
            lrs.append(float(tr.optimizer.param_groups[0]["lr"]))
            tr.train_step(tiny_batch(cfg, batch_size=2, seed=i))
        assert_close(lrs, [0.25, 0.5, 0.75, 1.0, 1.0], 1e-9,
                     "warmup_steps=4 must ramp LR linearly then hold it constant")

    # warmup_steps=0 keeps the previous behaviour: constant LR at learning_rate
    cfg0 = tiny_cfg(**{"trainer.warmup_steps": 0, "trainer.learning_rate": 1.0})
    m0, cfg0 = tiny_model(cfg0, seed=4)
    with tempfile.TemporaryDirectory() as d:
        cfg0.trainer.output_dir = d
        tr0 = EnergyWAMTrainer(cfg0, m0)
        lrs0 = []
        for i in range(3):
            lrs0.append(float(tr0.optimizer.param_groups[0]["lr"]))
            tr0.train_step(tiny_batch(cfg0, batch_size=2, seed=i))
        assert_close(lrs0, [1.0, 1.0, 1.0], 1e-9,
                     "warmup_steps=0 must leave the LR constant (tiny runs unchanged)")

    # a custom scheduler passed to the trainer always wins over the default
    cfgc = tiny_cfg(**{"trainer.warmup_steps": 4, "trainer.learning_rate": 1.0})
    mc, cfgc = tiny_model(cfgc, seed=5)
    with tempfile.TemporaryDirectory() as d:
        cfgc.trainer.output_dir = d
        opt = torch.optim.AdamW(mc.parameters(), lr=0.5)
        custom = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        trc = EnergyWAMTrainer(cfgc, mc, optimizer=opt, scheduler=custom)
        assert_true(trc.scheduler is custom, "custom scheduler was replaced by the default")
        trc.train_step(tiny_batch(cfgc, batch_size=2, seed=0))
        assert_close(float(opt.param_groups[0]["lr"]), 0.25, 1e-9, "custom scheduler not stepped")


def test_scheduler_resume_restores_next_step_lr():
    """Same-stage resume: the NEXT step's LR must equal the uninterrupted run's."""
    from energywam.training.trainer import EnergyWAMTrainer

    def lr_now(tr):
        return float(tr.optimizer.param_groups[0]["lr"])

    cfg = tiny_cfg(**{"trainer.warmup_steps": 8, "trainer.learning_rate": 1.0})
    batches = [tiny_batch(cfg, batch_size=2, seed=100 + i) for i in range(4)]

    # uninterrupted reference: 3 steps, note the LR the 4th step would use
    m_a, cfg_a = tiny_model(cfg, seed=6)
    with tempfile.TemporaryDirectory() as d:
        cfg_a.trainer.output_dir = d
        cfg_a.trainer.logging_frequency = 100
        tr_a = EnergyWAMTrainer(cfg_a, m_a)
        for i in range(3):
            tr_a.train_step(batches[i])
        lr_next_ref = lr_now(tr_a)               # 4/8 = 0.5
        assert_close(lr_next_ref, 0.5, 1e-9, "unexpected reference LR")
        tr_a.train_step(batches[3])
        lr_after_ref = lr_now(tr_a)

    # interrupted: 3 steps, save, reload into a fresh trainer
    with tempfile.TemporaryDirectory() as d:
        cfg_b = tiny_cfg(**{"trainer.warmup_steps": 8, "trainer.learning_rate": 1.0})
        cfg_b.trainer.output_dir = d
        cfg_b.trainer.logging_frequency = 100
        m_b, cfg_b = tiny_model(cfg_b, seed=7)
        tr_b = EnergyWAMTrainer(cfg_b, m_b)
        for i in range(3):
            tr_b.train_step(batches[i])
        path = tr_b.save("s.pt")

        cfg_c = tiny_cfg(**{"trainer.warmup_steps": 8, "trainer.learning_rate": 1.0})
        cfg_c.trainer.output_dir = d
        cfg_c.trainer.logging_frequency = 100
        m_c, cfg_c = tiny_model(cfg_c, seed=8)
        tr_c = EnergyWAMTrainer(cfg_c, m_c)
        meta = tr_c.load(path, mode="resume")
        assert_true(tr_c.global_step == 3, "step counter not restored")
        assert_true(tr_c.scheduler.last_epoch == 3,
                    f"scheduler position not restored: last_epoch={tr_c.scheduler.last_epoch}")
        assert_close(lr_now(tr_c), lr_next_ref, 1e-9,
                     "resume must restore the LR in force for the next step "
                     "(not the fresh-start LR)")
        tr_c.train_step(batches[3])
        assert_close(lr_now(tr_c), lr_after_ref, 1e-9,
                     "post-resume LR schedule diverged from the uninterrupted run")
        assert_true(meta.scheduler_policy == "continue", f"policy {meta.scheduler_policy}")


def test_scheduler_switch_reset_restarts_schedule():
    """Stage switch with scheduler_policy='reset' restarts the LR schedule."""
    from energywam.training.trainer import EnergyWAMTrainer

    cfg = tiny_cfg(**{"trainer.warmup_steps": 8, "trainer.learning_rate": 1.0,
                      "stage.stage": "bi"})
    with tempfile.TemporaryDirectory() as d:
        cfg.trainer.output_dir = d
        cfg.trainer.logging_frequency = 100
        m, cfg = tiny_model(cfg, seed=9)
        tr = EnergyWAMTrainer(cfg, m)
        for i in range(3):
            tr.train_step(tiny_batch(cfg, batch_size=2, seed=200 + i))
        path = tr.save("stage1.pt")
        assert_close(float(tr.optimizer.param_groups[0]["lr"]), 0.5, 1e-9, "pre-switch LR")

        cfg2 = tiny_cfg(**{"trainer.warmup_steps": 8, "trainer.learning_rate": 1.0,
                           "stage.stage": "joint"})
        cfg2.trainer.output_dir = d
        cfg2.trainer.logging_frequency = 100
        m2, cfg2 = tiny_model(cfg2, seed=10)
        tr2 = EnergyWAMTrainer(cfg2, m2)
        tr2.switch_stage("joint", path, optimizer_policy="reset", scheduler_policy="reset")
        assert_true(tr2.global_step == 0, "switch must reset the step counter")
        assert_true(tr2.scheduler.last_epoch == 0,
                    f"switch/reset must restart the schedule, last_epoch={tr2.scheduler.last_epoch}")
        assert_close(float(tr2.optimizer.param_groups[0]["lr"]), 0.125, 1e-9,
                     "switch/reset must return the LR to the initial schedule value "
                     "(warmup step 1 of 8), not keep the switched-from LR")
        lrs = []
        for i in range(2):
            lrs.append(float(tr2.optimizer.param_groups[0]["lr"]))
            tr2.train_step(tiny_batch(cfg2, batch_size=2, seed=300 + i))
        assert_close(lrs, [0.125, 0.25], 1e-9,
                     "after switch/reset the warmup schedule must run from the start")


# ---------------------------------------------------------------------------
# 4. evaluate.py CLI smoke (synthetic subset) + sincos state contract
# ---------------------------------------------------------------------------
def test_evaluate_cli_reports_offline_subset():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "r.json"
        env = dict(os.environ)
        env.update({
            "PYTHONPATH": str(ROOT),
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        })
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "evaluate.py"),
             "--config", str(ROOT / "configs" / "tiny_cpu.yaml"),
             "--synthetic", "3", "--out", str(out)],
            capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=300,
        )
        assert_true(proc.returncode == 0, f"evaluate.py failed: {proc.stderr[-2000:]}")
        report = json.loads(out.read_text())
        assert_true(report["metric"] == "offline_action_mse", f"metric {report.get('metric')}")
        assert_true(report["is_closed_loop_success_rate"] is False,
                    "offline report must never claim to be a closed-loop success rate")
        assert_true(report["subset"]["num_samples_evaluated"] == 3, f"subset {report.get('subset')}")
        assert_true(report["subset"]["is_full_dataset"] is False, "subset must be labelled as a subset")
        assert_true(report["num_samples"] == 3, f"num_samples {report.get('num_samples')}")
        assert_true("closed-loop" in report["note"] or report["is_closed_loop_success_rate"] is False,
                    "report must state it is not a success rate")


def test_sincos_state_layout_is_sin_then_cos():
    from energywam.data.energy_dataset import sincos_state

    q = np.asarray([0.1, -0.2, 0.3], dtype="float32")
    out = sincos_state(q, 6)
    assert_true(out.shape == (6,), f"shape {out.shape}")
    assert_close(out[:3], np.sin(q), 1e-7, "first block must be sin(q)")
    assert_close(out[3:], np.cos(q), 1e-7, "second block must be cos(q)")
    assert_true(not np.allclose(out[:3], np.cos(q)), "sin/cos blocks swapped")


TESTS = [
    ("eval_examples_follow_predict_contract", test_eval_examples_follow_predict_contract),
    ("eval_examples_resolve_data_mix_root", test_eval_examples_resolve_data_mix_root),
    ("serve_assembly_uses_checkpoint_norm_stats", test_serve_assembly_uses_checkpoint_norm_stats),
    ("serve_units_and_missing_stats_policy", test_serve_units_and_missing_stats_policy),
    ("default_scheduler_warms_up_then_constant", test_default_scheduler_warms_up_then_constant),
    ("scheduler_resume_restores_next_step_lr", test_scheduler_resume_restores_next_step_lr),
    ("scheduler_switch_reset_restarts_schedule", test_scheduler_switch_reset_restarts_schedule),
    ("evaluate_cli_reports_offline_subset", test_evaluate_cli_reports_offline_subset),
    ("sincos_state_layout_is_sin_then_cos", test_sincos_state_layout_is_sin_then_cos),
]
