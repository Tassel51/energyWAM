"""Service loopback + checkpoint/diagnostics integration tests.

The serve test starts the REAL StarVLA ``WebsocketPolicyServer`` bound to
127.0.0.1 and drives it with the real ``WebsocketClientPolicy`` — a genuine
request/response round trip, not a stub.  Nothing is exposed externally.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .harness import TestFailure, assert_true
from .real_fixtures import build_real_model, make_real_batch


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ---------------------------------------------------------------------------
# P0-6: real websocket serve + loopback round trip
# ---------------------------------------------------------------------------
def test_serve_loopback_request_response():
    root = Path(__file__).resolve().parents[1].parent / "starVLA"
    if not root.exists():
        raise TestFailure(f"starVLA snapshot not found at {root}")
    import sys

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from deployment.model_server.tools.websocket_policy_client import (
            WebsocketClientPolicy,
        )
    except Exception as exc:
        raise TestFailure(f"WebsocketClientPolicy not importable: {exc}")

    from energywam.inference.predictor import EnergyWAMPredictor, serve

    ew, cfg, geo = build_real_model(with_real_dino=False, seed=21)
    predictor = EnergyWAMPredictor(ew)

    port = _free_port()
    server = serve(
        predictor,
        host="127.0.0.1",
        port=port,
        action_keys=["action.end_effector_position"],
        state_keys=["state.end_effector_position_relative"],
        unnorm_key=None,
        obs_image_size=(224, 224),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # wait for the socket to accept
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.1)
    else:
        raise TestFailure("server never started listening")

    try:
        client = WebsocketClientPolicy(host="127.0.0.1", port=port)
        meta = client.get_server_metadata()
        assert_true(isinstance(meta, dict) and meta, "no metadata at handshake")
        for key in ("action_chunk_size", "action_keys", "state_keys", "available_unnorm_keys"):
            assert_true(key in meta, f"metadata missing {key!r}: {sorted(meta)}")
        assert_true(int(meta["action_chunk_size"]) == cfg.data.action_horizon,
                    f"action_chunk_size {meta['action_chunk_size']} != {cfg.data.action_horizon}")

        # a real infer round trip
        batch = make_real_batch(ew, cfg, geo, B=1, seed=2)
        payload = {
            "examples": [
                {
                    "image": batch["images"][0],
                    "lang": batch["langs"][0],
                    "state": batch["states"][0],
                }
            ]
        }
        reply = client.predict_action(payload)
        assert_true(isinstance(reply, dict), f"reply is not a dict: {type(reply)}")
        # the websocket protocol wraps the policy output in `data`
        assert_true(bool(reply.get("ok", False)) or reply.get("status") == "ok",
                    f"server reported failure: {sorted(reply)}")
        inner = reply.get("data", reply)
        assert_true(isinstance(inner, dict), f"reply data is not a dict: {type(inner)}")
        assert_true("actions" in inner, f"reply missing 'actions': {sorted(inner)}")
        acts = np.asarray(inner["actions"])
        assert_true(acts.shape == (1, cfg.data.action_horizon, cfg.data.action_dim),
                    f"actions shape {acts.shape}")
        assert_true(np.isfinite(acts).all(), "non-finite actions served")
    finally:
        try:
            thread.join(timeout=1)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# P1-8: trainer diagnostics actually land in the jsonl
# ---------------------------------------------------------------------------
def test_trainer_jsonl_contains_required_diagnostics():
    import tempfile

    from energywam.config import load_config
    from energywam.data.tiny_data import make_tiny_batch
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.training.trainer import EnergyWAMTrainer

    ROOT = Path(__file__).resolve().parents[1]
    cfg = load_config(ROOT / "configs" / "tiny_overfit.yaml")
    cfg.trainer.logging_frequency = 1
    torch.manual_seed(0)
    m = QwenEnergyWAM(cfg)
    with tempfile.TemporaryDirectory() as d:
        cfg.trainer.output_dir = d
        tr = EnergyWAMTrainer(cfg, m)
        for i in range(3):
            tr.train_step(make_tiny_batch(cfg, batch_size=2, seed=i))
        log = Path(d) / "train_log.jsonl"
        assert_true(log.exists(), "no train_log.jsonl written")
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        assert_true(len(rows) >= 1, "empty train log")
        row = rows[-1]
        required = [
            "action_loss",
            "step_seconds",
            "forward_seconds",
            "backward_seconds",
            "optimizer_seconds",
            "param_grad_norm",
            "param_grad_nonfinite",
            "params_with_grad",
            "global_step",
        ]
        for key in required:
            assert_true(key in row, f"train log missing {key!r}: {sorted(row)}")
        assert_true(any(k.startswith("diag/") for k in row),
                    f"no solver diagnostics in the log: {sorted(row)[:20]}")
        traj = [k for k in row if k.startswith("diag/") and "energy_traj_" in k]
        assert_true(len(traj) > 0, f"no energy trajectory in the log: {sorted(row)}")
        upd = [k for k in row if "update_norm" in k]
        assert_true(len(upd) > 0, "no candidate update magnitudes in the log")
        assert_true("diag/steps" in row or any("steps" in k for k in row),
                    "inner step count K not logged")
        assert_true(float(row["param_grad_nonfinite"]) == 0.0, "non-finite grads reported")
        # valid mask element counts must be present (from the stage logs)
        assert_true(any("valid_elems" in k for k in row),
                    f"no valid-element counts: {sorted(row)}")


# ---------------------------------------------------------------------------
# P1-9: checkpoint carries norm stats + rejects a stage-mismatched resume
# ---------------------------------------------------------------------------
def test_checkpoint_norm_stats_and_stage_guard():
    import tempfile

    from energywam.config import load_config
    from energywam.data.normalization import NormStats
    from energywam.data.tiny_data import make_tiny_batch
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.training.checkpoint import load_checkpoint
    from energywam.training.trainer import EnergyWAMTrainer

    ROOT = Path(__file__).resolve().parents[1]
    cfg = load_config(ROOT / "configs" / "tiny_overfit.yaml")
    torch.manual_seed(0)
    m = QwenEnergyWAM(cfg)
    m.norm_stats = {
        "task_a": NormStats(mode="min_max",
                            min=np.zeros(3, dtype="float32"),
                            max=np.ones(3, dtype="float32"),
                            version="test_v1")
    }
    with tempfile.TemporaryDirectory() as d:
        cfg.trainer.output_dir = d
        tr = EnergyWAMTrainer(cfg, m)
        tr.train_step(make_tiny_batch(cfg, batch_size=2, seed=1))
        path = tr.save("c.pt")

        payload = load_checkpoint(path)
        ns = payload["meta"].get("norm_stats")
        assert_true(ns and "task_a" in ns, f"norm stats not saved: {payload['meta'].keys()}")
        assert_true(ns["task_a"]["min"] == [0.0, 0.0, 0.0], f"norm stats mangled: {ns['task_a']}")
        assert_true(payload["meta"]["stage"] == cfg.stage.stage, "stage not saved")

        # resume with a DIFFERENT stage must be refused
        cfg2 = load_config(ROOT / "configs" / "tiny_joint.yaml")
        cfg2.trainer.output_dir = d
        m2 = QwenEnergyWAM(cfg2)
        tr2 = EnergyWAMTrainer(cfg2, m2)
        try:
            tr2.load(path, mode="resume")
        except ValueError as exc:
            assert_true("stage" in str(exc).lower(), str(exc))
        else:
            raise TestFailure("stage-mismatched resume was not rejected")

        # a matching resume must work and restore the step counter
        cfg3 = load_config(ROOT / "configs" / "tiny_overfit.yaml")
        cfg3.trainer.output_dir = d
        m3 = QwenEnergyWAM(cfg3)
        tr3 = EnergyWAMTrainer(cfg3, m3)
        meta = tr3.load(path, mode="resume")
        assert_true(tr3.global_step == meta.global_step, "step counter not restored")


def test_resume_next_step_matches_uninterrupted_training():
    """Saving at step k and resuming must reproduce the step-k+1 loss that an
    uninterrupted run produces (same RNG state, same optimiser)."""
    import tempfile

    from energywam.config import load_config
    from energywam.data.tiny_data import make_tiny_batch
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.training.trainer import EnergyWAMTrainer

    ROOT = Path(__file__).resolve().parents[1]

    def fresh(cfg, seed):
        torch.manual_seed(seed)
        return QwenEnergyWAM(cfg)

    batches = None

    # uninterrupted: 4 steps
    cfg_a = load_config(ROOT / "configs" / "tiny_overfit.yaml")
    cfg_a.trainer.logging_frequency = 100
    torch.manual_seed(123)
    m_a = fresh(cfg_a, 123)
    tr_a = EnergyWAMTrainer(cfg_a, m_a)
    batches = [make_tiny_batch(cfg_a, batch_size=2, seed=100 + i) for i in range(4)]
    for i in range(3):
        tr_a.train_step(batches[i])
    snap = {k: v.clone() for k, v in m_a.state_dict().items()}
    loss_uninterrupted = tr_a.train_step(batches[3])["action_loss"]

    # interrupted: 3 steps, save, reload into a fresh model, 4th step
    with tempfile.TemporaryDirectory() as d:
        cfg_b = load_config(ROOT / "configs" / "tiny_overfit.yaml")
        cfg_b.trainer.output_dir = d
        cfg_b.trainer.logging_frequency = 100
        torch.manual_seed(123)
        m_b = fresh(cfg_b, 123)
        tr_b = EnergyWAMTrainer(cfg_b, m_b)
        for i in range(3):
            tr_b.train_step(batches[i])
        path = tr_b.save("s.pt")

        torch.manual_seed(999)  # different init — must come from the checkpoint
        m_c = fresh(cfg_b, 999)
        tr_c = EnergyWAMTrainer(cfg_b, m_c)
        tr_c.load(path, mode="resume")
        loss_resumed = tr_c.train_step(batches[3])["action_loss"]

    assert_true(abs(loss_uninterrupted - loss_resumed) < 1e-4,
                f"resume diverged from uninterrupted training: "
                f"{loss_uninterrupted} vs {loss_resumed}")


TESTS = [
    ("serve_loopback_request_response", test_serve_loopback_request_response),
    ("trainer_jsonl_contains_required_diagnostics", test_trainer_jsonl_contains_required_diagnostics),
    ("checkpoint_norm_stats_and_stage_guard", test_checkpoint_norm_stats_and_stage_guard),
    ("resume_next_step_matches_uninterrupted_training", test_resume_next_step_matches_uninterrupted_training),
]
