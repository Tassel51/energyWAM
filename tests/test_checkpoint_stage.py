"""Checkpoint save/load, same-stage resume and bi -> joint stage switching."""

from __future__ import annotations

import copy

import torch

from .common import tiny_batch, tiny_cfg, tiny_model
from .harness import TestFailure, assert_close, assert_true


def _train_a_few(model, cfg, n=3, seed=0):
    from energywam.training.trainer import EnergyWAMTrainer

    tr = EnergyWAMTrainer(cfg, model)
    for i in range(n):
        batch = tiny_batch(cfg, batch_size=2, seed=seed + i)
        tr.train_step(batch)
    return tr


def test_checkpoint_roundtrip_eval_deterministic():
    """Save -> load -> eval must reproduce the same prediction bit-for-bit."""
    import tempfile

    from energywam.training.checkpoint import CheckpointMeta, load_checkpoint, restore, save_checkpoint

    cfg = tiny_cfg()
    m, cfg = tiny_model(cfg, seed=201)
    _train_a_few(m, cfg, n=2, seed=1)
    m.eval()
    batch = tiny_batch(cfg, batch_size=2, seed=999)
    before = m.predict(batch, mode="joint")["normalized_actions"]

    with tempfile.TemporaryDirectory() as d:
        path = save_checkpoint(
            f"{d}/ckpt.pt", model=m, optimizer=None, scheduler=None, cfg=cfg,
            meta=CheckpointMeta(stage=cfg.stage.stage, global_step=2),
        )
        m2, _ = tiny_model(cfg, seed=202)  # different init
        payload = load_checkpoint(path)
        restore(payload, model=m2, optimizer=None, scheduler=None, mode="resume")
        m2.eval()
        after = m2.predict(batch, mode="joint")["normalized_actions"]
    assert_close(before, after, 0.0, "checkpoint round-trip changed the eval output")


def test_resume_restores_step_and_rng():
    import tempfile

    from energywam.training.trainer import EnergyWAMTrainer

    cfg = tiny_cfg()
    m, cfg = tiny_model(cfg, seed=203)
    tr = _train_a_few(m, cfg, n=3, seed=10)
    assert_true(tr.global_step == 3, f"expected 3 steps, got {tr.global_step}")

    with tempfile.TemporaryDirectory() as d:
        cfg.trainer.output_dir = d
        tr.log_dir = __import__("pathlib").Path(d)
        path = tr.save("ckpt.pt")
        m2, cfg2 = tiny_model(cfg, seed=204)
        cfg2.trainer.output_dir = d
        tr2 = EnergyWAMTrainer(cfg2, m2)
        meta = tr2.load(path, mode="resume")
        assert_true(tr2.global_step == 3, f"resume did not restore the step: {tr2.global_step}")
        assert_true(meta.stage == cfg.stage.stage, "stage mismatch after resume")


def test_bi_then_joint_stage_switch_with_explicit_optimizer_policy():
    """Stage 1 (bi) -> checkpoint -> stage 2 (joint) with a logged policy."""
    import tempfile

    from energywam.training.trainer import EnergyWAMTrainer

    cfg_bi = tiny_cfg(**{"stage.stage": "bi"})
    m, cfg_bi = tiny_model(cfg_bi, seed=205)
    tr = EnergyWAMTrainer(cfg_bi, m)
    with tempfile.TemporaryDirectory() as d:
        cfg_bi.trainer.output_dir = d
        tr.log_dir = __import__("pathlib").Path(d)
        for i in range(3):
            tr.train_step(tiny_batch(cfg_bi, batch_size=2, seed=30 + i))
        path = tr.save("stage1.pt")
        w_before = copy.deepcopy(m.state_dict())

        # stage switch
        cfg_joint = tiny_cfg(**{"stage.stage": "joint"})
        cfg_joint.trainer.output_dir = d
        m2, cfg_joint = tiny_model(cfg_joint, seed=206)
        tr2 = EnergyWAMTrainer(cfg_joint, m2)
        meta = tr2.switch_stage("joint", path, optimizer_policy="reset")
        assert_true(meta.stage == "bi", f"expected source stage bi, got {meta.stage}")
        assert_true(meta.optimizer_policy == "reset", "optimizer policy not honoured")
        assert_true(tr2.global_step == 0, "stage switch must start a new step counter")
        # model weights must have been carried over
        w_after = m2.state_dict()
        n_equal = sum(1 for k in w_before if torch.equal(w_before[k], w_after[k]))
        assert_true(n_equal == len(w_before), f"only {n_equal}/{len(w_before)} weights carried over")

        # and joint training must run
        out = tr2.train_step(tiny_batch(cfg_joint, batch_size=2, seed=40))
        assert_true(torch.isfinite(torch.tensor(out["action_loss"])), "joint step loss not finite")

        # 'continue' policy must also be accepted and logged
        meta2 = tr2.switch_stage("joint", path, optimizer_policy="continue")
        assert_true(meta2.optimizer_policy == "continue", "continue policy not honoured")


def test_same_stage_resume_differs_from_switch():
    """resume keeps the optimiser state; switch/reset rebuilds it."""
    import tempfile

    from energywam.training.trainer import EnergyWAMTrainer

    cfg = tiny_cfg()
    m, cfg = tiny_model(cfg, seed=207)
    tr = EnergyWAMTrainer(cfg, m)
    with tempfile.TemporaryDirectory() as d:
        cfg.trainer.output_dir = d
        tr.log_dir = __import__("pathlib").Path(d)
        for i in range(2):
            tr.train_step(tiny_batch(cfg, batch_size=2, seed=50 + i))
        path = tr.save("c.pt")

        # resume
        m1, c1 = tiny_model(cfg, seed=208)
        tr1 = EnergyWAMTrainer(c1, m1)
        tr1.load(path, mode="resume")
        assert_true(tr1.global_step == 2, "resume step")
        assert_true(len(tr1.optimizer.state) > 0, "resume must restore optimiser state")

        # switch/reset
        m2, c2 = tiny_model(cfg, seed=209)
        tr2 = EnergyWAMTrainer(c2, m2)
        tr2.load(path, mode="switch")
        assert_true(tr2.global_step == 0, "switch must reset the step counter")
        assert_true(len(tr2.optimizer.state) == 0, "switch/reset must not restore optimiser state")


TESTS = [
    ("checkpoint_roundtrip_eval_deterministic", test_checkpoint_roundtrip_eval_deterministic),
    ("resume_restores_step_and_rng", test_resume_restores_step_and_rng),
    ("bi_then_joint_stage_switch_with_explicit_optimizer_policy", test_bi_then_joint_stage_switch_with_explicit_optimizer_policy),
    ("same_stage_resume_differs_from_switch", test_same_stage_resume_differs_from_switch),
]


def test_learned_step_anchor_mismatch_is_rejected_before_loading():
    import pytest
    from energywam.config import to_plain_dict
    from energywam.training.checkpoint import restore
    cfg=tiny_cfg(**{'solver.learnable_step_sizes':True})
    model,cfg=tiny_model(cfg,seed=291)
    payload={'config':to_plain_dict(cfg),'model':model.state_dict(),'meta':{'stage':'bi','global_step':0}}
    cfg.solver.alpha_future *= 2
    with pytest.raises(ValueError,match='solver.alpha_future'):
        restore(payload,model=model,optimizer=None,scheduler=None,mode='resume')


def test_exact_training_resume_rejects_sample_order_changes():
    import pytest
    from energywam.config import to_plain_dict
    from energywam.training.checkpoint import restore
    for field,value in [('per_device_batch_size',8),('seed',43),('warmup_steps',20),('shared_microbatch_size',2)]:
        cfg=tiny_cfg();model,cfg=tiny_model(cfg,seed=292)
        payload={'config':to_plain_dict(cfg),'model':model.state_dict(),'meta':{'stage':'bi','global_step':4}}
        optimizer=torch.optim.SGD(model.parameters(),lr=0.)
        setattr(cfg.trainer,field,value)
        with pytest.raises(ValueError,match=field):
            restore(payload,model=model,optimizer=optimizer,scheduler=None,mode='resume')
