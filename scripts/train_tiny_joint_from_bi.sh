#!/usr/bin/env bash
# bi -> joint stage switch on tiny mode, with an explicit optimiser policy.
#
#   bash scripts/train_tiny_joint_from_bi.sh logs/runs/tiny_overfit/tiny_bi.pt
#   POLICY=continue bash scripts/train_tiny_joint_from_bi.sh <stage1.ckpt>
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
cd "${ENERGYWAM_ROOT}"

CKPT="${1:-logs/runs/tiny_overfit/tiny_bi.pt}"
POLICY="${POLICY:-reset}"
SCHED_POLICY="${SCHED_POLICY:-reset}"
STEPS="${STEPS:-300}"

"${PYTHON_BIN}" - <<PY
import sys
sys.path.insert(0, "${ENERGYWAM_ROOT}")
import torch
from energywam.config import load_config
from energywam.data.tiny_data import make_tiny_batch
from energywam.models.energy_wam import QwenEnergyWAM
from energywam.training.trainer import EnergyWAMTrainer

cfg = load_config("${ENERGYWAM_ROOT}/configs/tiny_overfit.yaml")
cfg.stage.stage = "joint"
torch.manual_seed(cfg.trainer.seed)
m = QwenEnergyWAM(cfg)
tr = EnergyWAMTrainer(cfg, m)
meta = tr.switch_stage("joint", "${CKPT}", optimizer_policy="${POLICY}", scheduler_policy="${SCHED_POLICY}")
print("switched:", meta.stage, "->", cfg.stage.stage,
      "| optimizer:", meta.optimizer_policy, "| scheduler:", meta.scheduler_policy,
      "| step:", tr.global_step)
data = [make_tiny_batch(cfg, batch_size=cfg.trainer.per_device_batch_size, seed=2000)]
info = tr.fit(data, max_steps=int("${STEPS}"))
print("train:", info["steps"], "steps in", info["seconds"], "s")
print("first loss", info["first_loss"]["action_loss"], "-> last", info["last_loss"]["action_loss"])
print("checkpoint:", tr.save("tiny_joint.pt"))
PY
