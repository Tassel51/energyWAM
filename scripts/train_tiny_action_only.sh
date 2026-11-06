#!/usr/bin/env bash
# Tiny-mode action-only energy OVERFIT (CPU).  Ablation entry.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
cd "${ENERGYWAM_ROOT}"

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
cfg.stage.stage = "action_only"
torch.manual_seed(cfg.trainer.seed)
m = QwenEnergyWAM(cfg)
tr = EnergyWAMTrainer(cfg, m)
tr.export_config()
data = [make_tiny_batch(cfg, batch_size=cfg.trainer.per_device_batch_size, seed=3000)]
info = tr.fit(data, max_steps=int("${STEPS}"))
print("train:", info["steps"], "steps in", info["seconds"], "s")
print("first loss", info["first_loss"]["action_loss"], "-> last", info["last_loss"]["action_loss"])
print("checkpoint:", tr.save("tiny_action_only.pt"))
PY
