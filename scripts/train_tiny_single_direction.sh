#!/usr/bin/env bash
# Tiny-mode single-direction ablation (CPU overfit demo).
#   bash scripts/train_tiny_single_direction.sh action_given_future
#   bash scripts/train_tiny_single_direction.sh future_given_action
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
cd "${ENERGYWAM_ROOT}"

DIR="${1:-action_given_future}"
case "${DIR}" in
  action_given_future|future_given_action) ;;
  *) echo "usage: $0 {action_given_future|future_given_action}"; exit 1 ;;
esac
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
cfg.stage.stage = "${DIR}"
torch.manual_seed(cfg.trainer.seed)
m = QwenEnergyWAM(cfg)
tr = EnergyWAMTrainer(cfg, m)
tr.export_config()
data = [make_tiny_batch(cfg, batch_size=cfg.trainer.per_device_batch_size, seed=4000)]
info = tr.fit(data, max_steps=int("${STEPS}"))
print("stage:", cfg.stage.stage)
print("train:", info["steps"], "steps in", info["seconds"], "s")
print("first loss", info["first_loss"]["action_loss"], "-> last", info["last_loss"]["action_loss"])
print("checkpoint:", tr.save("tiny_${DIR}.pt"))
PY
