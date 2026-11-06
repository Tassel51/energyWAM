#!/usr/bin/env bash
# Ablation: single-direction conditional training (Future->Action or Action->Future).
#   bash scripts/train_single_direction.sh action_given_future
#   bash scripts/train_single_direction.sh future_given_action
# SCRIPT ONLY: same GPU-approval gate.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "${ENERGYWAM_ROOT}"
DIR="${1:-action_given_future}"
CONFIG="configs/${DIR}.yaml"
[[ -f "${CONFIG}" ]] || { echo "missing ${CONFIG}"; exit 1; }
shift || true
: "${ENERGYWAM_ALLOW_GPU_TRAINING:?Refusing to start a real training run without GPU approval.}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}"
"${PYTHON_BIN}" scripts/train.py --config "${CONFIG}" "$@"
