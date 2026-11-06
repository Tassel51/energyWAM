#!/usr/bin/env bash
# Ablation: direct joint training from the pretrained Qwen (no stage 1).
# SCRIPT ONLY: same GPU-approval gate.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "${ENERGYWAM_ROOT}"
CONFIG="${1:-configs/direct_joint.yaml}"; shift || true
: "${ENERGYWAM_ALLOW_GPU_TRAINING:?Refusing to start a real training run without GPU approval.}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}"
"${PYTHON_BIN}" scripts/train.py --config "${CONFIG}" "$@"
