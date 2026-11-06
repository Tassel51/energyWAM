#!/usr/bin/env bash
# REAL smoke: 2 training steps on the pretrained Qwen3-VL-4B + frozen DINOv3,
# to verify the double backward and the token injection on actual weights.
# Still GPU-gated — needs a card with enough free memory.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "${ENERGYWAM_ROOT}"
: "${ENERGYWAM_ALLOW_GPU_TRAINING:?Refusing to start a real training run without GPU approval.}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}"
"${PYTHON_BIN}" scripts/train.py --config configs/real_smoke.yaml "$@"
