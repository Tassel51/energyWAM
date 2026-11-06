#!/usr/bin/env bash
# Stage 1 (bi) FULL training — SCRIPT ONLY, DO NOT RUN WITHOUT RESOURCE APPROVAL.
#
# The current server has no free GPU (all 9 cards carry active jobs as of
# 2026-10-28).  Launching this needs the main agent / user to confirm available
# memory first.  See docs/ARCHITECTURE.md "resource limits".
#
#   bash scripts/train_bi.sh [CONFIG] [extra KEY=VALUE overrides...]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "${ENERGYWAM_ROOT}"

CONFIG="${1:-configs/bi.yaml}"
shift || true

: "${ENERGYWAM_ALLOW_GPU_TRAINING:?\
Refusing to start a real training run.
Set ENERGYWAM_ALLOW_GPU_TRAINING=1 after confirming free GPU memory,
and set CUDA_VISIBLE_DEVICES to a card that has headroom.
(All 9 GPUs were occupied at the time this script was written.)}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to a free GPU}"
"${PYTHON_BIN}" scripts/train.py --config "${CONFIG}" "$@"
