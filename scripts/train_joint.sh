#!/usr/bin/env bash
# Stage 2 (joint) — loads the stage-1 checkpoint and switches stage.
# SCRIPT ONLY: same GPU-approval gate as scripts/train_bi.sh.
#
#   ENERGYWAM_ALLOW_GPU_TRAINING=1 CUDA_VISIBLE_DEVICES=0 \
#     bash scripts/train_joint.sh configs/joint.yaml logs/runs/energywam_bi/ckpt.pt
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "${ENERGYWAM_ROOT}"

CONFIG="${1:-configs/joint.yaml}"
CKPT="${2:-}"
shift 2 2>/dev/null || shift $#

: "${ENERGYWAM_ALLOW_GPU_TRAINING:?Refusing to start a real training run without GPU approval.}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}"

ARGS=(--config "${CONFIG}")
if [[ -n "${CKPT}" ]]; then
  ARGS+=(--init-from "${CKPT}" --stage-switch-policy "${STAGE_SWITCH_POLICY:-reset}")
fi
"${PYTHON_BIN}" scripts/train.py "${ARGS[@]}" "$@"
