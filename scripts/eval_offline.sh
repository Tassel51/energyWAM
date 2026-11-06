#!/usr/bin/env bash
# Offline evaluation: action MSE against the demonstration.
# This is NOT a closed-loop success rate.  For a closed-loop SR use the existing
# StarVLA RoboCasa365 eval entry (examples/simBenchmarks/Robocasa_365/eval_files).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "${ENERGYWAM_ROOT}"

CONFIG="${1:-configs/eval.yaml}"
CKPT="${2:-}"
shift 2 2>/dev/null || shift $#
"${PYTHON_BIN}" scripts/evaluate.py --config "${CONFIG}" ${CKPT:+--checkpoint "${CKPT}"} "$@"
