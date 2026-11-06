#!/usr/bin/env bash
# Run the full CPU test suite (explicit tiny mode + structure tests).
#   bash scripts/run_tiny_tests.sh              # everything
#   bash scripts/run_tiny_tests.sh test_gradients
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

cd "${ENERGYWAM_ROOT}"
FILTER="${1:-}"
"${PYTHON_BIN}" tests/run_all.py ${FILTER}
