#!/usr/bin/env bash
# Shared environment for EnergyWAM scripts.
# CPU-only work must set CUDA_VISIBLE_DEVICES='' and cap the thread count.
set -euo pipefail

export ENERGYWAM_ROOT="${ENERGYWAM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="${ENERGYWAM_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON_BIN="${PYTHON_BIN:-/usr/local/miniconda3/envs/py312/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

mkdir -p "${ENERGYWAM_ROOT}/logs"
