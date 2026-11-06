#!/usr/bin/env bash
# The ordinary StarVLA VLA baseline uses the EXISTING StarVLA entry point.
# EnergyWAM does not reimplement or replace it.
#
#   bash scripts/run_baseline_starvla.sh <train|eval>
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
STARVLA="${ENERGYWAM_ROOT}/../starVLA"
MODE="${1:-train}"

if [[ "${MODE}" == "train" ]]; then
  echo "Baseline training entry (read-only reference, not modified by EnergyWAM):"
  echo "  cd ${STARVLA} && bash examples/simBenchmarks/Robocasa_365/train_files/run_robocasa365.sh"
  echo "  config: examples/simBenchmarks/Robocasa_365/train_files/starvla_qwenoft_robocasa365.yaml"
else
  echo "Baseline evaluation entry:"
  echo "  cd ${STARVLA} && bash examples/simBenchmarks/Robocasa_365/eval_files/run_eval.sh"
fi
echo
echo "LDA-1B is reference-only. EnergyWAM does not claim to reproduce an"
echo "LDA baseline under matched conditions."
