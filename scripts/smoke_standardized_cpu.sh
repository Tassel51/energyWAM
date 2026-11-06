#!/usr/bin/env bash
# CPU smoke of the paper §4.8 standardized baseline (tiny backend, no GPU).
#
# Runs the whole chain on the synthetic tiny corpus:
#   1. fit the (explicitly synthetic) statistics bundle with the offline CLI
#   2. train bi for a few steps
#   3. same-stage resume from the checkpoint
#   4. bi -> joint stage switch
#   5. offline evaluation (explicit subset)
#   6. serve assembly with the EXTERNAL stats file removed (checkpoint-embedded
#      bundle must be enough) + exact raw-action inverse
#
# This is engineering smoke evidence for the standardized pipeline.  It does NOT
# load Qwen/DINO weights and is NOT evidence about the real 4B path or RoboCasa365.
#
# Usage:  bash scripts/smoke_standardized_cpu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-/data3/xyx_zhuzicheng/miniconda3/envs/LDA/bin/python}"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

RUN="${RUN_DIR:-logs/runs/standardized_tiny_$(date +%Y%m%d_%H%M%S)}"
CFG=configs/tiny_paper_standardized.yaml
BUNDLE=$RUN/tiny_synthetic_bundle.json
STEPS="${STEPS:-20}"
mkdir -p "$RUN"

echo "== [1/6] fit synthetic tiny statistics bundle (offline CLI) =="
$PY scripts/fit_normalization_stats.py --config "$CFG" --split train --synthetic-tiny \
    --out "$BUNDLE" --overwrite | tee "$RUN/00_fit_stats.log"

echo "== [2/6] train bi =="
$PY scripts/train.py --config "$CFG" --steps "$STEPS" --smoke \
    --set data.norm_stats_bundle_path="$BUNDLE" trainer.output_dir="$RUN" trainer.save_interval=1000 \
    | tee "$RUN/01_train_bi.log"
CKPT=$(ls -t "$RUN"/tiny_paper_standardized_step*.pt | head -1)
echo "   checkpoint: $CKPT"

echo "== [3/6] same-stage resume =="
$PY scripts/train.py --config "$CFG" --steps 5 --smoke --resume "$CKPT" \
    --set data.norm_stats_bundle_path="$BUNDLE" trainer.output_dir="$RUN" trainer.save_interval=1000 \
    | tee "$RUN/02_resume.log"

echo "== [4/6] bi -> joint stage switch =="
$PY scripts/train.py --config "$CFG" --steps 5 --smoke --init-from "$CKPT" \
    --set data.norm_stats_bundle_path="$BUNDLE" trainer.output_dir="$RUN/joint" \
          stage.stage=joint trainer.save_interval=1000 \
    | tee "$RUN/03_switch_joint.log"

echo "== [5/6] offline evaluation (subset) =="
$PY scripts/evaluate.py --config "$CFG" --checkpoint "$CKPT" --synthetic 8 \
    --set data.norm_stats_bundle_path="$BUNDLE" \
    --out "$RUN/04_eval_report.json" | tee "$RUN/04_eval.log"

echo "== [6/6] serve assembly with the external stats file hidden =="
mv "$BUNDLE" "$BUNDLE.hidden"
trap 'if [ -f "$BUNDLE.hidden" ]; then mv "$BUNDLE.hidden" "$BUNDLE"; fi' EXIT
$PY - <<PYEOF | tee "$RUN/05_serve.log"
import importlib.util, sys, numpy as np
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("ew_serve", "scripts/serve.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from energywam.config import load_config
from energywam.inference.predictor import build_policy_server_adapter
# the config points at the SAME bundle file training used — it is currently
# hidden on disk, so the checkpoint-embedded bundle must be enough
cfg = load_config("$CFG", overrides=["data.norm_stats_bundle_path=$BUNDLE"])
p = mod.build_serve_predictor(cfg, checkpoint_path="$CKPT")
pol = build_policy_server_adapter(p)
meta = pol.metadata
assert meta["action_units"] == "raw_dataset_units", meta
out = pol.predict_action(examples=[{"cond_vec": np.zeros(7, dtype="float32"), "lang": "t"}])
a = np.asarray(out["actions"]); assert a.shape[-1] == 3, a.shape
print("serve ok:", out["action_units"], "shape", a.shape, "action_space", p.action_space)
print("note: served with the external stats file absent (checkpoint-embedded bundle)")
PYEOF
mv "$BUNDLE.hidden" "$BUNDLE"

echo
echo "smoke finished: all six steps passed.  Artifacts under $RUN/"
