#!/bin/bash
# EnergyWAM pipeline watchdog — supervised by supervisord (pid 98, OUTSIDE the
# agent harness's process tree).  The harness kills everything spawned from its
# Bash tool at session teardown (even setsid/nohup daemons), so long-running
# jobs live here instead.
#
# Stages (idempotent; the loop re-enters whatever is still missing):
#   1. fit   — 150k-window stats bundle (hard prerequisite of bi training)
#   2. smoke — memory smoke on GPU 3 with a THREE-RUNG fallback ladder:
#              rung 1 = K=2 exactly as configured
#              rung 2 = K=1 (documented engineering instantiation)
#              rung 3 = K=1 without saved-tensor CPU offload
#              The rung that passes is recorded as training overrides —
#              no silent substitution: the rung is written to
#              logs/.watchdog_train_overrides and reported.
#   3. train — Stage-I bi training with the smoke-validated overrides;
#              retry rung forces per_device_batch_size=1; resumes from the
#              rolling ckpt_latest.pt checkpoint when one exists.
#
# A stage that exhausts its attempt budget parks and writes logs/NEEDS_CLAUDE
# (never crash-loops).  Jobs are backgrounded with rc files so watchdog
# restarts neither kill running jobs nor lose their exit status.
set -u
cd /root/zzc/energyWAM

# single instance: relaunching the watchdog (after a session cleanup kills it)
# is always safe — a second instance exits immediately
exec 9>logs/.watchdog.lock
if ! flock -n 9; then
  echo "[watchdog $$] another watchdog holds the lock; exiting"
  exit 0
fi

PY=/usr/local/miniconda3/envs/py312/bin/python
export PYTHONUNBUFFERED=1 PYTHONPATH=/root/zzc/energyWAM
export ENERGYWAM_ALLOW_GPU_TRAINING=1 CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STATS=stats/trainmix_150k_paper_standardized_v1.json
FIT_LOG=logs/fit150k_gpu3.log
SMOKE_LOG=logs/local4090_smoke_ladder.log
TRAIN_LOG=logs/train_bi_gpu3.log
RUN_DIR=logs/runs/energywam_local4090_bi_20261031
OVR_FILE=logs/.watchdog_train_overrides
BI_CFG=configs/local4090_20261031_bi.yaml

log() { echo "[watchdog $(date '+%m-%d %H:%M:%S')] $*"; }
cnt() { cat "logs/.wd_$1" 2>/dev/null || echo 0; }
bump() { local n=$(( $(cnt "$1") + 1 )); echo "$n" > "logs/.wd_$1"; echo "$n"; }
alive() {
  local pid; pid=$(cat "logs/.wd_$1.pid" 2>/dev/null) || return 1
  [ -n "${pid:-}" ] || return 1
  local st; st=$(sed -n "s/.*) \([A-Z]\) .*/\1/p" "/proc/$pid/stat" 2>/dev/null)
  [ -n "$st" ] && [ "$st" != "Z" ]
}
spawn() {
  local name=$1 logfile=$2; shift 2
  rm -f "logs/.wd_${name}.rc"
  ( "$@" >> "$logfile" 2>&1; echo "$?" > "logs/.wd_${name}.rc" ) &
  echo $! > "logs/.wd_${name}.pid"
  log "started $name pid=$! -> $logfile"
}
park() {
  log "PARK: $*"
  echo "$*" > logs/NEEDS_CLAUDE
  while true; do sleep 300; done
}

# smoke rung N -> train.py --set args (smoke uses batch-1 smoke config)
smoke_sets() {
  case "$1" in
    1) echo "trainer.logging_frequency=1" ;;
    2) echo "trainer.logging_frequency=1 solver.train_inner_steps=1 solver.eval_inner_steps=1" ;;
    3) echo "trainer.logging_frequency=1 solver.train_inner_steps=1 solver.eval_inner_steps=1 trainer.offload_saved_tensors=false" ;;
  esac
}
# smoke rung N -> training overrides (bi config keeps K=2 unless rung said otherwise)
train_sets() {
  case "$1" in
    1) echo "" ;;
    2) echo "solver.train_inner_steps=1 solver.eval_inner_steps=1" ;;
    3) echo "solver.train_inner_steps=1 solver.eval_inner_steps=1 trainer.offload_saved_tensors=false" ;;
  esac
}

log "watchdog up (pid $$)"
while true; do
  # ---- stage 1: stats fit ------------------------------------------
  if [ ! -f "$STATS" ] && ! alive fit; then
    if [ -f logs/.wd_fit.rc ]; then
      log "fit exited rc=$(cat logs/.wd_fit.rc) with no stats file"
    fi
    n=$(bump fit_attempts)
    if [ "$n" -le 3 ]; then
      log "fit attempt $n (150k windows, GPU DINO)"
      spawn fit "$FIT_LOG" "$PY" scripts/fit_normalization_stats.py \
        --config "$BI_CFG" --split train \
        --episodes-manifest manifests/train_episodes.json \
        --val-manifest manifests/val_episodes.json \
        --test-manifest manifests/test_episodes.json \
        --out "$STATS" --max-samples 150000 --device cuda
    else
      park "fit failed $n attempts — see $FIT_LOG"
    fi
  fi

  # ---- stage 2: smoke ladder (gated on stats) -----------------------
  if [ -f "$STATS" ] && [ ! -f logs/SMOKE_PASSED ] && ! alive smoke; then
    if [ -f logs/.wd_smoke.rc ]; then
      rc=$(cat logs/.wd_smoke.rc)
      rm -f logs/.wd_smoke.rc
      rung=$(cnt smoke_rung)
      if [ "$rc" = "0" ] && grep -q "finished 2 steps" "$SMOKE_LOG"; then
        train_sets "$rung" > "$OVR_FILE"
        touch logs/SMOKE_PASSED
        log "smoke PASSED at rung $rung (training overrides: $(cat "$OVR_FILE"))"
        continue
      fi
      log "smoke rung $rung exited rc=$rc"
    fi
    n=$(bump smoke_rung)
    if [ "$n" -le 3 ]; then
      log "smoke rung $n: --set $(smoke_sets "$n")"
      # shellcheck disable=SC2046
      spawn smoke "$SMOKE_LOG" "$PY" scripts/train.py \
        --config configs/real_smoke.yaml \
        --config configs/local4090_20261031_real_smoke.yaml --set $(smoke_sets "$n")
    else
      park "smoke failed all 3 rungs (K=2 / K=1 / K=1-no-offload) — see $SMOKE_LOG"
    fi
  fi

  # ---- stage 3: bi training (gated on stats + smoke) ----------------
  if [ -f logs/SMOKE_PASSED ] && [ ! -f logs/TRAIN_DONE ] && ! alive train; then
    if [ -f logs/.wd_train.rc ]; then
      rc=$(cat logs/.wd_train.rc)
      rm -f logs/.wd_train.rc
      if [ "$rc" = "0" ] && grep -q "finished" "$TRAIN_LOG"; then
        touch logs/TRAIN_DONE
        log "training finished"
        continue
      fi
      log "train attempt $(cnt train_attempts) exited rc=$rc"
    fi
    n=$(bump train_attempts)
    if [ "$n" -le 3 ]; then
      ovr=$(cat "$OVR_FILE" 2>/dev/null || true)
      [ "$n" -ge 2 ] && ovr="$ovr trainer.per_device_batch_size=1"
      resume=""
      if [ -f "$RUN_DIR/ckpt_latest.pt" ]; then
        resume="--resume $RUN_DIR/ckpt_latest.pt"
      fi
      log "train attempt $n overrides='$ovr' resume='$resume'"
      # shellcheck disable=SC2086
      spawn train "$TRAIN_LOG" bash scripts/train_bi.sh "$BI_CFG" $resume --set $ovr
    else
      park "training failed $n attempts — see $TRAIN_LOG"
    fi
  fi

  sleep 20
done
