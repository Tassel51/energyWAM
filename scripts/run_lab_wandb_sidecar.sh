#!/usr/bin/env bash
set -u
cd /root/zzc/energyWAM
export HTTPS_PROXY="${ENERGYWAM_WANDB_PROXY:-http://127.0.0.1:17902}"
export HTTP_PROXY="$HTTPS_PROXY"
export https_proxy="$HTTPS_PROXY"
export http_proxy="$HTTPS_PROXY"
if [ "${1:-}" != "--upload-only" ]; then
  .venv-wandb/bin/python -u scripts/sync_lab_wandb.py >> logs/lab_remote_setup/wandb_mirror.log 2>&1 &
  mirror_pid=$!
  trap 'kill "$mirror_pid" 2>/dev/null || true' EXIT
fi
while true; do
  .venv-wandb/bin/python -u scripts/watch_wandb.py --settings configs/lab_wandb_bridge.json >> logs/lab_remote_setup/wandb_upload.log 2>&1
  sleep 15
done
