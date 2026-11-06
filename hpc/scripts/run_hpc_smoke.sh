#!/usr/bin/env bash
# EnergyWAM — bounded HPC smoke runner (real data, real weights, ARM64/A100).
#
# Pipeline (each training stage is hard-bounded to ${STEPS} steps, default 10;
# the whole thing is meant to fit inside a 30-minute Slurm allocation):
#
#   1. manifests  deterministic episode-disjoint train/val split (real data)
#   2. stats      paper §4.8 statistics fit  -> EXPLICITLY LABELLED SUBSET
#   3. bi         stage 1, 10 steps, batch 1, K=2 full unroll
#   4. joint      stage 2 from the bi checkpoint (switch_stage policy=reset)
#   5. resume     same-stage joint resume, 10 more steps
#   6. eval       offline action MSE on the HELD-OUT val manifest
#
# This is an ENGINEERING SMOKE.  It does not produce a scientific result.
# Nothing here changes the method: language is trained, LoRA is off, geometry
# is the real task geometry, the unroll is full, and the optimizer is untouched.
#
# Required environment (see hpc/docs/HPC_DEPLOYMENT.md):
#   PYTHON_BIN              remote python interpreter (probed by root; required)
#   ENERGYWAM_STORAGE_ROOT  remote storage root  (default below)
#   ENERGYWAM_ALLOW_GPU_TRAINING=1   explicit GPU approval gate
#   CUDA_VISIBLE_DEVICES    a single A100 with headroom
#
# Usage:
#   ENERGYWAM_ALLOW_GPU_TRAINING=1 CUDA_VISIBLE_DEVICES=0 \
#     PYTHON_BIN=<remote python> bash hpc/scripts/run_hpc_smoke.sh
#
# Stage selection (all on by default, set to 0 to skip):
#   DO_MANIFESTS=1 DO_STATS=1 DO_TRAIN=1 DO_EVAL=1
set -euo pipefail

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ENERGYWAM_ROOT="${ENERGYWAM_ROOT:-$(dirname "${HPC_DIR}")}"
cd "${ENERGYWAM_ROOT}"

# Remote storage layout (root is deploying this; all overridable).
export ENERGYWAM_STORAGE_ROOT="${ENERGYWAM_STORAGE_ROOT:-/home/bingxing2/home/scx8178/energywam_storage}"
QWEN_DIR="${QWEN_DIR:-${ENERGYWAM_STORAGE_ROOT}/models/Qwen3-VL-4B-Instruct}"
DINO_DIR="${DINO_DIR:-${ENERGYWAM_STORAGE_ROOT}/models/dinov3-vits16-pretrain-lvd1689m}"
DATASET_ROOT="${DATASET_ROOT:-${ENERGYWAM_STORAGE_ROOT}/datasets/robocasa365/v1.0/target/atomic/OpenDrawer/20250816}"
DATA_MIX="${DATA_MIX:-lerobot}"

export PYTHONPATH="${ENERGYWAM_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

: "${PYTHON_BIN:?PYTHON_BIN is not set.  Root probes the remote interpreters; \
export PYTHON_BIN=<arm64 python with torch/transformers> before running.}"

CONFIG="${CONFIG:-hpc/configs/hpc_smoke_gpu.yaml}"
MANIFEST_DIR="${MANIFEST_DIR:-${ENERGYWAM_ROOT}/manifests/hpc_opendrawer}"
STATS_BUNDLE="${STATS_BUNDLE:-${ENERGYWAM_ROOT}/stats/hpc_smoke_subset_paper_standardized_v1.json}"
RUN_DIR="${RUN_DIR:-${ENERGYWAM_ROOT}/logs/runs/hpc_smoke_gpu_v1}"
LOG_DIR="${LOG_DIR:-${ENERGYWAM_ROOT}/logs/hpc_smoke}"

# --- bounded knobs -------------------------------------------------------
STEPS="${STEPS:-10}"                 # per training stage
SEED="${SEED:-20261029}"
VAL_FRAC="${VAL_FRAC:-0.1}"
MANIFEST_MAX_EPISODES="${MANIFEST_MAX_EPISODES:-}"   # empty = all episodes
STATS_MAX_SAMPLES="${STATS_MAX_SAMPLES:-64}"         # -> coverage=subset
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-16}"

DO_MANIFESTS="${DO_MANIFESTS:-1}"
DO_STATS="${DO_STATS:-1}"
DO_TRAIN="${DO_TRAIN:-1}"
DO_EVAL="${DO_EVAL:-1}"

mkdir -p "${LOG_DIR}" "${RUN_DIR}" "$(dirname "${STATS_BUNDLE}")" "${MANIFEST_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUNLOG="${LOG_DIR}/run_${STAMP}.log"

log() { printf '[hpc-smoke] %s\n' "$*" | tee -a "${RUNLOG}"; }
die() { printf '[hpc-smoke] FATAL: %s\n' "$*" | tee -a "${RUNLOG}" >&2; exit 1; }

# Wall-clock guard: the whole pipeline is budgeted to fit a 30-minute Slurm
# allocation, so no new heavy stage is started past ${MAX_WALL_SECONDS}
# (default 25 min).  Already-started stages are allowed to finish; this only
# stops us from *starting* work we cannot complete.
START_TS="$(date +%s)"
MAX_WALL_SECONDS="${MAX_WALL_SECONDS:-1500}"
check_deadline() {
  local used=$(( $(date +%s) - START_TS ))
  if (( used > MAX_WALL_SECONDS )); then
    die "wall-clock budget exhausted (${used}s > ${MAX_WALL_SECONDS}s) before stage '$1'. \
Skipped stages can be re-run with DO_MANIFESTS=0 DO_STATS=0 and the existing checkpoints."
  fi
  log "wall-clock: ${used}s used of ${MAX_WALL_SECONDS}s budget (before '$1')"
}

# --------------------------------------------------------------------------
# guard rails — the project's GPU-approval convention
# --------------------------------------------------------------------------
: "${ENERGYWAM_ALLOW_GPU_TRAINING:?\
Refusing to start a real training run without GPU approval.
Set ENERGYWAM_ALLOW_GPU_TRAINING=1 and CUDA_VISIBLE_DEVICES to one A100 with
enough free memory, after confirming the node is not oversubscribed.}"
export ENERGYWAM_ALLOW_GPU_TRAINING
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to a single GPU}"

[[ -f "${CONFIG}" ]] || die "config not found: ${CONFIG}"
[[ -d "${QWEN_DIR}" ]] || die "Qwen weights not found: ${QWEN_DIR}"
[[ -d "${DINO_DIR}" ]] || die "DINO weights not found: ${DINO_DIR}"
[[ -d "${DATASET_ROOT}" ]] || die "dataset not found: ${DATASET_ROOT}"
if [[ -f "${DATASET_ROOT}/${DATA_MIX}/meta/info.json" ]]; then
  RESOLVED_DATASET_ROOT="${DATASET_ROOT}/${DATA_MIX}"
elif [[ -f "${DATASET_ROOT}/meta/info.json" ]]; then
  RESOLVED_DATASET_ROOT="${DATASET_ROOT}"
else
  die "no LeRobot meta/info.json under ${DATASET_ROOT}/${DATA_MIX} or ${DATASET_ROOT}"
fi

# --------------------------------------------------------------------------
# resource parameterization (recorded with every run)
# --------------------------------------------------------------------------
COMMON_SET=(
  "backbone.base_vlm=${QWEN_DIR}"
  "backbone.dino_path=${DINO_DIR}"
  "data.data_root_dir=${DATASET_ROOT}"
  "data.data_mix=${DATA_MIX}"
)
{
  echo "stamp=${STAMP}"
  echo "host=$(hostname)"
  echo "python_bin=${PYTHON_BIN}"
  echo "storage_root=${ENERGYWAM_STORAGE_ROOT}"
  echo "config=${CONFIG}"
  echo "dataset_root=${DATASET_ROOT}"
  echo "data_mix=${DATA_MIX}"
  echo "qwen=${QWEN_DIR}"
  echo "dino=${DINO_DIR}"
  echo "steps_per_stage=${STEPS}"
  echo "seed=${SEED}"
  echo "val_frac=${VAL_FRAC}"
  echo "stats_max_samples=${STATS_MAX_SAMPLES}"
  echo "eval_max_samples=${EVAL_MAX_SAMPLES}"
  echo "omp_threads=${OMP_NUM_THREADS}"
  echo "max_wall_seconds=${MAX_WALL_SECONDS}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "slurm_job_id=${SLURM_JOB_ID:-none}"
  echo "slurm_gpus=${SLURM_JOB_GPUS:-unset}"
} > "${LOG_DIR}/resource_params_${STAMP}.txt"
log "resource parameters -> ${LOG_DIR}/resource_params_${STAMP}.txt"

# --------------------------------------------------------------------------
# 1. deterministic episode-disjoint manifests
# --------------------------------------------------------------------------
if [[ "${DO_MANIFESTS}" == "1" ]]; then
  check_deadline manifests
  log "stage: manifests (seed=${SEED} val_frac=${VAL_FRAC})"
  if [[ -n "${MANIFEST_MAX_EPISODES}" ]]; then
    MAX_EP_ARGS=(--max-episodes "${MANIFEST_MAX_EPISODES}")
  else
    MAX_EP_ARGS=()
  fi
  "${PYTHON_BIN}" hpc/scripts/make_episode_manifests.py \
    --dataset-root "${RESOLVED_DATASET_ROOT}" \
    --out-dir "${MANIFEST_DIR}" \
    --seed "${SEED}" --val-frac "${VAL_FRAC}" --test-frac 0 \
    ${MAX_EP_ARGS[@]+"${MAX_EP_ARGS[@]}"} \
    --overwrite 2>&1 | tee -a "${RUNLOG}"
else
  log "stage: manifests SKIPPED (DO_MANIFESTS=${DO_MANIFESTS})"
fi

TRAIN_MANIFEST="${MANIFEST_DIR}/train_episodes.json"
VAL_MANIFEST="${MANIFEST_DIR}/val_episodes.json"
[[ -f "${TRAIN_MANIFEST}" ]] || die "missing ${TRAIN_MANIFEST}"
[[ -f "${VAL_MANIFEST}" ]] || die "missing ${VAL_MANIFEST} (needed for held-out eval)"

# --------------------------------------------------------------------------
# 2. statistics fit — SMOKE SUBSET, explicitly labelled
# --------------------------------------------------------------------------
if [[ "${DO_STATS}" == "1" ]]; then
  check_deadline stats
  log "stage: stats fit --max-samples ${STATS_MAX_SAMPLES} (coverage=subset, SMOKE ONLY)"
  "${PYTHON_BIN}" scripts/fit_normalization_stats.py \
    --config "${CONFIG}" --set "${COMMON_SET[@]}" \
    --split train \
    --episodes-manifest "${TRAIN_MANIFEST}" \
    --val-manifest "${VAL_MANIFEST}" \
    --max-samples "${STATS_MAX_SAMPLES}" \
    --out "${STATS_BUNDLE}" --overwrite 2>&1 | tee -a "${RUNLOG}"
else
  log "stage: stats SKIPPED (DO_STATS=${DO_STATS}); using ${STATS_BUNDLE}"
fi
[[ -f "${STATS_BUNDLE}" ]] || die "missing statistics bundle ${STATS_BUNDLE}"

# --------------------------------------------------------------------------
# helper: run one training stage; leaves the checkpoint path in ${LAST_CKPT}
# --------------------------------------------------------------------------
LAST_CKPT=""
run_stage() {
  local label="$1" stage="$2" outdir="$3" runid="$4"; shift 4
  check_deadline "train:${label}"
  log "---- train stage ${label}: stage=${stage} steps=${STEPS} -> ${outdir}"
  local logf="${LOG_DIR}/${label}_${STAMP}.log"
  "${PYTHON_BIN}" scripts/train.py --config "${CONFIG}" \
    --set "${COMMON_SET[@]}" \
      "stage.stage=${stage}" \
      "trainer.output_dir=${outdir}" \
      "run_id=${runid}" \
      "data.norm_stats_bundle_path=${STATS_BUNDLE}" \
      "trainer.max_train_steps=${STEPS}" \
    --smoke ${1+"$@"} 2>&1 | tee "${logf}" | tee -a "${RUNLOG}"
  local ckpt
  ckpt="$(sed -n 's/^\[EnergyWAM\] checkpoint: //p' "${logf}" | tail -1)"
  if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    die "stage ${label} produced no checkpoint (see ${logf})"
  fi
  log "stage ${label} checkpoint: ${ckpt}"
  LAST_CKPT="${ckpt}"
}

BI_CKPT=""; JOINT_CKPT=""; RESUME_CKPT=""

if [[ "${DO_TRAIN}" == "1" ]]; then
  # ---- 3. stage 1: bi -----------------------------------------------------
  run_stage bi bi "${RUN_DIR}/bi" hpc_smoke_bi_v1
  BI_CKPT="${LAST_CKPT}"
  # ---- 4. stage 2: joint from the bi checkpoint --------------------------
  run_stage joint joint "${RUN_DIR}/joint" hpc_smoke_joint_v1 \
    --init-from "${BI_CKPT}" --stage-switch-policy reset
  JOINT_CKPT="${LAST_CKPT}"
  # ---- 5. same-stage resume ----------------------------------------------
  run_stage joint_resume joint "${RUN_DIR}/joint_resume" hpc_smoke_joint_resume_v1 \
    --resume "${JOINT_CKPT}"
  RESUME_CKPT="${LAST_CKPT}"
else
  log "stage: train SKIPPED (DO_TRAIN=${DO_TRAIN})"
  BI_CKPT="$(ls -t "${RUN_DIR}"/bi/*_step*.pt 2>/dev/null | head -1 || true)"
  JOINT_CKPT="$(ls -t "${RUN_DIR}"/joint/*_step*.pt 2>/dev/null | head -1 || true)"
  RESUME_CKPT="$(ls -t "${RUN_DIR}"/joint_resume/*_step*.pt 2>/dev/null | head -1 || true)"
  log "existing checkpoints: bi=${BI_CKPT:-none} joint=${JOINT_CKPT:-none} resume=${RESUME_CKPT:-none}"
fi

# --------------------------------------------------------------------------
# 6. offline evaluation on the HELD-OUT val manifest
# --------------------------------------------------------------------------
if [[ "${DO_EVAL}" == "1" ]]; then
  check_deadline eval
  CKPT_FOR_EVAL="${RESUME_CKPT:-${JOINT_CKPT}}"
  [[ -n "${CKPT_FOR_EVAL}" && -f "${CKPT_FOR_EVAL}" ]] || die "no checkpoint to evaluate"
  log "stage: offline held-out eval (val manifest, max-samples=${EVAL_MAX_SAMPLES})"
  log "NOTE: offline action MSE against the demonstration — NOT a closed-loop success rate."
  "${PYTHON_BIN}" scripts/evaluate.py --config "${CONFIG}" \
    --set "${COMMON_SET[@]}" \
      "data.norm_stats_bundle_path=${STATS_BUNDLE}" \
    --checkpoint "${CKPT_FOR_EVAL}" \
    --mode joint \
    --eval-episodes-manifest "${VAL_MANIFEST}" \
    --max-samples "${EVAL_MAX_SAMPLES}" \
    --out "${LOG_DIR}/eval_report_${STAMP}.json" 2>&1 | tee -a "${RUNLOG}"
  log "eval report -> ${LOG_DIR}/eval_report_${STAMP}.json"
else
  log "stage: eval SKIPPED (DO_EVAL=${DO_EVAL})"
fi

log "done.  bi=${BI_CKPT:-skipped} joint=${JOINT_CKPT:-skipped} resume=${RESUME_CKPT:-skipped}"
log "this was a BOUNDED ENGINEERING SMOKE — do not report its metrics as a result."
