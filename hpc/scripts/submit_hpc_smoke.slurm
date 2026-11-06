#!/usr/bin/env bash
# EnergyWAM — Slurm wrapper for the bounded HPC smoke run.
#
# Total allocation is capped at 30 minutes, which is the budget agreed for the
# first real-data/real-weights GPU pass.  The runner itself enforces a 25-minute
# wall-clock deadline before starting any new stage (MAX_WALL_SECONDS).
#
# ---------------------------------------------------------------------------
# SITE SETTINGS — NOT invented here.  Root fills these in at submission time.
# The #SBATCH lines below are the portable minimum.  Depending on the site's
# GRES syntax you may need one of:
#     #SBATCH --gpus-per-node=1          (used below)
#     #SBATCH --gres=gpu:1
#     #SBATCH --gres=gpu:a100:1
# and the partition / account are passed on the sbatch command line rather
# than hard-coded here:
#     sbatch --partition=<PART> --account=<ACC> hpc/scripts/submit_hpc_smoke.slurm
# ---------------------------------------------------------------------------
#SBATCH --job-name=energywam_hpc_smoke
#SBATCH --output=logs/hpc_slurm/%x-%j.out
#SBATCH --error=logs/hpc_slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#
# No --signal handler is installed: the runner's own MAX_WALL_SECONDS budget
# (25 min) stops it from starting a stage it cannot finish before the 30-minute
# hard limit.  Add a --signal/B:USR1 trap here only if the site kills the whole
# process group and you need to flush logs first.

set -euo pipefail

echo "[slurm] job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "[slurm] start $(date -Is)"
echo "[slurm] allocated GPUs: ${SLURM_JOB_GPUS:-unset}"

# --- interpreter: root probes the remote candidates and exports PYTHON_BIN ---
: "${PYTHON_BIN:?PYTHON_BIN is not set.  Probe the remote interpreters first \
(e.g. /usr/bin/python3, module-provided python, or a project venv) and export \
PYTHON_BIN=<the arm64 one with torch + transformers>.}"

# --- storage layout ---------------------------------------------------------
export ENERGYWAM_STORAGE_ROOT="${ENERGYWAM_STORAGE_ROOT:-/home/bingxing2/home/scx8178/energywam_storage}"
export ENERGYWAM_ROOT="${ENERGYWAM_ROOT:-${ENERGYWAM_STORAGE_ROOT}/code/energyWAM}"

# --- resource parameterization ---------------------------------------------
# Exactly one GPU, single process, thread caps so the 8 allocated CPUs are not
# oversubscribed by OpenMP/MKL inside torch.
: "${CUDA_VISIBLE_DEVICES:?Slurm must expose the allocated GPU; refusing to guess device 0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false

# Explicit GPU-approval gate (project convention).  The wrapper sets it because
# running under Slurm IS the resource approval; unset it to abort.
export ENERGYWAM_ALLOW_GPU_TRAINING="${ENERGYWAM_ALLOW_GPU_TRAINING:-1}"

# Bounded knobs — 10 steps per training stage, 25 min soft wall budget.
export STEPS="${STEPS:-10}"
export MAX_WALL_SECONDS="${MAX_WALL_SECONDS:-1500}"

mkdir -p "${ENERGYWAM_ROOT}/logs/hpc_slurm"
cd "${ENERGYWAM_ROOT}"

echo "[slurm] ENERGYWAM_ROOT=${ENERGYWAM_ROOT}"
echo "[slurm] PYTHON_BIN=${PYTHON_BIN}"
echo "[slurm] STORAGE_ROOT=${ENERGYWAM_STORAGE_ROOT}"
echo "[slurm] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} STEPS=${STEPS}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "[slurm] nvidia-smi unavailable"

rc=0
bash hpc/scripts/run_hpc_smoke.sh || rc=$?
echo "[slurm] end $(date -Is) rc=${rc}"
exit ${rc}
