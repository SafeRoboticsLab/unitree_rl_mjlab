#!/usr/bin/env bash
# Non-Slurm launcher for unitree_rl_mjlab training.
# Target: workstation with 2x RTX PRO 6000 + 64 CPU cores.
# Equivalent to run_train.slurm but for direct execution.
#
# Usage:
#   ./run_train.sh                                  # default task, both GPUs
#   ./run_train.sh Unitree-Go2-Walking              # override task
#   ./run_train.sh Unitree-Go2-Safety-Shield 8192   # override task + num_envs
#   GPUS=0 ./run_train.sh                           # single GPU (index 0)
#   GPUS=1 ./run_train.sh                           # single GPU (index 1)
#   GPUS=0,1 ./run_train.sh                         # both GPUs (default)

set -euo pipefail

# --- Resolve project dir; run with CWD on scratch so all artifacts land there ---
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch/gpfs/FISAC/nr1053/unitree_rl_mjlab}"
mkdir -p "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

# --- Defaults (override via positional args or env) ---
TASK="${1:-Unitree-Go2-Safety-Shield}"
NUM_ENVS="${2:-2048}"
GPUS="${GPUS:-0,1}"   # comma-separated physical GPU indices

# --- Conda setup ---
# /etc/bashrc and conda's activate hooks reference unset vars; relax `-u`
# just for sourcing them, then re-enable for the rest of the script.
set +u
# shellcheck disable=SC1090
source ~/.bashrc
conda activate unitree_rl_mjlab
set -u

# --- Runtime env vars (mirror of run_train.slurm) ---
export WANDB_MODE=offline
export WANDB_DIR="$SCRATCH_DIR"   # offline wandb run dirs go under $SCRATCH_DIR/wandb/
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Pin visible GPUs. train.py reads CUDA_VISIBLE_DEVICES and indexes into it
# via --gpu-ids. With "all" it takes every entry below.
export CUDA_VISIBLE_DEVICES="$GPUS"

# Match EGL device to the first visible CUDA device. For multi-GPU,
# train.py overrides MUJOCO_EGL_DEVICE_ID per-rank inside run_train().
export EGL_DEVICE_ID=0

# --- CPU thread budget ---
# 64 cores / 2 worker processes -> ~32 cores each. Cap BLAS/OMP threads
# well below that to avoid oversubscription with PyTorch's own threadpools
# and the simulator's worker threads. Tune up if you see CPU underuse.
NUM_GPUS_USED="$(awk -F, '{print NF}' <<<"$GPUS")"
THREADS_PER_PROC=$(( 64 / NUM_GPUS_USED / 4 ))   # 8 with 2 GPUs
export OMP_NUM_THREADS="$THREADS_PER_PROC"
export MKL_NUM_THREADS="$THREADS_PER_PROC"
export OPENBLAS_NUM_THREADS="$THREADS_PER_PROC"
export NUMEXPR_NUM_THREADS="$THREADS_PER_PROC"

# --- Logging (mirror of slurm's %x_%A_%a.out, on scratch) ---
LOG_ROOT="$SCRATCH_DIR/local_runs"
mkdir -p "$LOG_ROOT"
TS="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_FILE="$LOG_ROOT/${TASK}_${TS}.log"

echo "[run_train.sh] task=$TASK num_envs=$NUM_ENVS gpus=$GPUS threads/proc=$THREADS_PER_PROC"
echo "[run_train.sh] cwd=$SCRATCH_DIR (rsl_rl logs -> logs/, wandb -> wandb/, MUJOCO_LOG.TXT here)"
echo "[run_train.sh] console log -> $LOG_FILE"

# --- Pick GPU mode for train.py ---
# Single GPU: pass a one-element list; multi-GPU: "all" triggers torchrunx.
if [[ "$NUM_GPUS_USED" -le 1 ]]; then
  GPU_ARG=(--gpu-ids 0)
else
  GPU_ARG=(--gpu-ids all)
fi

# --- Run ---
# 'stdbuf -oL' keeps tee output line-buffered so tailing the log is live.
stdbuf -oL python "$PROJECT_DIR/scripts/train.py" "$TASK" \
  --env.scene.num-envs="$NUM_ENVS" \
  "${GPU_ARG[@]}" \
  2>&1 | tee "$LOG_FILE"
