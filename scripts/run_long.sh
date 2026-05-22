#!/usr/bin/env bash
# tmux entrypoint for long training runs of devo-rl-jules on the VPS.
# Wraps run_train.py with unbuffered Python, nice -19, and PyTorch thread
# caps suitable for the 2 vCPU box.
#
# Launch pattern (from the Mac):
#   ssh stepan-vps "cd /opt/devo-rl-jules && \
#     tmux new-session -d -s dense_400 \
#       -e CONFIG=configs/ecosystem.yaml \
#       -e EPISODES=400 \
#       -e SAVE_DIR=runs/dense_400 \
#       -e LOG=logs/dense_400.log \
#       './scripts/run_long.sh'"
# Then:
#   ssh stepan-vps "tail -F /opt/devo-rl-jules/logs/dense_400.log"
#
# `tmux new-session -e VAR=val` is the only reliable way to propagate env
# vars per session — see the operations notes (lesson #1).

set -uo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/ecosystem.yaml}"
EPISODES="${EPISODES:-400}"
SAVE_DIR="${SAVE_DIR:-runs/long}"
DEVICE="${DEVICE:-cpu}"
LOG="${LOG:-logs/long.log}"
# 2 vCPU box — cap PyTorch threads so we don't oversubscribe.
NUM_THREADS="${NUM_THREADS:-2}"
export OMP_NUM_THREADS="$NUM_THREADS"
export MKL_NUM_THREADS="$NUM_THREADS"
export OPENBLAS_NUM_THREADS="$NUM_THREADS"

mkdir -p logs runs

PY="${PY:-./venv/bin/python}"
if [ ! -x "$PY" ]; then PY="python3"; fi

echo "[$(date -Iseconds)] launching config=$CONFIG episodes=$EPISODES save_dir=$SAVE_DIR threads=$NUM_THREADS" | tee -a "$LOG"

nice -n 19 "$PY" -u run_train.py \
    --config "$CONFIG" \
    --episodes "$EPISODES" \
    --save_dir "$SAVE_DIR" \
    --device "$DEVICE" \
    >> "$LOG" 2>&1
rc=$?

echo "[$(date -Iseconds)] exited rc=$rc" | tee -a "$LOG"
exit $rc
