#!/usr/bin/env bash
# Deploy devo-rl-jules to the VPS. Idempotent — safe to re-run.
#
# Usage (from the Mac or any client with rsync+ssh):
#   ./scripts/vps_bootstrap.sh
#
# Env knobs:
#   VPS_HOST    — SSH alias (default: stepan-vps)
#   PROJ        — VPS-side project dir (default: /opt/devo-rl-jules)
#   PY          — system Python to seed the venv (default: python3)

set -euo pipefail

VPS_HOST="${VPS_HOST:-stepan-vps}"
PROJ_REMOTE="${PROJ:-/opt/devo-rl-jules}"
PY="${PY:-python3}"

# 1. Reachability.
echo "[bootstrap] verifying $VPS_HOST is reachable…"
ssh -o BatchMode=yes "$VPS_HOST" "uname -a && which $PY tmux rsync git ffmpeg"

# 2. OS-level deps. PettingZoo MPE imports pygame which needs SDL2 at runtime;
#    ffmpeg is needed by tools/replay.py for video rendering (already on the VPS
#    per the operations notes, but we verify).
echo "[bootstrap] ensuring OS packages…"
ssh "$VPS_HOST" 'set -e
  needed="libsdl2-2.0-0 libsdl2-image-2.0-0 libsdl2-mixer-2.0-0 libsdl2-ttf-2.0-0"
  missing=""
  for pkg in $needed; do
    dpkg -s "$pkg" >/dev/null 2>&1 || missing="$missing $pkg"
  done
  if [ -n "$missing" ]; then
    apt-get update -y && apt-get install -y --no-install-recommends $missing
  fi'

# 3. Project tree (idempotent).
ssh "$VPS_HOST" "mkdir -p $PROJ_REMOTE/{runs,logs,artifacts}"

# 4. Rsync code. CRITICAL: exclude every output dir from --delete so re-deploys
#    don't wipe long-training results. .gitignore already excludes these from
#    git but rsync doesn't read .gitignore.
echo "[bootstrap] rsyncing code to $VPS_HOST:$PROJ_REMOTE…"
rsync -avh --delete \
  --exclude .git --exclude venv --exclude __pycache__ \
  --exclude .pytest_cache --exclude '*.pyc' --exclude '.venv' \
  --exclude artifacts --exclude runs --exclude logs \
  --exclude '*.npz' --exclude '*.mp4' --exclude '*.csv' \
  ./ "$VPS_HOST:$PROJ_REMOTE/"

# 5. Virtualenv + CPU-only PyTorch + project requirements + pytest.
#    CPU-only torch wheel is much smaller and the VPS has no GPU.
echo "[bootstrap] preparing venv + dependencies…"
ssh "$VPS_HOST" "set -e
  cd $PROJ_REMOTE
  if [ ! -d venv ]; then
    $PY -m venv venv
    ./venv/bin/pip install --upgrade pip wheel
    ./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
    ./venv/bin/pip install -r requirements.txt
    ./venv/bin/pip install pytest
  else
    ./venv/bin/pip show pytest >/dev/null 2>&1 || ./venv/bin/pip install pytest
  fi"

# 6. Verify with the test suite. Skip the slow replay end-to-end test (it
#    invokes run_train.py via subprocess; we just want a fast smoke). 'true'
#    so a single broken test doesn't fail the bootstrap.
echo "[bootstrap] running pytest -q…"
ssh "$VPS_HOST" "cd $PROJ_REMOTE && \
  ./venv/bin/python -m pytest -q --ignore=tests/test_replay_positions.py || true"

echo "[bootstrap] done. SSH in and launch with scripts/run_long.sh"
