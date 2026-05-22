# VPS operations for devo-rl-jules

Two scripts to deploy and run long experiments on the user's VPS (`stepan-vps`,
2 vCPUs / 8 GB RAM / Ubuntu 24.04 / no GPU).

| Script | Where it runs | Purpose |
|---|---|---|
| `scripts/vps_bootstrap.sh` | From your Mac (or any client with ssh+rsync) | One-shot deploy: rsync code, install OS deps + venv + CPU-only PyTorch + project requirements, run `pytest -q`. Idempotent. |
| `scripts/run_long.sh` | On the VPS, inside a tmux session | Wraps `run_train.py` with unbuffered Python, thread caps for the 2-vCPU box, and structured logging. |

## What this repo trains

devo-rl-jules is a predator/prey co-evolution sandbox. Each training run drives
two PPO policies (predators vs prey) on either PettingZoo MPE `simple_tag_v3`
or a custom `ecosystem` env that adds mortality, reproduction, food field, and
a heritable 3-trait genome. Default entrypoint: `python run_train.py --config
configs/<name>.yaml --episodes N --save_dir <path> --device cpu`. Configs of
interest are in `configs/`: `base.yaml` is the simple_tag baseline,
`ecosystem.yaml` is the full ecosystem env, `ecosystem_lv.yaml` is tuned for
Lotka-Volterra dynamics.

## Quick start

```bash
# (from the Mac)
ssh stepan-vps "uname -a"                                # sanity-check the alias
./scripts/vps_bootstrap.sh                               # ~3 min first time, faster after
ssh stepan-vps "cd /opt/devo-rl-jules && \
  tmux new-session -d -s eco_400 \
    -e CONFIG=configs/ecosystem.yaml \
    -e EPISODES=400 \
    -e SAVE_DIR=runs/eco_400 \
    -e LOG=logs/eco_400.log \
    './scripts/run_long.sh'"
ssh stepan-vps "tail -F /opt/devo-rl-jules/logs/eco_400.log"
# … later …
rsync -avh stepan-vps:/opt/devo-rl-jules/runs/eco_400/ ./results/eco_400/
```

## Calibrating run length before committing

The VPS is genuinely slow (2 vCPUs). Always time a small run first:

```bash
ssh stepan-vps "cd /opt/devo-rl-jules && \
  time ./venv/bin/python run_train.py \
    --config configs/ecosystem.yaml --episodes 10 \
    --save_dir runs/_smoke --device cpu"
```

Multiply the resulting wall-clock by your target episode count. Anything above
24 h is probably better done on a cloud GPU.

## Things to never do

- Do **not** drop `--exclude runs --exclude logs --exclude artifacts` from the
  rsync. Without them, re-running `vps_bootstrap.sh` deletes your training
  outputs (real bug from a prior project).
- Do **not** set env vars before `tmux new-session` and expect them to reach
  the wrapped command. Use `tmux new-session -e VAR=val ...` per the launch
  pattern above.
- Do **not** add `pytest` to `requirements.txt` — it's a dev/test dep,
  installed by the bootstrap.
- Do **not** `apt upgrade` the VPS. Other projects share its venv root.
- Do **not** bind any new service to `0.0.0.0`. Use `127.0.0.1` + SSH tunnel.

## Pulling results back

Trajectory `.npz` files are large (~1 GB per 300-ep ecosystem run); you
probably want to copy only the `metrics.csv`, `plots/`, and `checkpoints/`
back, not the whole `traj/`:

```bash
rsync -avh \
  --exclude 'traj/' \
  stepan-vps:/opt/devo-rl-jules/runs/eco_400/ \
  ./results/eco_400/
```

For a final demo replay, render the MP4 *on the VPS* with `tools/replay.py`,
then rsync just the MP4 back.
