[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dasein93/devo-rl-jules/blob/main/notebooks/colab_quickstart.ipynb)

# Devo-RL-Jules

Predator–prey digital evolution + RL sandbox on PettingZoo MPE `simple_tag_v3`.

- **Per-team PPO** — one policy for predators, one for prey, trained jointly.
- **MAPPO-style centralised critic** — each team's critic sees the concatenation
  of all its own agents' observations (gated by `train.centralized_critic`).
- **Evolutionary league** — past policy snapshots are sampled as opponents during rollouts, stabilising the co-adaptation between the two teams.
- **GAE, value-loss clipping, gradient clipping** in PPO.
- **Trajectory recording** with per-step positions for offline replay.
- **Two envs**: PettingZoo `simple_tag_v3` (fixed roster, no death) and a custom
  `ecosystem` env (per-agent HP / energy, mortality, reproduction, food field).
  Switch via `env.id` in the config — `configs/base.yaml` vs `configs/ecosystem.yaml`.

## Quickstart

```bash
pip install -r requirements.txt

# CPU on the classic simple_tag env
python run_train.py --config configs/base.yaml --episodes 200 --device cpu

# CPU on the ecosystem env (mortality + reproduction + food field)
python run_train.py --config configs/ecosystem.yaml --episodes 300 --device cpu

# GPU (Colab/RunPod)
python run_train.py --config configs/base.yaml --episodes 1000 --device cuda

# Resume from an existing run dir; auto-loads the newest pred_*.pt / prey_*.pt
python run_train.py --config configs/base.yaml --episodes 2000 --resume_from artifacts/run_YYYYMMDD_HHMMSS
```

## Evaluation

Run a trained pair against each other (or against a random-init baseline) without
learning. Glob patterns are expanded; the most recent match wins.

```bash
python tools/eval.py \
  --pred 'artifacts/run_*/checkpoints/pred_*.pt' \
  --prey 'artifacts/run_*/checkpoints/prey_*.pt' \
  --episodes 20

# Random baseline for either side: omit the flag.
python tools/eval.py --pred 'artifacts/run_*/checkpoints/pred_*.pt' --episodes 10
```

## Replay

The replay tool takes the trajectory directory (or a single `.npz`) as a
positional argument.

```bash
LATEST=$(ls -d artifacts/run_* | tail -n1)

# Heatmap of per-agent observations over time
python tools/replay.py "$LATEST/traj" --out "$LATEST/heatmap.mp4" --mode heatmap

# 2D positions, predators red, prey green; --trail N draws a fading tail
python tools/replay.py "$LATEST/traj" --out "$LATEST/positions.mp4" --mode positions --frameskip 2 --trail 20

# Ecosystem split panel — 2D scene with agents winking in/out next to an
# animated population-over-time curve. Requires the `ecosystem` env.
python tools/replay.py "$LATEST/traj" --out "$LATEST/ecosystem.mp4" --mode ecosystem --frameskip 2 --trail 20
```

Position replay requires `recording.enabled: true` in the config (default) so
that `traj/ep_*.npz` files contain a `pos` array.

## Comparing runs

Overlay learning curves from multiple runs — useful for ablations (MAPPO on/off,
league on/off, hyperparameter sweeps).

```bash
python tools/compare_runs.py \
  --runs artifacts/run_mappo artifacts/run_dec \
  --labels MAPPO Decentralised \
  --metric pred_return \
  --out compare.png
```

`--metric` is any column from `metrics.csv` (`pred_return`, `prey_return`,
`captures`, `ep_steps`, `pred_v_loss`, ...).

## Tournament

Score every (predator snapshot, prey snapshot) pair from a run's league against
each other, producing a capture-count heatmap and Elo ratings per snapshot per
team. Useful for detecting non-transitive cycles in the co-adapting league pool.

```bash
python tools/tournament.py --run artifacts/run_YYYYMMDD_HHMMSS --episodes 3
# Outputs into artifacts/<run>/tournament/:
#   scores.csv     # one row per (pred, prey) pair
#   ratings.csv    # Elo per snapshot per team
#   heatmap.png    # captures matrix
```

## Artifacts

Each `artifacts/run_<UTC timestamp>/` contains:

- `checkpoints/{pred,prey}_<ep>.pt` — per-team policy + optimizer snapshots.
- `league/{predator,prey}/snap_*.pt` — frozen snapshots periodically pushed into the league pool.
- `plots/return.png` — per-team return curves with moving-average overlays.
- `traj/ep_*.{npz,jsonl}` + `manifest.json` — per-episode recordings (when enabled).
- `metrics.csv` — episode, per-team returns, captures, episode length, PPO loss components, league size.
- `tournament/` — `scores.csv`, `ratings.csv`, `heatmap.png` after running `tools/tournament.py`.

## Configuration

`configs/base.yaml` is the single source of truth. Notable sections:

```yaml
train:
  gae_lambda: 0.95    # GAE smoothing
  minibatch_size: 1024
  max_grad_norm: 0.5

league:
  enabled: true
  snapshot_every: 25   # episodes between league snapshots per team
  opponent_prob: 0.3   # per-episode chance to replace one team with a frozen snapshot
  max_snapshots: 20    # FIFO eviction past this cap

recording:
  enabled: true
  sample_rate: 1
```

## Testing

```bash
pytest -q
```

The full suite includes a subprocess-based end-to-end test that runs PPO + the
replay video pipeline; it takes ~30 s. Skip it with `-k 'not replay'`.

## Logs & Notebooks

- Run log: `docs/run_log.md`
- Colab quickstart: `notebooks/colab_quickstart.ipynb`
