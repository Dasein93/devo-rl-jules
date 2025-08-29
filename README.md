[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dasein93/devo-rl-jules/blob/main/notebooks/colab_quickstart.ipynb)

# Devo-RL-Jules

Predator–prey digital evolution + RL sandbox.

This repo is bootstrapped for Jules to extend.


## Quickstart

### CPU Training
To run training on a CPU:
```bash
python run_train.py --config configs/base.yaml --episodes 50 --device cpu
```

### GPU Training (Colab/RunPod)
To run training on a GPU, make sure you have a CUDA-compatible environment:
```bash
python run_train.py --config configs/base.yaml --episodes 200 --device cuda
```

### Resuming Training
To resume a previous run, use the `--resume_from` argument with the path to the run directory:
```bash
python run_train.py --config configs/base.yaml --episodes 100 --resume_from artifacts/run_YYYYMMDD_HHMM
```

### Replays (Heatmap)
The default replay mode visualizes agent observations as a feature heatmap. This is useful for debugging an agent's inputs.
```bash
# Render heatmap from the latest run
python tools/replay.py --in artifacts/run_2025*/traj --out artifacts/latest_heatmap.mp4 --mode heatmap
```

### Replays (Positions)
A new replay mode renders a 2D scatter plot of agent positions over time. This provides a much more intuitive and faster replay than the heatmap mode.

**1. Enable Position Recording:**
To record agent positions, you must enable the `recording` setting in your configuration file (e.g., `configs/base.yaml`).

```yaml
# In your config.yaml
recording:
  enabled: True
  sample_rate: 1 # Record every step
```

Then, run your training. The trajectory `.npz` files will now contain a `pos` array with shape `(T, A, 2)` and an `agent_names` array. A `manifest.json` will also be saved, containing agent roles ("predator" or "prey").

```bash
python run_train.py --config configs/base.yaml --episodes 50
```

**2. Render Position-Based Replay:**
Once you have recorded trajectories with position data, you can generate a video.

```bash
# Find the latest run directory
LATEST_RUN=$(ls -d artifacts/run_* | tail -n 1)

# Render the video
python tools/replay.py "$LATEST_RUN/traj" \
  --out "artifacts/latest_positions.mp4" \
  --mode "positions"
```

The replay script will automatically color agents based on their team (predators in red, prey in green). If a trajectory file is missing the `pos` array, it will be skipped with a warning.

**Additional flags:**
- `--fps`: Frames per second for the output video.
- `--frameskip`: Render every Nth frame to speed up video generation.
- `--dpi`: Resolution of the video.

## Artifacts
Each run directory in `artifacts/` contains:
- `checkpoints/`: Model checkpoints (`.pt`).
- `plots/`: Training curve plots (`.png`).
- `traj/`: Per-episode trajectory logs. Contains `.npz` files for each episode and a `manifest.json` with metadata.
- `metrics.csv`: CSV log of per-episode returns.

## Logs & Notebooks
- Run Log: `docs/run_log.md`
- Colab Quickstart: `notebooks/colab_quickstart.ipynb`


## Testing
Run the unit tests with:
```bash
pytest -q
```
