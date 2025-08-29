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

### Trajectory Recording
To record trajectories, enable the `recording` section in your config file.

```yaml
recording:
  enabled: True
  sample_rate: 1 # Record every Nth episode
  # To record 2D positions for scatter plot replay
  pos_xy_idx: [2, 4] # Slice of the observation vector [start, end)
```

The recorded data is saved in the `traj/` directory of your run. Each episode is saved as a `.npz` file containing:
- `obs`: The observation vectors.
- `act`: The actions taken by the agents.
- `pos`: The (x, y) positions of the agents (if `pos_xy_idx` is set).
- `agent_names`: The names of the agents.

A `manifest.jsonl` file is also created, containing metadata for each episode.

### Replays (Positions)
A new, fast replay mode renders the 2D positions of agents directly from the recorded `.npz` files. This is much faster than the old method of re-simulating the environment.

**1. Record trajectories with position data:**
Make sure `recording.pos_xy_idx` is set in your config, as described above.

**2. Render positions from the latest run:**
```bash
# find the latest run directory
LATEST_RUN=$(ls -d artifacts/run_* | tail -n 1)

# render
python tools/replay.py --in $LATEST_RUN/traj \
  --out artifacts/latest_replay.mp4 \
  --mode positions --fps 24
```

The replay tool will automatically detect the presence of `pos` data in the `.npz` files. If it's not found, it will fall back to the heatmap mode. Predators (named "adversary*") are colored red, and prey (named "agent*") are colored blue.

**Additional flags:**
- `--fps`: Frames per second for the output video.
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
