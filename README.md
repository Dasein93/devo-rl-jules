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
A new replay mode renders the actual 2D positions of agents (predators in red, prey in green) by reconstructing the environment state from recorded actions. This provides a much more intuitive view of the episode.

**1. Record during training:**
This is enabled by default in `configs/base.yaml`. A `manifest.json` is saved into the trajectory folder.
```bash
python run_train.py --config configs/base.yaml --episodes 50
```

**2. Render positions from the latest run:**
```bash
# find the latest run directory
LATEST_RUN=$(ls -d artifacts/run_* | tail -n 1)

# render
python tools/replay.py --in $LATEST_RUN/traj \
  --out artifacts/latest_replay.mp4 \
  --mode positions --fps 12
```

If `manifest.json` is missing, the replay tool will print a warning and fall back to heatmap mode.

**Additional flags:**
- `--fps`: Frames per second for the output video.
- `--frameskip`: Render every Nth frame to speed up video generation.
- `--dpi`: Resolution of the video.
- `--speed`: Adjust the playback speed in the title.

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
