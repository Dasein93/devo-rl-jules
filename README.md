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

### Replaying Trajectories
To replay a saved trajectory and save it as an MP4 video:
```bash
python tools/replay.py artifacts/run_YYYYMMDD_HHMM/replays/ep_1.jsonl
```
Note: `imageio` requires `ffmpeg` to be installed for saving MP4 files.

## Logs & Notebooks
- Run Log: `docs/run_log.md`
- Colab Quickstart: `notebooks/colab_quickstart.ipynb`
