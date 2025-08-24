[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dasein93/devo-rl-jules/blob/main/notebooks/colab_quickstart.ipynb)

# Devo-RL-Jules

Predator–prey digital evolution + RL sandbox. This repository contains a PPO baseline for the PettingZoo MPE `simple_tag_v3` environment.

## Project Links
- [**Colab Quickstart Notebook**](notebooks/colab_quickstart.ipynb)
- [**Run Log**](docs/run_log.md)

## Quickstart (Local)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_cpu.py --config configs/base.yaml --episodes 50 --save_dir artifacts/
```

## Running the Training

You can start a training run using the `run_cpu.py` script. It reads hyperparameters from `configs/base.yaml` by default. You can override parameters via command-line arguments:

-   `--episodes`: Number of episodes to run.
-   `--save_dir`: Directory to save artifacts.

## Viewing Results

Artifacts for each run (including `metrics.csv` and `return.png`) are saved in a timestamped directory inside `artifacts/`.

To view the latest learning curve plot, you can use the provided helper script:

```bash
python tools/show_plot.py
```

## Troubleshooting on Colab
- If you face issues with the runtime, try "Reset runtime" from the "Runtime" menu.
- Make sure you are in the correct directory. The first cell in the notebook handles this, but if you run commands manually, you might need to use `%cd /content/devo-rl-jules`.
- If cloning fails or the repository state becomes corrupted, you can delete the directory with `!rm -rf devo-rl-jules` and re-clone it.
