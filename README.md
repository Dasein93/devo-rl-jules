# Devo-RL-Jules

Predator–prey digital evolution + RL sandbox. This repository contains a PPO baseline for the PettingZoo MPE `simple_tag_v3` environment.

## Quickstart

### Local Setup (venv)

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd devo-rl-jules
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Google Colab Setup

1.  **Clone the repository in your Colab notebook:**
    ```python
    !git clone <repository_url>
    %cd devo-rl-jules
    ```

2.  **Install dependencies:**
    ```python
    !pip install -r requirements.txt
    ```

### Running the Training

You can start a training run using the `run_cpu.py` script. It reads hyperparameters from `configs/base.yaml` by default.

```bash
python run_cpu.py --config configs/base.yaml --episodes 500
```

You can override parameters via command-line arguments:

-   `--episodes`: Number of episodes to run.
-   `--save_dir`: Directory to save artifacts.

### Viewing Results

Artifacts for each run (including `metrics.csv` and `return.png`) are saved in a timestamped directory inside `artifacts/`.

To view the latest learning curve plot, you can use the provided helper script:

```bash
python tools/show_plot.py
```
