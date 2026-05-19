"""End-to-end test: trainer runs the ecosystem env and produces sensible artifacts."""
import csv
import os

import numpy as np
import yaml

from run_train import main as run_train_main


def test_ecosystem_train_runs_and_logs_population(tmpdir):
    config = {
        "seed": 42,
        "device": "cpu",
        "env": {
            "id": "ecosystem",
            "max_steps": 100,
            "n_predators": 4,
            "n_prey": 4,
            "ecosystem": {
                # Tuned to ensure some mortality within 100 steps.
                "prey_max_hp": 15.0,
                "attack_damage": 10.0,
                "attack_range": 0.15,
                "pred_energy_cost": 0.3,
                "survival_bonus": 0.05,
            },
        },
        "train": {"total_episodes": 5, "lr": 1e-4, "hidden": 32, "centralized_critic": False},
        "league": {"enabled": False},
        "checkpoint": {"enabled": False},
        "recording": {"enabled": True, "sample_rate": 1},
        "logging": {},
    }
    config_path = os.path.join(tmpdir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    save_dir = os.path.join(tmpdir, "art")
    run_train_main(config_path, override_eps=5, save_dir=save_dir, device="cpu")

    run_dir = [os.path.join(save_dir, d) for d in os.listdir(save_dir) if d.startswith("run_")][0]
    metrics = os.path.join(run_dir, "metrics.csv")
    with open(metrics) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    assert "pred_pop_end" in header
    assert "prey_pop_end" in header

    # Population columns should be valid integers, and the env should produce
    # some captures (collisions resulting in damage), confirming the env is
    # actually exercising mortality machinery.
    pred_col = header.index("pred_pop_end")
    prey_col = header.index("prey_pop_end")
    cap_col = header.index("captures")
    for r in rows[1:]:
        assert int(r[pred_col]) >= 0
        assert int(r[prey_col]) >= 0
    total_captures = sum(int(r[cap_col]) for r in rows[1:])
    assert total_captures > 0, "Expected at least some captures across the episodes"


def test_ecosystem_recorder_alive_mask(tmpdir):
    config = {
        "seed": 42,
        "device": "cpu",
        "env": {
            "id": "ecosystem",
            "max_steps": 50,
            "n_predators": 3,
            "n_prey": 3,
            "ecosystem": {
                "max_predators": 3, "max_prey": 3,  # disable reproduction by capping at start
                "prey_max_hp": 10.0, "attack_damage": 20.0, "attack_range": 0.2,
            },
        },
        "train": {"total_episodes": 2, "lr": 1e-4, "hidden": 32, "centralized_critic": False},
        "league": {"enabled": False},
        "checkpoint": {"enabled": False},
        "recording": {"enabled": True, "sample_rate": 1},
        "logging": {},
    }
    config_path = os.path.join(tmpdir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    save_dir = os.path.join(tmpdir, "art")
    run_train_main(config_path, override_eps=2, save_dir=save_dir, device="cpu")

    run_dir = [os.path.join(save_dir, d) for d in os.listdir(save_dir) if d.startswith("run_")][0]
    npz_path = os.path.join(run_dir, "traj", "ep_2.npz")
    assert os.path.exists(npz_path)

    data = np.load(npz_path, allow_pickle=True)
    assert "alive" in data.files, "Recorder should write an alive mask for ecosystem episodes"
    assert "pos" in data.files
    alive = data["alive"]
    pos = data["pos"]
    assert alive.shape[0] == pos.shape[0]
    assert alive.shape[1] == pos.shape[1] == 6
    # With reproduction disabled (max == start) and damaging predators, alive count
    # should be monotonically non-increasing.
    counts = alive.sum(axis=1)
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)), \
        f"Alive count should monotonically decrease when reproduction is off; got {counts}"
