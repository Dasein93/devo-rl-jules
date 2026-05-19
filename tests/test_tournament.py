import os
import csv
import yaml

from run_train import main as run_train_main
from tools.tournament import run_tournament, _elo
import numpy as np


def test_tournament_runs_and_writes_artifacts(tmpdir):
    config = {
        'seed': 42,
        'device': 'cpu',
        'env': {'id': 'mpe.simple_tag_v3', 'max_steps': 15, 'n_predators': 1, 'n_prey': 1},
        'train': {'total_episodes': 6, 'lr': 1e-4, 'hidden': 32},
        # snapshot every 2 episodes → with 6 episodes we get 3 snapshots per team.
        'league': {'enabled': True, 'snapshot_every': 2, 'opponent_prob': 0.0,
                   'max_snapshots': 5},
        'checkpoint': {'enabled': False},
        'recording': {'enabled': False},
        'logging': {},
    }
    config_path = os.path.join(tmpdir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    save_dir = os.path.join(tmpdir, "art")
    run_train_main(config_path, override_eps=6, save_dir=save_dir, device='cpu')

    run_dir = sorted([os.path.join(save_dir, d) for d in os.listdir(save_dir) if d.startswith("run_")])[0]
    out_dir = os.path.join(run_dir, "tournament")

    result = run_tournament(
        run_dir=run_dir, episodes=1, out_dir=out_dir,
        config_path=config_path, device='cpu', seed=42, deterministic=True,
    )

    # 3 snapshots per team → 3x3 matrix
    assert result["cap_mat"].shape == (3, 3)
    assert result["pred_elo"].shape == (3,)
    assert result["prey_elo"].shape == (3,)

    # Expected output files
    assert os.path.exists(os.path.join(out_dir, "scores.csv"))
    assert os.path.exists(os.path.join(out_dir, "ratings.csv"))
    assert os.path.exists(os.path.join(out_dir, "heatmap.png"))

    with open(os.path.join(out_dir, "scores.csv")) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["pred", "prey", "mean_captures", "mean_ep_steps", "mean_pred_return", "mean_prey_return"]
    assert len(rows) == 1 + 3 * 3

    with open(os.path.join(out_dir, "ratings.csv")) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["team", "snapshot", "elo"]
    # 3 predator + 3 prey ratings
    assert len(rows) == 1 + 6


def test_elo_predator_dominance_yields_higher_pred_rating():
    """If every pred beats every prey, all pred ratings should rise above starting 1500."""
    cap_mat = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    pred_elo, prey_elo = _elo(cap_mat, k=32.0, iters=5)
    assert (pred_elo > 1500).all()
    assert (prey_elo < 1500).all()


def test_elo_mixed_outcomes_keeps_ratings_finite():
    cap_mat = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    pred_elo, prey_elo = _elo(cap_mat)
    assert np.all(np.isfinite(pred_elo))
    assert np.all(np.isfinite(prey_elo))
