import os
import glob
import yaml

from run_train import main as run_train_main
from tools.eval import evaluate


def test_eval_runs_against_random_init_baseline(tmpdir):
    config = {
        'seed': 7,
        'device': 'cpu',
        'env': {'id': 'mpe.simple_tag_v3', 'max_steps': 20, 'n_predators': 1, 'n_prey': 1},
        'train': {'total_episodes': 1, 'lr': 1e-4, 'hidden': 32},
        'league': {'enabled': False},
        'checkpoint': {'enabled': False},
        'recording': {'enabled': False},
        'logging': {},
    }
    config_path = os.path.join(tmpdir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    out = evaluate(
        pred_ckpt=None, prey_ckpt=None,
        episodes=2, config_path=config_path,
        device='cpu', seed=7, deterministic=True,
    )
    assert len(out["pred_returns"]) == 2
    assert len(out["prey_returns"]) == 2
    assert all(t > 0 for t in out["ep_lengths"])


def test_eval_with_trained_checkpoints(tmpdir):
    config = {
        'seed': 11,
        'device': 'cpu',
        'env': {'id': 'mpe.simple_tag_v3', 'max_steps': 20, 'n_predators': 1, 'n_prey': 1},
        'train': {'total_episodes': 3, 'lr': 1e-3, 'hidden': 32},
        'league': {'enabled': False},
        'checkpoint': {'enabled': True, 'every': 3},
        'recording': {'enabled': False},
        'logging': {},
    }
    config_path = os.path.join(tmpdir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    save_dir = os.path.join(tmpdir, "art")
    run_train_main(config_path, override_eps=3, save_dir=save_dir, device='cpu')

    pred_ckpt = sorted(glob.glob(os.path.join(save_dir, "run_*", "checkpoints", "pred_*.pt")))[-1]
    prey_ckpt = sorted(glob.glob(os.path.join(save_dir, "run_*", "checkpoints", "prey_*.pt")))[-1]

    out = evaluate(
        pred_ckpt=pred_ckpt, prey_ckpt=prey_ckpt,
        episodes=2, config_path=config_path,
        device='cpu', seed=11, deterministic=True,
    )
    assert len(out["pred_returns"]) == 2
    assert len(out["prey_returns"]) == 2
