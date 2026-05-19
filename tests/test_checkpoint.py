import os
import csv
import yaml

from run_train import main as run_train_main


def _write_config(path, total_episodes=5, every=5):
    config = {
        'seed': 42,
        'device': 'cpu',
        'env': {
            'id': 'mpe.simple_tag_v3',
            'max_steps': 30,
            'n_predators': 1,
            'n_prey': 1,
        },
        'train': {
            'total_episodes': total_episodes,
            'lr': 1e-4,
        },
        'league': {'enabled': False},
        'checkpoint': {'enabled': True, 'every': every},
        'recording': {'enabled': False},
        'logging': {},
    }
    with open(path, 'w') as f:
        yaml.dump(config, f)
    return config


def test_checkpoint_resume(tmpdir):
    config_path = os.path.join(tmpdir, "config.yaml")
    _write_config(config_path, total_episodes=5, every=5)
    save_dir = os.path.join(tmpdir, "artifacts")

    run_train_main(config_path, override_eps=5, save_dir=save_dir, device='cpu')

    run_dirs = [d for d in os.listdir(save_dir) if d.startswith('run_')]
    assert len(run_dirs) == 1
    run_dir = os.path.join(save_dir, run_dirs[0])

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    files = sorted(os.listdir(ckpt_dir))
    assert files == ["pred_5.pt", "prey_5.pt"], f"unexpected checkpoint set: {files}"

    _write_config(config_path, total_episodes=10, every=5)
    run_train_main(config_path, override_eps=10, save_dir=save_dir, device='cpu', resume_from=run_dir)

    metrics_path = os.path.join(run_dir, "metrics.csv")
    with open(metrics_path) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    assert header[0] == "episode"
    assert "captures" in header and "ep_steps" in header
    assert rows[-1][0] == "10"
    assert len(rows) == 11  # header + 10 episodes
    # captures should be a non-negative integer-like value
    cap_col = header.index("captures")
    assert int(rows[-1][cap_col]) >= 0
