import os
import shutil
import yaml
import csv
from run_train import main as run_train_main

def test_checkpoint_resume(tmpdir):
    # Create a temporary config file
    config = {
        'seed': 42,
        'device': 'cpu',
        'env': {
            'id': 'mpe.simple_tag_v3',
            'max_steps': 50,
            'n_predators': 1,
            'n_prey': 1,
        },
        'train': {
            'total_episodes': 5,
            'lr': 1e-4,
        },
        'checkpoint': {
            'enabled': True,
            'every': 5,
        },
        'recording': {'enabled': False},
        'logging': {}
    }
    config_path = os.path.join(tmpdir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run training for 5 episodes
    save_dir = os.path.join(tmpdir, "artifacts")
    run_train_main(config_path, override_eps=5, save_dir=save_dir, device='cpu')

    # Find the run directory
    run_dirs = [d for d in os.listdir(save_dir) if d.startswith('run_')]
    assert len(run_dirs) == 1
    run_dir = os.path.join(save_dir, run_dirs[0])

    # Check that a checkpoint was saved
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    assert os.path.exists(checkpoints_dir)
    checkpoint_files = os.listdir(checkpoints_dir)
    assert len(checkpoint_files) == 1
    assert checkpoint_files[0] == "ckpt_5.pt"

    # Now, resume training from this checkpoint for 5 more episodes
    config['train']['total_episodes'] = 10
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    run_train_main(config_path, override_eps=10, save_dir=save_dir, device='cpu', resume_from=run_dir)

    # Check that the metrics file has 10 episodes
    metrics_path = os.path.join(run_dir, "metrics.csv")
    with open(metrics_path, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        assert len(data) == 11 # 10 episodes + header
        assert data[0] == ["episode", "return_mean"]
        assert data[-1][0] == "10"
