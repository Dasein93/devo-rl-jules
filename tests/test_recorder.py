import os
import shutil
import yaml
import numpy as np
import glob
from run_train import main as run_train_main

def test_recorder_ragged_obs(tmpdir):
    # Create a temporary config file
    config = {
        'seed': 42,
        'device': 'cpu',
        'env': {
            'max_steps': 20,
            'n_predators': 2, # Different obs sizes
            'n_prey': 1,
        },
        'train': {
            'total_episodes': 2,
            'lr': 1e-4,
        },
        'checkpoint': {
            'enabled': False,
        },
        'recording': {
            'enabled': True,
            'sample_rate': 1,
        },
        'logging': {}
    }
    config_path = os.path.join(tmpdir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run training for 2 episodes
    save_dir = os.path.join(tmpdir, "artifacts")
    run_train_main(config_path, override_eps=2, save_dir=save_dir, device='cpu')

    # Find the run directory
    run_dirs = [d for d in os.listdir(save_dir) if d.startswith('run_')]
    assert len(run_dirs) == 1
    run_dir = os.path.join(save_dir, run_dirs[0])

    # Check that a trajectory file was saved
    traj_dir = os.path.join(run_dir, "traj")
    assert os.path.exists(traj_dir)

    # Check for the last episode's npz file
    traj_path = os.path.join(traj_dir, "ep_2.npz")
    assert os.path.exists(traj_path), f"Expected trajectory file not found at {traj_path}"

    # Load the trajectory and check its contents
    data = np.load(traj_path)
    assert "obs" in data
    assert "act" in data

    obs = data["obs"]
    act = data["act"]

    assert obs.ndim == 2, f"obs should be 2D, but has shape {obs.shape}"
    assert act.ndim == 2, f"act should be 2D, but has shape {act.shape}"
    # obs is (T*A, D), act is (T, A). This is a bit awkward.
    # The number of agents is obs.shape[0] / act.shape[0]
    assert obs.shape[0] % act.shape[0] == 0, f"obs length should be a multiple of act length"
