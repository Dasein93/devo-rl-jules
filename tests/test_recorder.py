import os
import yaml
import numpy as np

from run_train import main as run_train_main


def test_recorder_records_per_team_obs(tmpdir):
    config = {
        'seed': 42,
        'device': 'cpu',
        'env': {
            'id': 'mpe.simple_tag_v3',
            'max_steps': 20,
            'n_predators': 2,   # different obs sizes between teams
            'n_prey': 1,
        },
        'train': {
            'total_episodes': 2,
            'lr': 1e-4,
        },
        'league': {'enabled': False},
        'checkpoint': {'enabled': False},
        'recording': {'enabled': True, 'sample_rate': 1},
        'logging': {},
    }
    config_path = os.path.join(tmpdir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    save_dir = os.path.join(tmpdir, "artifacts")
    run_train_main(config_path, override_eps=2, save_dir=save_dir, device='cpu')

    run_dirs = [d for d in os.listdir(save_dir) if d.startswith('run_')]
    assert len(run_dirs) == 1
    run_dir = os.path.join(save_dir, run_dirs[0])

    traj_path = os.path.join(run_dir, "traj", "ep_2.npz")
    assert os.path.exists(traj_path)

    data = np.load(traj_path, allow_pickle=True)
    assert {"obs", "act", "agent_names"} <= set(data.files)

    obs, act = data["obs"], data["act"]
    assert obs.ndim == 3, f"obs should be (T,A,D), got shape {obs.shape}"
    assert act.ndim == 2, f"act should be (T,A), got shape {act.shape}"
    assert obs.shape[0] == act.shape[0], "T must match between obs and act"
    assert obs.shape[1] == act.shape[1] == 3, "A=2 predators + 1 prey"

    # Manifest with roles
    manifest_path = os.path.join(run_dir, "traj", "manifest.json")
    assert os.path.exists(manifest_path)
    import json
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["agent_roles"].count("predator") == 2
    assert manifest["agent_roles"].count("prey") == 1
