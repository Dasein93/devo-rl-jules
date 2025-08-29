import os
import shutil
import yaml
import numpy as np
import glob
from run_train import main as run_train_main
from tools.replay import main as replay_main

def test_record_and_replay_positions(tmpdir):
    # 1. Create a temporary config file for a tiny run
    config = {
        'seed': 42,
        'device': 'cpu',
        'env': {
            'id': 'mpe.simple_tag_v3',
            'max_steps': 20,
            'n_predators': 2,
            'n_prey': 2,
        },
        'train': {
            'total_episodes': 1,
            'lr': 1e-4,
        },
        'checkpoint': {'enabled': False},
        'recording': {
            'enabled': True,
            'sample_rate': 1,
            'pos_xy_idx': [2, 4], # For simple_tag_v3, obs[2:4] is agent's position
        },
        'logging': {}
    }
    config_path = os.path.join(tmpdir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # 2. Run training for 1 episode to generate a trajectory
    save_dir = os.path.join(tmpdir, "artifacts")
    run_train_main(config_path, override_eps=1, save_dir=save_dir, device='cpu')

    # 3. Verify the trajectory file and its contents
    run_dirs = [d for d in os.listdir(save_dir) if d.startswith('run_')]
    assert len(run_dirs) == 1, "Expected exactly one run directory"
    run_dir = os.path.join(save_dir, run_dirs[0])
    traj_dir = os.path.join(run_dir, "traj")
    assert os.path.isdir(traj_dir)

    # Check for the episode npz file
    npz_files = glob.glob(os.path.join(traj_dir, "ep_*.npz"))
    assert len(npz_files) == 1, "Expected one episode .npz file"
    traj_path = npz_files[0]

    # Load the trajectory and check for 'pos' and 'agent_names'
    data = np.load(traj_path)
    assert "pos" in data, "Trajectory file must contain 'pos' data"
    assert "agent_names" in data, "Trajectory file must contain 'agent_names' data"

    pos = data["pos"]
    agent_names = data["agent_names"]
    n_agents = config['env']['n_predators'] + config['env']['n_prey']
    assert pos.shape[1] == n_agents, f"pos shape {pos.shape} has wrong number of agents"
    assert pos.shape[2] == 2, f"pos should be 2D (x,y), but shape is {pos.shape}"
    assert len(agent_names) == n_agents

    # Check for manifest.jsonl
    manifest_path = os.path.join(traj_dir, "manifest.jsonl")
    assert os.path.exists(manifest_path), "manifest.jsonl was not created"
    with open(manifest_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1, "manifest.jsonl should have one line for one episode"

    # 4. Run replay in 'positions' mode and check for MP4 output
    replay_out_path = os.path.join(tmpdir, "replay.mp4")

    import sys
    original_argv = sys.argv
    sys.argv = ["replay.py", traj_dir, "--out", replay_out_path, "--mode", "positions"]
    try:
        replay_main()
    finally:
        sys.argv = original_argv

    assert os.path.exists(replay_out_path), "Replay script did not create the mp4 file"
    assert os.path.getsize(replay_out_path) > 1000, "Replay mp4 file is too small"
