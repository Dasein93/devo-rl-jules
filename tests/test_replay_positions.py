import os
import subprocess
import numpy as np
import glob
import yaml

def test_record_and_replay_positions(tmp_path):
    """
    Run a short training, record trajectories with positions,
    and then replay them as a positions scatter plot video.
    """
    # 1. Run training with recording enabled
    config_path = "configs/base.yaml"
    run_dir = os.path.join(tmp_path, "run")

    # Modify config on the fly to enable recording
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['recording'] = {
        'enabled': True,
        'sample_rate': 1
    }

    new_config_path = os.path.join(tmp_path, "test_config.yaml")
    with open(new_config_path, 'w') as f:
        yaml.dump(config, f)

    cmd = [
        "python", "run_train.py",
        "--config", new_config_path,
        "--episodes", "2",
        "--save_dir", run_dir
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Training run failed: {result.stderr}"

    # 2. Verify trajectory files
    run_subdirs = glob.glob(os.path.join(run_dir, "run_*"))
    assert len(run_subdirs) > 0, "No run subdirectory found"
    traj_path = os.path.join(run_subdirs[0], "traj")

    npz_files = sorted(glob.glob(os.path.join(traj_path, "*.npz")))
    assert len(npz_files) == 2, f"Expected 2 trajectory files, found {len(npz_files)}"

    for npz_file in npz_files:
        with np.load(npz_file, allow_pickle=True) as data:
            assert "pos" in data, f"'pos' array not found in {npz_file}"
            assert "agent_names" in data, f"'agent_names' array not found in {npz_file}"

            pos = data["pos"]
            agent_names = data["agent_names"]

            assert pos.ndim == 3, f"Expected pos.ndim==3, got {pos.ndim}"
            assert pos.shape[2] == 2, f"Expected pos.shape[2]==2, got {pos.shape[2]}"
            assert pos.shape[1] == len(agent_names), "Mismatch between pos shape and agent_names length"

    # 3. Run replay script
    out_mp4 = os.path.join(tmp_path, "replay.mp4")
    cmd_replay = [
        "python", "tools/replay.py",
        traj_path,
        "--mode", "positions",
        "--out", out_mp4,
    ]

    result_replay = subprocess.run(cmd_replay, capture_output=True, text=True, check=False)
    assert result_replay.returncode == 0, f"Replay script failed: {result_replay.stderr}"

    # 4. Verify video output
    assert os.path.exists(out_mp4), f"Output video not created: {out_mp4}"
    assert os.path.getsize(out_mp4) > 0, f"Output video is empty: {out_mp4}"

    print(f"Successfully created and verified replay video: {out_mp4}")
