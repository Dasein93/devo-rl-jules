import os
import json
import numpy as np
import pytest
from tools.replay import make_video

@pytest.fixture
def dummy_trajectory(tmp_path):
    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()

    # Create manifest.json
    manifest = {
        "env_id": "mpe.simple_tag_v3",
        "env_cfg": {
            "num_adversaries": 1,
            "num_good": 1,
            "max_cycles": 25,
            "continuous_actions": False,
        },
        "global_seed": 42,
        "episode_seeds": [42, 43],
        "agent_names": ["adversary_0", "agent_0"],
    }
    with open(traj_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    # Create a dummy trajectory file
    ep1_path = traj_dir / "ep_1.npz"
    actions = np.random.randint(0, 5, size=(10, 2))  # 10 steps, 2 agents
    np.savez_compressed(ep1_path, act=actions, obs=np.zeros((10, 2, 10)))

    return str(traj_dir)

def test_make_video_positions_mode(dummy_trajectory, tmp_path):
    """
    Test that make_video in 'positions' mode creates a non-empty MP4 file.
    """
    output_mp4 = tmp_path / "replay.mp4"

    try:
        n, used = make_video(
            trajectory_path=dummy_trajectory,
            out_mp4=str(output_mp4),
            fps=10,
            mode="positions",
            frame_stride=5, # speed up test
        )
    except ImportError as e:
        pytest.skip(f"Skipping replay test, missing dependency: {e}")


    assert n == 1
    assert os.path.exists(output_mp4)
    assert os.path.getsize(output_mp4) > 0

def test_make_video_fallback_to_heatmap(dummy_trajectory, tmp_path, capsys):
    """
    Test that make_video falls back to heatmap mode if manifest is missing.
    """
    output_mp4 = tmp_path / "replay_heatmap.mp4"

    # Remove manifest to trigger fallback
    os.remove(os.path.join(dummy_trajectory, "manifest.json"))

    n, used = make_video(
        trajectory_path=dummy_trajectory,
        out_mp4=str(output_mp4),
        fps=10,
        mode="positions", # request positions
    )

    assert n == 1
    assert os.path.exists(output_mp4)
    assert os.path.getsize(output_mp4) > 0

    captured = capsys.readouterr()
    assert "Warning: manifest.json not found" in captured.out
    assert "Falling back to heatmap mode" in captured.out
