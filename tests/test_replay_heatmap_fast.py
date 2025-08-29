import os
import subprocess
import tempfile
import numpy as np
import pytest

# Mark this test as slow, as it involves file I/O and subprocesses.
@pytest.mark.slow
def test_replay_heatmap_fast_execution():
    """
    Test the replay script with fast settings to ensure it runs quickly and produces an output file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Generate a tiny fake .npz file
        npz_path = os.path.join(tmpdir, "test_episode.npz")
        # obs shape: (T, A, D)
        dummy_obs = np.random.rand(15, 4, 8)
        np.savez(npz_path, obs=dummy_obs)

        # 2. Prepare to run the replay script
        output_mp4_path = os.path.join(tmpdir, "replay.mp4")
        # Correctly locate the replay.py script relative to this test file.
        replay_script_path = os.path.join(
            os.path.dirname(__file__), "..", "tools", "replay.py"
        )

        command = [
            "python",
            replay_script_path,
            npz_path,
            "--out",
            output_mp4_path,
            "--mode",
            "heatmap",
            "--frame_stride",
            "3",
            "--dpi",
            "80",
            "--size",
            "480x270",
        ]

        # 3. Run the script with a timeout
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,  # 30-second timeout
            )
            # Optional: print output for debugging if needed
            # print("stdout:", result.stdout)
            # print("stderr:", result.stderr)
        except subprocess.TimeoutExpired as e:
            pytest.fail(
                f"Replay script took too long to execute (>30s).\n"
                f"stdout: {e.stdout.decode() if e.stdout else 'N/A'}\n"
                f"stderr: {e.stderr.decode() if e.stderr else 'N/A'}"
            )
        except subprocess.CalledProcessError as e:
            pytest.fail(
                f"Replay script failed with exit code {e.returncode}.\n"
                f"stdout: {e.stdout}\n"
                f"stderr: {e.stderr}"
            )

        # 4. Assert that the mp4 exists and is not empty
        assert os.path.exists(output_mp4_path), "Output MP4 file was not created."
        assert os.path.getsize(output_mp4_path) > 0, "Output MP4 file is empty."
