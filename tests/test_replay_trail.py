import os
import numpy as np
import imageio.v2 as imageio

from tools.replay import _render_episode_positions


def test_trail_renders_without_error(tmpdir):
    T, A = 30, 4
    rng = np.random.default_rng(0)
    positions = np.cumsum(rng.normal(scale=0.05, size=(T, A, 2)).astype(np.float32), axis=0)
    agent_names = ["adversary_0", "adversary_1", "agent_0", "agent_1"]

    out_path = str(tmpdir / "trail.mp4")
    writer = imageio.get_writer(out_path, fps=10, codec="libx264", bitrate="2000k", quality=8)
    try:
        _render_episode_positions(positions, agent_names, writer,
                                  title_prefix="test ", dpi=80, frameskip=2, trail=8)
    finally:
        writer.close()

    assert os.path.getsize(out_path) > 0


def test_no_trail_path_still_works(tmpdir):
    positions = np.zeros((10, 2, 2), dtype=np.float32)
    agent_names = ["adversary_0", "agent_0"]

    out_path = str(tmpdir / "notrail.mp4")
    writer = imageio.get_writer(out_path, fps=10, codec="libx264", bitrate="2000k", quality=8)
    try:
        _render_episode_positions(positions, agent_names, writer, trail=0)
    finally:
        writer.close()

    assert os.path.getsize(out_path) > 0
