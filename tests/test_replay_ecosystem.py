"""Tests for the ecosystem split-panel replay mode."""
import os

import imageio.v2 as imageio
import numpy as np

from tools.replay import _render_episode_ecosystem, make_video


def test_ecosystem_render_with_alive_mask(tmpdir):
    T, A = 20, 6
    rng = np.random.default_rng(0)
    positions = np.cumsum(rng.normal(scale=0.05, size=(T, A, 2)).astype(np.float32), axis=0)
    # 2 predators, 4 prey. Half the predators die at t=10, gaining a prey at t=15.
    agent_names = ["adversary_0", "adversary_1", "agent_0", "agent_1", "agent_2", "agent_3"]
    alive = np.ones((T, A), dtype=bool)
    alive[10:, 1] = False           # predator dies mid-episode
    alive[:15, 5] = False            # prey unborn until step 15

    pop_pred_global = alive[:, :2].sum(axis=1)
    pop_prey_global = alive[:, 2:].sum(axis=1)

    out_path = str(tmpdir / "eco.mp4")
    writer = imageio.get_writer(out_path, fps=10, codec="libx264", bitrate="2000k", quality=8)
    try:
        _render_episode_ecosystem(
            positions, alive, agent_names, writer,
            pop_pred_global=pop_pred_global, pop_prey_global=pop_prey_global,
            cumulative_offset=0, title_prefix="test ", dpi=80, frameskip=2, trail=5,
        )
    finally:
        writer.close()

    assert os.path.getsize(out_path) > 0


def test_make_video_ecosystem_mode_uses_pop_curves(tmpdir):
    """make_video in ecosystem mode should build the global pop time series
    from all input npz files and produce a non-empty MP4."""
    traj_dir = tmpdir / "traj"
    traj_dir.mkdir()

    # Two short fake episodes with alive masks.
    for ep, T in enumerate([6, 8], start=1):
        A = 4
        positions = np.random.randn(T, A, 2).astype(np.float32)
        alive = np.ones((T, A), dtype=bool)
        # Kill a prey halfway through.
        alive[T // 2:, -1] = False
        names = np.array(["adversary_0", "adversary_1", "agent_0", "agent_1"])
        np.savez_compressed(
            str(traj_dir / f"ep_{ep}.npz"),
            pos=positions, alive=alive, agent_names=names,
            obs=positions, act=np.zeros((T, A), dtype=np.int64),
        )

    out_path = str(tmpdir / "out.mp4")
    n, used = make_video(
        str(traj_dir), out_path, fps=10, mode="ecosystem",
        dpi=80, frameskip=2, trail=3,
    )
    assert n == 2
    assert os.path.getsize(out_path) > 0
