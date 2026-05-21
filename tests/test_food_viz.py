"""Smoke tests for food-field recording + replay underlay."""
import os

import imageio.v2 as imageio
import numpy as np

from tools.replay import _render_episode_ecosystem, _load_food, make_video
from train.ppo import TrajectoryRecorder


def test_recorder_persists_food_field_to_npz(tmpdir):
    rec = TrajectoryRecorder(
        save_dir=str(tmpdir),
        env_id="ecosystem",
        env_cfg={},
        global_seed=0,
        agent_names=["adversary_0", "agent_0"],
    )
    obs = {"adversary_0": np.zeros(8, dtype=np.float32),
           "agent_0":     np.zeros(8, dtype=np.float32)}
    acts = {"adversary_0": 0, "agent_0": 0}
    rews = {"adversary_0": 0.0, "agent_0": 0.0}
    for t in range(3):
        rec.record_step(t, obs, acts, rews, done_any=False, infos={})
        rec.record_food(np.full((4, 4), float(t), dtype=np.float32))
    rec.save(episode_idx=1, episode_seed=0)

    food = _load_food(str(tmpdir / "ep_1.npz"))
    assert food is not None
    assert food.shape == (3, 4, 4)
    assert food[0, 0, 0] == 0.0
    assert food[2, 0, 0] == 2.0


def test_render_with_food_does_not_crash(tmpdir):
    T, A = 10, 4
    rng = np.random.default_rng(0)
    positions = np.cumsum(rng.normal(scale=0.05, size=(T, A, 2)).astype(np.float32), axis=0)
    alive = np.ones((T, A), dtype=bool)
    agent_names = ["adversary_0", "adversary_1", "agent_0", "agent_1"]
    food = np.clip(rng.normal(loc=5.0, scale=2.0, size=(T, 8, 8)).astype(np.float32), 0, 10)
    pop_pred = alive[:, :2].sum(axis=1)
    pop_prey = alive[:, 2:].sum(axis=1)

    out = str(tmpdir / "food_render.mp4")
    writer = imageio.get_writer(out, fps=10, codec="libx264", bitrate="2000k", quality=8)
    try:
        _render_episode_ecosystem(
            positions, alive, agent_names, writer,
            pop_pred_global=pop_pred, pop_prey_global=pop_prey,
            cumulative_offset=0, title_prefix="test ", dpi=80, frameskip=2,
            food=food,
        )
    finally:
        writer.close()
    assert os.path.getsize(out) > 0


def test_make_video_handles_food_in_npz(tmpdir):
    traj = tmpdir / "traj"; traj.mkdir()
    T, A = 6, 2
    np.savez_compressed(
        str(traj / "ep_1.npz"),
        pos=np.zeros((T, A, 2), dtype=np.float32),
        alive=np.ones((T, A), dtype=bool),
        agent_names=np.array(["adversary_0", "agent_0"]),
        obs=np.zeros((T, A, 4), dtype=np.float32),
        act=np.zeros((T, A), dtype=np.int64),
        food=np.full((T, 8, 8), 5.0, dtype=np.float32),
    )
    out = str(tmpdir / "out.mp4")
    n, _used = make_video(str(traj), out, fps=10, mode="ecosystem", dpi=80, frameskip=2)
    assert n == 1
    assert os.path.getsize(out) > 0
