import argparse
import json
import os
import numpy as np
import imageio
from pettingzoo.mpe import simple_tag_v3

def make_env(env_id, n_predators=2, n_prey=2, max_cycles=200, seed=42):
    if env_id == "mpe.simple_tag_v3":
        env = simple_tag_v3.parallel_env(
            num_adversaries=n_predators,
            num_good=n_prey,
            num_obstacles=0,
            max_cycles=max_cycles,
            continuous_actions=False,
            render_mode="rgb_array",
        )
        env.reset(seed=seed)
        return env
    else:
        raise ValueError(f"Unknown env_id: {env_id}")

def replay_trajectory(trajectory_path, env_id):
    npz_path = trajectory_path.replace(".jsonl", ".npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found for trajectory: {trajectory_path}")

    with open(trajectory_path, "r") as f:
        jsonl_data = [json.loads(line) for line in f]

    npz_data = np.load(npz_path)
    obs_data = npz_data["obs"]
    act_data = npz_data["act"]

    # Assuming the env config is stored somewhere or passed as args
    # For now, using defaults from run_train.py
    env = make_env(env_id)

    frames = []
    obs = env.reset()

    num_agents = len(env.agents)
    for i in range(0, len(jsonl_data), num_agents):
        actions = {agent: act_data[i+j] for j, agent in enumerate(env.agents)}

        # The stored observations are from the *previous* step.
        # We don't strictly need them for replay if we have actions.

        env.step(actions)
        frames.append(env.render())

    env.close()

    # Save the video
    replay_dir = os.path.dirname(trajectory_path)
    video_path = os.path.join(replay_dir, os.path.basename(trajectory_path).replace(".jsonl", ".mp4"))
    imageio.mimsave(video_path, frames, fps=10)
    print(f"Saved replay to {video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_path", type=str, help="Path to the .jsonl trajectory file")
    parser.add_argument("--env_id", type=str, default="mpe.simple_tag_v3", help="Environment ID")
    args = parser.parse_args()

    replay_trajectory(args.trajectory_path, args.env_id)
