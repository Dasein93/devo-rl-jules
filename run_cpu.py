import argparse
import datetime
import os
import random
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import yaml
from pettingzoo.mpe import simple_tag_v3
from tqdm import tqdm

from train.ppo import ActorCritic, set_seed, flatten_obs


def _reset(env):
    """Wrapper for env.reset() to handle different return formats."""
    ret = env.reset()
    if isinstance(ret, tuple) and len(ret) == 2:
        return ret[0]  # Return obs dict
    return ret


def _step(env, actions):
    """
    Wrapper for env.step() to handle 4-tuple or 5-tuple returns.
    Returns (obs, rewards, terminations, truncations, infos)
    """
    ret = env.step(actions)
    if len(ret) == 5:
        return ret
    elif len(ret) == 4:
        obs, rewards, dones, infos = ret
        # Assume dones are terminations, and no truncations
        terminations = dones
        truncations = {agent: False for agent in dones}
        return obs, rewards, terminations, truncations, infos
    else:
        raise ValueError(f"Unexpected number of return values from env.step(): {len(ret)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to the config file.")
    parser.add_argument("--episodes", type=int, default=None, help="Override total episodes from config.")
    parser.add_argument("--save_dir", type=str, default=None, help="Override save directory from config.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override config with CLI args if provided
    if args.episodes:
        config["train"]["total_episodes"] = args.episodes
    if args.save_dir:
        config["logging"]["save_dir"] = args.save_dir

    # Create save directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    save_path = Path(config["logging"]["save_dir"]) / run_name
    plots_path = save_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    # Set seed
    set_seed(config["seed"])

    # Environment setup
    env = simple_tag_v3.parallel_env(
        num_good=config["env"]["n_prey"],
        num_adversaries=config["env"]["n_predators"],
        num_obstacles=2,
        max_cycles=config["env"]["max_steps"],
        continuous_actions=False,
    )

    # Get observation and action space sizes
    # We assume all agents share the same observation and action space
    temp_obs, _ = env.reset()
    flat_obs_sample, agent_ids_sample = flatten_obs(temp_obs)
    obs_size = flat_obs_sample.shape[1]
    # The action space is discrete, so we need the number of possible actions
    num_actions = env.action_space(agent_ids_sample[0]).n
    possible_agents = env.possible_agents

    # Agent setup
    agent = ActorCritic(num_inputs=obs_size, num_actions=num_actions)
    optimizer = optim.Adam(agent.parameters(), lr=config["train"]["lr"], eps=1e-5)

    # --- PPO Buffer Setup ---
    # We collect transitions per agent-step, not per environment-step.
    # This handles variable numbers of agents gracefully.
    buffer_size = config["train"]["batch_size"]
    b_obs = np.zeros((buffer_size, obs_size))
    b_actions = np.zeros(buffer_size)
    b_logprobs = np.zeros(buffer_size)
    b_rewards = np.zeros(buffer_size)
    b_dones = np.zeros(buffer_size)
    b_values = np.zeros(buffer_size)

    # --- Training Loop ---
    global_step = 0
    start_time = time.time()
    all_returns = []

    pbar = tqdm(range(config["train"]["total_episodes"]), desc="Training Episodes")
    for episode in pbar:
        episode_rewards = []
        obs_dict = _reset(env)

        for step in range(config["env"]["max_steps"]):
            # Get flat observation and agent IDs for active agents
            flat_obs, agent_ids = flatten_obs(obs_dict)
            if len(agent_ids) == 0:
                break

            # Sample action from policy
            with torch.no_grad():
                actions, logprobs, _, values = agent.get_action_and_value(torch.Tensor(flat_obs))

            # Execute action in environment
            action_dict = {agent_id: act.item() for agent_id, act in zip(agent_ids, actions)}
            next_obs_dict, rewards_dict, terminations_dict, truncations_dict, _ = _step(env, action_dict)

            # Store transition data for each agent
            for i, agent_id in enumerate(agent_ids):
                if global_step < buffer_size:
                    b_obs[global_step] = flat_obs[i]
                    b_actions[global_step] = actions[i].item()
                    b_logprobs[global_step] = logprobs[i].item()
                    b_rewards[global_step] = rewards_dict[agent_id]
                    b_dones[global_step] = terminations_dict[agent_id] or truncations_dict[agent_id]
                    b_values[global_step] = values[i].item()
                    global_step += 1

            episode_rewards.append(sum(rewards_dict.values()))
            obs_dict = next_obs_dict

            # Check if episode is done for all agents
            if not env.agents:
                break

        all_returns.append(sum(episode_rewards))

        # PPO Update
        if global_step >= buffer_size:
            with torch.no_grad():
                next_value = agent.get_value(torch.Tensor(b_obs[-1])).reshape(1, -1).item()
                advantages = np.zeros(buffer_size)
                lastgaelam = 0
                for t in reversed(range(buffer_size)):
                    if t == buffer_size - 1:
                        nextnonterminal = 1.0 - b_dones[t]
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - b_dones[t+1]
                        nextvalues = b_values[t+1]
                    delta = b_rewards[t] + config["train"]["gamma"] * nextvalues * nextnonterminal - b_values[t]
                    advantages[t] = lastgaelam = delta + config["train"]["gamma"] * 0.95 * nextnonterminal * lastgaelam

            b_returns = advantages + b_values

            # Convert buffers to torch tensors
            b_obs_torch = torch.Tensor(b_obs)
            b_actions_torch = torch.Tensor(b_actions).long()
            b_logprobs_torch = torch.Tensor(b_logprobs)
            b_advantages_torch = torch.Tensor(advantages)
            b_returns_torch = torch.Tensor(b_returns)

            b_inds = np.arange(buffer_size)
            for epoch in range(config["train"]["update_epochs"]):
                np.random.shuffle(b_inds)
                minibatch_size = buffer_size // 4
                for start in range(0, buffer_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs_torch[mb_inds], b_actions_torch[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs_torch[mb_inds]
                    ratio = logratio.exp()

                    mb_advantages = b_advantages_torch[mb_inds]

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config["train"]["clip_coef"], 1 + config["train"]["clip_coef"])
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    v_loss = 0.5 * ((newvalue - b_returns_torch[mb_inds]) ** 2).mean()

                    # Entropy loss
                    entropy_loss = entropy.mean()

                    loss = pg_loss - config["train"]["ent_coef"] * entropy_loss + v_loss * config["train"]["vf_coef"]

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optimizer.step()

            # Reset buffer pointer
            global_step = 0

        # Logging
        if (episode + 1) % config["logging"]["plot_every"] == 0:
            pbar.set_description(f"Episode {episode+1} | Avg Return: {np.mean(all_returns[-50:]):.2f}")

            # Save metrics
            df = pd.DataFrame({"episode": range(len(all_returns)), "return": all_returns})
            df.to_csv(save_path / "metrics.csv", index=False)

            # Save plot
            plt.style.use('ggplot')
            plt.figure(figsize=(10, 6))
            plt.plot(df['episode'], df['return'], label='Episodic Return')
            plt.xlabel("Episode")
            plt.ylabel("Return")
            plt.title("Learning Curve")
            plt.legend()
            plt.savefig(plots_path / "return.png")
            plt.close()

    env.close()
    print(f"Training finished. Artifacts saved in {save_path}")
