#!/usr/bin/env python3
"""Deterministic evaluation of trained predator/prey policies.

Loads checkpoints saved by run_train.py and runs N episodes in the same env,
reporting per-team return statistics (mean, std, min, max) and capture proxy
(sum of negative-distance reward to nearest prey).

Usage:
    python tools/eval.py \
        --pred artifacts/run_*/checkpoints/pred_100.pt \
        --prey artifacts/run_*/checkpoints/prey_100.pt \
        --episodes 20

Either checkpoint can be omitted to use an untrained random-init policy as a baseline.
"""
from __future__ import annotations
import argparse
import os, sys, glob
from typing import Optional

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.ppo import PPO, PPOConfig, ActorCritic, split_teams, set_seed
from run_train import make_env, _reset, _step, _stack_team, _team_obs_dim, _act_team


def _expand_glob(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    matches = sorted(glob.glob(path))
    if not matches:
        raise FileNotFoundError(f"No match for {path}")
    return matches[-1]


def _load_actor(ckpt_path: Optional[str], obs_dim: int, act_dim: int, hidden: int, device: str) -> ActorCritic:
    ac = ActorCritic(obs_dim, act_dim, hidden).to(device)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ac.load_state_dict(ckpt["ac_state_dict"])
    ac.eval()
    for p in ac.parameters():
        p.requires_grad_(False)
    return ac


def evaluate(pred_ckpt: Optional[str], prey_ckpt: Optional[str], episodes: int,
             config_path: str, device: str, seed: int, deterministic: bool):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg.get("env", {})
    set_seed(seed)

    env = make_env(
        n_predators=int(env_cfg.get("n_predators", 2)),
        n_prey=int(env_cfg.get("n_prey", 2)),
        max_cycles=int(env_cfg.get("max_steps", 200)),
        seed=seed,
    )
    obs0 = _reset(env, seed=seed)
    agents = sorted(obs0.keys())
    pred_agents, prey_agents = split_teams(agents)
    pred_obs_dim = _team_obs_dim(pred_agents, obs0)
    prey_obs_dim = _team_obs_dim(prey_agents, obs0)
    act_dim = env.action_space(agents[0]).n
    hidden = int(cfg.get("train", {}).get("hidden", 128))

    pred_ac = _load_actor(pred_ckpt, pred_obs_dim, act_dim, hidden, device)
    prey_ac = _load_actor(prey_ckpt, prey_obs_dim, act_dim, hidden, device)

    pred_returns, prey_returns, ep_lengths = [], [], []
    for ep in range(episodes):
        obs = _reset(env, seed=seed + 1000 + ep)
        ep_pred_ret = ep_prey_ret = 0.0
        done_any = False
        t = 0
        while not done_any:
            pred_obs = _stack_team(pred_agents, obs, pred_obs_dim)
            prey_obs = _stack_team(prey_agents, obs, prey_obs_dim)
            pa, _, _ = _act_team(pred_ac, pred_obs, device, deterministic=deterministic)
            ya, _, _ = _act_team(prey_ac, prey_obs, device, deterministic=deterministic)
            acts = {}
            for i, a in enumerate(pred_agents): acts[a] = int(pa[i])
            for i, a in enumerate(prey_agents): acts[a] = int(ya[i])
            obs, rewards, done_any, _ = _step(env, acts)
            ep_pred_ret += sum(rewards[a] for a in pred_agents)
            ep_prey_ret += sum(rewards[a] for a in prey_agents)
            t += 1
        pred_returns.append(ep_pred_ret / len(pred_agents))
        prey_returns.append(ep_prey_ret / len(prey_agents))
        ep_lengths.append(t)

    def summary(name, xs):
        xs = np.asarray(xs, dtype=np.float64)
        return f"{name}: mean={xs.mean():+.3f}  std={xs.std():.3f}  min={xs.min():+.3f}  max={xs.max():+.3f}"

    print(f"\n=== Eval over {episodes} episode(s), deterministic={deterministic} ===")
    print(f"predator ckpt: {pred_ckpt or '(random init)'}")
    print(f"prey ckpt:     {prey_ckpt or '(random init)'}")
    print(summary("predator return", pred_returns))
    print(summary("prey return    ", prey_returns))
    print(summary("episode length ", ep_lengths))
    return {
        "pred_returns": pred_returns,
        "prey_returns": prey_returns,
        "ep_lengths": ep_lengths,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=None, help="Predator checkpoint .pt (glob ok)")
    ap.add_argument("--prey", default=None, help="Prey checkpoint .pt (glob ok)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stochastic", action="store_true", help="Sample actions instead of argmax")
    args = ap.parse_args()

    evaluate(
        pred_ckpt=_expand_glob(args.pred),
        prey_ckpt=_expand_glob(args.prey),
        episodes=args.episodes,
        config_path=args.config,
        device=args.device,
        seed=args.seed,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()
