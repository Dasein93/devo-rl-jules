#!/usr/bin/env python3
"""Round-robin tournament between league snapshots.

For each (predator_snapshot, prey_snapshot) pair in a run's league directory,
play K episodes and score them. Output:
  - tournament/scores.csv: pair-wise mean predator captures (and prey
    survival time) over K episodes.
  - tournament/heatmap.png: capture-count matrix.
  - tournament/ratings.csv: Elo-style rating per snapshot per team,
    inferred from win/loss outcomes (predator "wins" if captures > 0).

Usage:
    python tools/tournament.py \\
        --run artifacts/run_20260519_024431 \\
        --episodes 3 \\
        --out artifacts/run_20260519_024431/tournament
"""
from __future__ import annotations
import argparse
import os, sys, glob, csv, re
from typing import List, Tuple

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.ppo import ActorCritic, split_teams, set_seed
from run_train import make_env, _reset, _step, _stack_team, _team_obs_dim, _act_team_actor_only


SNAP_RE = re.compile(r"snap_(\d+)\.pt$")


def _list_snapshots(league_dir: str, team: str) -> List[str]:
    paths = glob.glob(os.path.join(league_dir, team, "snap_*.pt"))
    return sorted(paths, key=lambda p: int(SNAP_RE.search(p).group(1)))


def _load_actor(path: str, device: str) -> ActorCritic:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ac = ActorCritic(
        obs_dim=ckpt["obs_dim"],
        act_dim=ckpt["act_dim"],
        hidden=ckpt["hidden"],
        state_dim=ckpt.get("state_dim"),
    ).to(device)
    ac.load_state_dict(ckpt["ac_state_dict"])
    ac.eval()
    for p in ac.parameters():
        p.requires_grad_(False)
    return ac


def _short_name(path: str) -> str:
    base = os.path.basename(path)
    m = SNAP_RE.search(base)
    return f"ep{int(m.group(1))}" if m else base


def _play(env, pred_ac, prey_ac, pred_agents, prey_agents, pred_obs_dim, prey_obs_dim,
          episodes: int, max_steps_per_ep: int, device: str, seed: int, deterministic: bool):
    """Run K episodes; return (mean_captures, mean_ep_steps, mean_pred_return, mean_prey_return)."""
    captures, lengths, pred_rets, prey_rets = [], [], [], []
    for ep in range(episodes):
        obs = _reset(env, seed=seed + ep)
        ep_caps = 0
        ep_pred_ret = 0.0
        ep_prey_ret = 0.0
        done_any = False
        t = 0
        while not done_any:
            pred_obs = _stack_team(pred_agents, obs, pred_obs_dim)
            prey_obs = _stack_team(prey_agents, obs, prey_obs_dim)
            pa, _ = _act_team_actor_only(pred_ac, pred_obs, device, deterministic=deterministic)
            ya, _ = _act_team_actor_only(prey_ac, prey_obs, device, deterministic=deterministic)
            acts = {a: int(pa[i]) for i, a in enumerate(pred_agents)}
            acts.update({a: int(ya[i]) for i, a in enumerate(prey_agents)})
            obs, rewards, done_any, _ = _step(env, acts)
            ep_caps += sum(1 for a in prey_agents if rewards[a] <= -10.0 + 1e-6)
            ep_pred_ret += sum(rewards[a] for a in pred_agents)
            ep_prey_ret += sum(rewards[a] for a in prey_agents)
            t += 1
            if t >= max_steps_per_ep:
                break
        captures.append(ep_caps)
        lengths.append(t)
        pred_rets.append(ep_pred_ret / max(1, len(pred_agents)))
        prey_rets.append(ep_prey_ret / max(1, len(prey_agents)))
    return float(np.mean(captures)), float(np.mean(lengths)), float(np.mean(pred_rets)), float(np.mean(prey_rets))


def _elo(matrix: np.ndarray, k: float = 32.0, iters: int = 5):
    """Compute Elo for both sides of an asymmetric round-robin.
    matrix[i,j] = predator i's mean captures vs prey j; >0 → predator win.
    Returns (pred_ratings, prey_ratings) each starting at 1500."""
    P, Q = matrix.shape
    pred = np.full(P, 1500.0)
    prey = np.full(Q, 1500.0)
    for _ in range(iters):
        for i in range(P):
            for j in range(Q):
                score = 1.0 if matrix[i, j] > 0 else (0.5 if matrix[i, j] == 0 else 0.0)
                exp = 1.0 / (1.0 + 10.0 ** ((prey[j] - pred[i]) / 400.0))
                pred[i] += k * (score - exp)
                prey[j] += k * ((1.0 - score) - (1.0 - exp))
    return pred, prey


def run_tournament(run_dir: str, episodes: int, out_dir: str, config_path: str,
                   device: str, seed: int, deterministic: bool):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg.get("env", {})
    set_seed(seed)

    league_dir = os.path.join(run_dir, "league")
    pred_snaps = _list_snapshots(league_dir, "predator")
    prey_snaps = _list_snapshots(league_dir, "prey")
    if not pred_snaps or not prey_snaps:
        raise FileNotFoundError(f"League is empty under {league_dir}")

    os.makedirs(out_dir, exist_ok=True)

    env = make_env(
        env_id=env_cfg.get("id", "mpe.simple_tag_v3"),
        n_predators=int(env_cfg.get("n_predators", 2)),
        n_prey=int(env_cfg.get("n_prey", 2)),
        max_cycles=int(env_cfg.get("max_steps", 200)),
        seed=seed,
        ecosystem_overrides=dict(env_cfg.get("ecosystem", {}) or {}),
        num_obstacles=int(env_cfg.get("num_obstacles", 0)),
    )
    obs0 = _reset(env, seed=seed)
    agents = sorted(obs0.keys())
    pred_agents, prey_agents = split_teams(agents)
    pred_obs_dim = _team_obs_dim(pred_agents, obs0)
    prey_obs_dim = _team_obs_dim(prey_agents, obs0)
    max_steps = int(env_cfg.get("max_steps", 200))

    P, Q = len(pred_snaps), len(prey_snaps)
    cap_mat = np.zeros((P, Q), dtype=np.float32)
    len_mat = np.zeros((P, Q), dtype=np.float32)
    pred_ret_mat = np.zeros((P, Q), dtype=np.float32)
    prey_ret_mat = np.zeros((P, Q), dtype=np.float32)

    pred_actors = [_load_actor(p, device) for p in pred_snaps]
    prey_actors = [_load_actor(p, device) for p in prey_snaps]

    print(f"Tournament: {P} predator snaps × {Q} prey snaps × {episodes} episode(s) = {P * Q * episodes} games")
    for i, pa in enumerate(pred_actors):
        for j, ya in enumerate(prey_actors):
            cap, ln, pr, yr = _play(
                env, pa, ya, pred_agents, prey_agents, pred_obs_dim, prey_obs_dim,
                episodes, max_steps, device, seed + 1000 + i * Q + j, deterministic,
            )
            cap_mat[i, j] = cap
            len_mat[i, j] = ln
            pred_ret_mat[i, j] = pr
            prey_ret_mat[i, j] = yr
            print(f"  pred={_short_name(pred_snaps[i])} vs prey={_short_name(prey_snaps[j])}: "
                  f"captures={cap:.2f}  steps={ln:.1f}  pred_ret={pr:+.1f}  prey_ret={yr:+.1f}")

    pred_names = [_short_name(p) for p in pred_snaps]
    prey_names = [_short_name(p) for p in prey_snaps]

    scores_path = os.path.join(out_dir, "scores.csv")
    with open(scores_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pred", "prey", "mean_captures", "mean_ep_steps", "mean_pred_return", "mean_prey_return"])
        for i in range(P):
            for j in range(Q):
                w.writerow([pred_names[i], prey_names[j],
                            f"{cap_mat[i, j]:.4f}", f"{len_mat[i, j]:.2f}",
                            f"{pred_ret_mat[i, j]:.4f}", f"{prey_ret_mat[i, j]:.4f}"])

    pred_elo, prey_elo = _elo(cap_mat)
    ratings_path = os.path.join(out_dir, "ratings.csv")
    with open(ratings_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["team", "snapshot", "elo"])
        for n, r in zip(pred_names, pred_elo):
            w.writerow(["predator", n, f"{r:.1f}"])
        for n, r in zip(prey_names, prey_elo):
            w.writerow(["prey", n, f"{r:.1f}"])

    heatmap_path = os.path.join(out_dir, "heatmap.png")
    fig, ax = plt.subplots(figsize=(max(4, 0.6 * Q + 2), max(4, 0.6 * P + 2)), dpi=130)
    im = ax.imshow(cap_mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(Q)); ax.set_xticklabels(prey_names, rotation=45, ha="right")
    ax.set_yticks(range(P)); ax.set_yticklabels(pred_names)
    ax.set_xlabel("prey snapshot")
    ax.set_ylabel("predator snapshot")
    ax.set_title(f"Mean captures over {episodes} ep")
    for i in range(P):
        for j in range(Q):
            ax.text(j, i, f"{cap_mat[i, j]:.1f}", ha="center", va="center",
                    color="white" if cap_mat[i, j] < cap_mat.max() * 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(heatmap_path)
    plt.close(fig)

    print(f"\nWrote {scores_path}")
    print(f"Wrote {ratings_path}")
    print(f"Wrote {heatmap_path}")
    print(f"\nTop predator (Elo): {pred_names[int(np.argmax(pred_elo))]}  ({pred_elo.max():.1f})")
    print(f"Top prey (Elo):     {prey_names[int(np.argmax(prey_elo))]}  ({prey_elo.max():.1f})")

    return {
        "cap_mat": cap_mat, "len_mat": len_mat,
        "pred_ret_mat": pred_ret_mat, "prey_ret_mat": prey_ret_mat,
        "pred_elo": pred_elo, "prey_elo": prey_elo,
        "pred_names": pred_names, "prey_names": prey_names,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to artifacts/run_* directory")
    ap.add_argument("--episodes", type=int, default=3, help="Episodes per pair")
    ap.add_argument("--out", default=None, help="Output dir (default: <run>/tournament)")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stochastic", action="store_true", help="Sample actions instead of argmax")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.run, "tournament")
    run_tournament(
        run_dir=args.run, episodes=args.episodes, out_dir=out_dir,
        config_path=args.config, device=args.device, seed=args.seed,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()
