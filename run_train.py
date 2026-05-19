"""Per-team PPO training on PettingZoo MPE simple_tag_v3.

One PPO per team (predators, prey). Each team's transitions are collected
separately; only the team's own samples update its policy. An optional League
of frozen snapshots is sampled as the opponent to stabilise co-adaptation.
"""
import os, csv, argparse, glob
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from train.ppo import PPO, PPOConfig, ActorCritic, TrajectoryRecorder, flatten_obs, set_seed, split_teams
from train.league import League


def make_env(n_predators=2, n_prey=2, max_cycles=200, seed=42):
    from pettingzoo.mpe import simple_tag_v3
    env = simple_tag_v3.parallel_env(
        num_adversaries=n_predators,
        num_good=n_prey,
        num_obstacles=0,
        max_cycles=max_cycles,
        continuous_actions=False,
        render_mode=None,
    )
    env.reset(seed=seed)
    return env


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def _reset(env, seed=None):
    out = env.reset(seed=seed)
    return out[0] if isinstance(out, tuple) and len(out) == 2 else out


def _step(env, actions):
    out = env.step(actions)
    if isinstance(out, tuple) and len(out) == 5:
        next_obs, rewards, terminations, truncations, infos = out
        done_any = bool(any(terminations.values()) or any(truncations.values()))
        return next_obs, rewards, done_any, infos
    if isinstance(out, tuple) and len(out) == 4:
        next_obs, rewards, dones, infos = out
        done_any = bool(any(dones.values()))
        return next_obs, rewards, done_any, infos
    raise RuntimeError("Unexpected step() return format")


def _team_obs_dim(agents: List[str], obs_dict: Dict[str, np.ndarray]) -> int:
    return max(int(np.size(obs_dict[a])) for a in agents)


def _pad(vec: np.ndarray, dim: int) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).ravel()
    if v.size < dim:
        return np.concatenate([v, np.zeros(dim - v.size, dtype=np.float32)])
    return v[:dim]


def _stack_team(agents: List[str], obs_dict: Dict[str, np.ndarray], dim: int) -> np.ndarray:
    return np.stack([_pad(obs_dict[a], dim) for a in agents], axis=0)


def _act_team(ac: ActorCritic, obs_arr: np.ndarray, device: str, deterministic: bool = False):
    if obs_arr.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    with torch.no_grad():
        a, logp, v = ac.step(torch.from_numpy(obs_arr).to(device), deterministic=deterministic)
    return a.cpu().numpy(), logp.cpu().numpy(), v.cpu().numpy()


def _plot_returns(plots_dir: str, pred_ret: List[float], prey_ret: List[float], window: int):
    if not pred_ret and not prey_ret:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(1, len(pred_ret) + 1)

    def smooth(series):
        if len(series) < window:
            return None
        return np.convolve(series, np.ones(window) / window, mode="valid")

    if pred_ret:
        ax.plot(xs, pred_ret, color="tab:red", alpha=0.35, label="predator")
        ma = smooth(pred_ret)
        if ma is not None:
            ax.plot(np.arange(window, len(pred_ret) + 1), ma, color="tab:red", label=f"predator MA{window}")
    if prey_ret:
        ax.plot(xs, prey_ret, color="tab:green", alpha=0.35, label="prey")
        ma = smooth(prey_ret)
        if ma is not None:
            ax.plot(np.arange(window, len(prey_ret) + 1), ma, color="tab:green", label=f"prey MA{window}")

    ax.set_xlabel("episode"); ax.set_ylabel("mean return per agent"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "return.png"))
    plt.close(fig)


def _load_latest_ckpt(ppo: PPO, ckpt_dir: str, prefix: str) -> Tuple[int, List[float]]:
    candidates = glob.glob(os.path.join(ckpt_dir, f"{prefix}_*.pt"))
    if not candidates:
        return 0, []
    latest = max(candidates, key=os.path.getctime)
    print(f"Resuming {prefix} from {latest}")
    return ppo.load(latest)


def main(cfg_path, override_eps=None, save_dir=None, device=None, resume_from=None):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = device or cfg.get("device", "cpu")

    env_cfg = cfg.get("env", {})
    max_steps = int(env_cfg.get("max_steps", 200))
    n_pred = int(env_cfg.get("n_predators", 2))
    n_prey = int(env_cfg.get("n_prey", 2))
    total_episodes = int(override_eps or cfg.get("train", {}).get("total_episodes", 300))

    if resume_from:
        out_dir = resume_from
    else:
        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
        out_dir = os.path.join(save_dir or cfg.get("logging", {}).get("save_dir", "artifacts/"), run_id)

    plots_dir = os.path.join(out_dir, "plots"); ensure_dir(plots_dir)
    ckpt_dir = os.path.join(out_dir, "checkpoints"); ensure_dir(ckpt_dir)
    league_dir = os.path.join(out_dir, "league"); ensure_dir(league_dir)

    env = make_env(n_predators=n_pred, n_prey=n_prey, max_cycles=max_steps, seed=seed)
    obs0 = _reset(env, seed=seed)
    agents = sorted(obs0.keys())
    pred_agents, prey_agents = split_teams(agents)
    if not pred_agents or not prey_agents:
        raise RuntimeError(f"Need at least one predator and one prey; got pred={pred_agents}, prey={prey_agents}")

    pred_obs_dim = _team_obs_dim(pred_agents, obs0)
    prey_obs_dim = _team_obs_dim(prey_agents, obs0)
    act_dim = env.action_space(agents[0]).n

    tcfg = cfg.get("train", {})
    ppo_cfg = PPOConfig(
        lr=float(tcfg.get("lr", 3e-4)),
        gamma=float(tcfg.get("gamma", 0.99)),
        gae_lambda=float(tcfg.get("gae_lambda", 0.95)),
        clip_coef=float(tcfg.get("clip_coef", 0.2)),
        ent_coef=float(tcfg.get("ent_coef", 0.01)),
        vf_coef=float(tcfg.get("vf_coef", 0.5)),
        update_epochs=int(tcfg.get("update_epochs", 4)),
        minibatch_size=int(tcfg.get("minibatch_size", tcfg.get("batch_size", 1024))),
        hidden=int(tcfg.get("hidden", 128)),
        max_grad_norm=float(tcfg.get("max_grad_norm", 0.5)),
    )
    ppo_pred = PPO(pred_obs_dim, act_dim, ppo_cfg, device=device)
    ppo_prey = PPO(prey_obs_dim, act_dim, ppo_cfg, device=device)

    lcfg = cfg.get("league", {})
    league_enabled = bool(lcfg.get("enabled", True))
    league_pred = League(os.path.join(league_dir, "predator"), team="predator",
                         snapshot_every=int(lcfg.get("snapshot_every", 25)),
                         opponent_prob=float(lcfg.get("opponent_prob", 0.3)) if league_enabled else 0.0,
                         max_snapshots=int(lcfg.get("max_snapshots", 20)),
                         device=device, rng=np.random.default_rng(seed))
    league_prey = League(os.path.join(league_dir, "prey"), team="prey",
                         snapshot_every=int(lcfg.get("snapshot_every", 25)),
                         opponent_prob=float(lcfg.get("opponent_prob", 0.3)) if league_enabled else 0.0,
                         max_snapshots=int(lcfg.get("max_snapshots", 20)),
                         device=device, rng=np.random.default_rng(seed + 1))

    recorder = None
    rec_cfg = cfg.get("recording", {})
    if rec_cfg.get("enabled", False):
        traj_dir = os.path.join(out_dir, "traj"); ensure_dir(traj_dir)
        env_record_cfg = {
            "num_adversaries": n_pred, "num_good": n_prey,
            "max_cycles": max_steps, "continuous_actions": False,
        }
        recorder = TrajectoryRecorder(
            save_dir=traj_dir,
            env_id=env_cfg.get("id", "mpe.simple_tag_v3"),
            env_cfg=env_record_cfg,
            global_seed=seed,
            agent_names=agents,
            rec_cfg=rec_cfg,
        )

    pred_returns: List[float] = []
    prey_returns: List[float] = []
    start_ep = 1
    if cfg.get("checkpoint", {}).get("enabled", True):
        pe, pr = _load_latest_ckpt(ppo_pred, ckpt_dir, "pred")
        ye, yr = _load_latest_ckpt(ppo_prey, ckpt_dir, "prey")
        if pe or ye:
            start_ep = max(pe, ye) + 1
            pred_returns = list(pr)
            prey_returns = list(yr)

    metrics_path = os.path.join(out_dir, "metrics.csv")
    if start_ep == 1:
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "episode", "pred_return", "prey_return",
                "pred_pg_loss", "pred_v_loss", "pred_entropy",
                "prey_pg_loss", "prey_v_loss", "prey_entropy",
                "pred_league", "prey_league",
            ])

    for ep in range(start_ep, total_episodes + 1):
        obs = _reset(env, seed=seed + ep)

        opp_pred = league_pred.sample()
        opp_prey = league_prey.sample()
        pred_actor = opp_pred if opp_pred is not None else ppo_pred.ac
        prey_actor = opp_prey if opp_prey is not None else ppo_prey.ac
        train_pred = opp_pred is None
        train_prey = opp_prey is None

        pred_buf = {"obs": [], "acts": [], "logps": [], "vals": [], "rews": [], "dones": []}
        prey_buf = {"obs": [], "acts": [], "logps": [], "vals": [], "rews": [], "dones": []}
        ep_pred_ret = 0.0
        ep_prey_ret = 0.0
        done_any = False
        t = 0

        while not done_any:
            pred_obs = _stack_team(pred_agents, obs, pred_obs_dim)
            prey_obs = _stack_team(prey_agents, obs, prey_obs_dim)

            pa, plogp, pv = _act_team(pred_actor, pred_obs, device)
            ya, ylogp, yv = _act_team(prey_actor, prey_obs, device)

            acts = {}
            for i, a in enumerate(pred_agents): acts[a] = int(pa[i])
            for i, a in enumerate(prey_agents): acts[a] = int(ya[i])

            next_obs, rewards, done_any, infos = _step(env, acts)

            pred_step_rew = float(np.mean([rewards[a] for a in pred_agents]))
            prey_step_rew = float(np.mean([rewards[a] for a in prey_agents]))
            ep_pred_ret += sum(rewards[a] for a in pred_agents)
            ep_prey_ret += sum(rewards[a] for a in prey_agents)

            if train_pred:
                pred_buf["obs"].extend(pred_obs)
                pred_buf["acts"].extend(pa.tolist())
                pred_buf["logps"].extend(plogp.tolist())
                pred_buf["vals"].extend(pv.tolist())
                # per-agent rew/done, replicated across team size
                pred_buf["rews"].extend([pred_step_rew] * len(pred_agents))
                pred_buf["dones"].extend([float(done_any)] * len(pred_agents))
            if train_prey:
                prey_buf["obs"].extend(prey_obs)
                prey_buf["acts"].extend(ya.tolist())
                prey_buf["logps"].extend(ylogp.tolist())
                prey_buf["vals"].extend(yv.tolist())
                prey_buf["rews"].extend([prey_step_rew] * len(prey_agents))
                prey_buf["dones"].extend([float(done_any)] * len(prey_agents))

            if recorder and t % recorder.sample_rate == 0:
                recorder.record_step(t, obs, acts, rewards, done_any, infos)

            obs = next_obs
            t += 1

        if recorder:
            recorder.save(ep, episode_seed=(seed + ep))

        pred_stats = ppo_pred.update(**pred_buf) if train_pred else {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "samples": 0}
        prey_stats = ppo_prey.update(**prey_buf) if train_prey else {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "samples": 0}

        pred_returns.append(ep_pred_ret / max(1, len(pred_agents)))
        prey_returns.append(ep_prey_ret / max(1, len(prey_agents)))

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([
                ep,
                f"{pred_returns[-1]:.6f}", f"{prey_returns[-1]:.6f}",
                f"{pred_stats['pg_loss']:.6f}", f"{pred_stats['v_loss']:.6f}", f"{pred_stats['entropy']:.6f}",
                f"{prey_stats['pg_loss']:.6f}", f"{prey_stats['v_loss']:.6f}", f"{prey_stats['entropy']:.6f}",
                len(league_pred), len(league_prey),
            ])

        plot_every = int(cfg.get("logging", {}).get("plot_every", 50))
        if ep % plot_every == 0 or ep == total_episodes:
            _plot_returns(plots_dir, pred_returns, prey_returns, window=min(plot_every, len(pred_returns)))
            tail = 10
            print(f"[{ep}/{total_episodes}] "
                  f"pred(last {tail}) {np.mean(pred_returns[-tail:]):+.3f}  "
                  f"prey(last {tail}) {np.mean(prey_returns[-tail:]):+.3f}  "
                  f"league pred={len(league_pred)} prey={len(league_prey)}")

        if cfg.get("checkpoint", {}).get("enabled", True) and ep % int(cfg.get("checkpoint", {}).get("every", 100)) == 0:
            ppo_pred.save(os.path.join(ckpt_dir, f"pred_{ep}.pt"), ep, pred_returns)
            ppo_prey.save(os.path.join(ckpt_dir, f"prey_{ep}.pt"), ep, prey_returns)
            print(f"Saved checkpoint at episode {ep}")

        if league_enabled:
            league_pred.maybe_snapshot(ppo_pred.ac, ep)
            league_prey.maybe_snapshot(ppo_prey.ac, ep)

    if recorder:
        recorder.save_manifest()
        print("Saved manifest:", os.path.join(recorder.run_dir, "manifest.json"))

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved plot:    {os.path.join(plots_dir, 'return.png')}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--save_dir", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--resume_from", type=str, default=None, help="Path to run directory to resume from")
    a = ap.parse_args()
    main(a.config, a.episodes, a.save_dir, a.device, a.resume_from)
