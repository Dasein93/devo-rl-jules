
import os, csv, argparse, yaml, numpy as np, glob
from datetime import datetime, timezone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train.ppo import PPO, PPOConfig, flatten_obs, set_seed, TrajectoryRecorder

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

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def _reset(env, seed=None):
    out = env.reset(seed=seed)
    return out[0] if isinstance(out, tuple) and len(out)==2 else out

def _step(env, actions):
    out = env.step(actions)
    if isinstance(out, tuple) and len(out)==5:
        next_obs, rewards, terminations, truncations, infos = out
        done_any = bool(any(terminations.values()) or any(truncations.values()))
        return next_obs, rewards, done_any, infos
    elif isinstance(out, tuple) and len(out)==4:
        next_obs, rewards, dones, infos = out
        done_any = bool(any(dones.values()))
        return next_obs, rewards, done_any, infos
    raise RuntimeError("Unexpected step() return format")

def main(cfg_path, override_eps=None, save_dir=None, device=None, resume_from=None):
    with open(cfg_path,"r") as f: cfg = yaml.safe_load(f)
    seed = int(cfg.get("seed",42)); set_seed(seed)
    device = device or cfg.get("device","cpu")
    env_cfg = cfg.get("env",{})
    max_steps = int(env_cfg.get("max_steps",200))
    n_pred = int(env_cfg.get("n_predators",2))
    n_ev = int(env_cfg.get("n_prey",2))
    total_episodes = int(override_eps or cfg.get("train",{}).get("total_episodes",300))

    if resume_from:
        out_dir = resume_from
    else:
        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M")
        out_dir = os.path.join(save_dir or cfg.get("logging",{}).get("save_dir","artifacts/"), run_id)

    plots_dir = os.path.join(out_dir, "plots"); ensure_dir(plots_dir)
    checkpoints_dir = os.path.join(out_dir, "checkpoints"); ensure_dir(checkpoints_dir)

    recorder = None
    if cfg.get("recording",{}).get("enabled",False):
        replays_dir = os.path.join(out_dir, "replays"); ensure_dir(replays_dir)
        recorder = TrajectoryRecorder(replays_dir)

    env = make_env(n_predators=n_pred, n_prey=n_ev, max_cycles=max_steps, seed=seed)
    obs0 = _reset(env, seed=seed)
    obs_arr, agents = flatten_obs(obs0)
    obs_dim = obs_arr.shape[1]; act_dim = env.action_space(agents[0]).n; n_agents = len(agents)

    ppo_cfg = PPOConfig(
        lr=float(cfg["train"].get("lr",3e-4)),
        gamma=float(cfg["train"].get("gamma",0.99)),
        clip_coef=float(cfg["train"].get("clip_coef",0.2)),
        ent_coef=float(cfg["train"].get("ent_coef",0.01)),
        vf_coef=float(cfg["train"].get("vf_coef",0.5)),
        update_epochs=int(cfg["train"].get("update_epochs",4)),
        batch_size=int(cfg["train"].get("batch_size",2048)),
        hidden=128,
    )
    ppo = PPO(obs_dim, act_dim, ppo_cfg, device=device)

    start_ep, returns = 1, []
    if cfg.get("checkpoint",{}).get("enabled",True):
        latest_ckpt = max(glob.glob(os.path.join(checkpoints_dir, "*.pt")), key=os.path.getctime, default=None)
        if latest_ckpt:
            print("Resuming from", latest_ckpt)
            start_ep, returns = ppo.load(latest_ckpt)
            start_ep += 1

    metrics_path = os.path.join(out_dir,"metrics.csv")
    if start_ep == 1:
        with open(metrics_path,"w",newline="") as f: csv.writer(f).writerow(["episode","return_mean"])

    S={"obs":[], "acts":[], "logps":[], "rews":[], "dones":[], "vals":[]}
    import torch
    for ep in range(start_ep, total_episodes+1):
        obs = _reset(env, seed=seed+ep); ep_ret=0.0; done_any=False; t=0
        while not done_any:
            obs_arr, agents = flatten_obs(obs); obs_t = np.asarray(obs_arr, dtype=np.float32)
            with torch.no_grad(): a, logp, v = ppo.ac.step(torch.from_numpy(obs_t).to(device))
            acts = {agent:int(a[i].item()) for i,agent in enumerate(agents)}
            next_obs, rewards, done_any, infos = _step(env, acts)

            if recorder and ep % cfg["recording"].get("sample_rate",1) == 0:
                for i,agent_id in enumerate(agents):
                    recorder.record_step(t, agent_id, obs[agent_id], acts[agent_id], rewards[agent_id], done_any, infos[agent_id])

            avg_r = float(np.mean(list(rewards.values())))
            S["obs"].extend(obs_t)
            S["acts"].extend([acts[a] for a in agents])
            S["logps"].extend([logp[i].item() for i in range(len(agents))])
            S["vals"].extend([v[i].item() for i in range(len(agents))])
            S["rews"].append(avg_r)      # per-step
            S["dones"].append(float(done_any))
            ep_ret += sum(rewards.values()); obs = next_obs; t+=1

        if recorder: recorder.save(ep)
        n_total = len(S["acts"]); n_steps = len(S["rews"])
        rep = max(1, n_total//max(1,n_steps))
        if n_total != n_steps:
            S["rews"]  = [r for r in S["rews"]  for _ in range(rep)]
            S["dones"] = [d for d in S["dones"] for _ in range(rep)]
            S["rews"]  = S["rews"][:n_total]; S["dones"] = S["dones"][:n_total]

        ppo.update(S["obs"], S["acts"], S["logps"], S["rews"], S["dones"], S["vals"])
        for k in S: S[k]=[]

        returns.append(ep_ret/max(1,n_agents))
        with open(metrics_path,"a",newline="") as f: csv.writer(f).writerow([ep, returns[-1]])

        plot_every = int(cfg["logging"].get("plot_every",50))
        if ep % plot_every==0 or ep==total_episodes:
            xs = np.arange(start_ep, len(returns)+start_ep); w=min(plot_every,len(returns))
            ma = np.convolve(returns, np.ones(w)/w, mode="valid") if len(returns)>=w else []
            plt.figure(); plt.plot(xs, returns, label="return")
            if len(ma)>1: plt.plot(np.arange(w,len(returns)+1)+start_ep-1, ma, label=f"MA{w}")
            plt.xlabel("episode"); plt.ylabel("avg return per-agent"); plt.legend()
            plt.tight_layout(); plt.savefig(os.path.join(plots_dir,"return.png")); plt.close()
            print(f"[{ep}/{total_episodes}] mean return (last 10): {np.mean(returns[-10:]):.3f}")

        if cfg.get("checkpoint",{}).get("enabled",True) and ep % cfg["checkpoint"].get("every",100) == 0:
            ckpt_path = os.path.join(checkpoints_dir, f"ckpt_{ep}.pt")
            ppo.save(ckpt_path, ep, returns)
            print("Saved checkpoint:", ckpt_path)

    print("Saved:", metrics_path, "and", os.path.join(plots_dir,"return.png"))

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--save_dir", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--resume_from", type=str, default=None, help="Path to run directory to resume from")
    a = ap.parse_args()
    main(a.config, a.episodes, a.save_dir, a.device, a.resume_from)
