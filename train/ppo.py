
import os, json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def set_seed(seed: int):
    import random, numpy as _np, torch as _torch
    random.seed(seed); _np.random.seed(seed); _torch.manual_seed(seed)

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.actor = MLP(obs_dim, act_dim, hidden)
        self.critic = MLP(obs_dim, 1, hidden)
    def step(self, obs: torch.Tensor):
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        v = self.critic(obs).squeeze(-1)
        return a, logp, v

@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    update_epochs: int = 4
    batch_size: int = 2048
    hidden: int = 128

class PPO:
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.ac = ActorCritic(obs_dim, act_dim, cfg.hidden).to(device)
        self.opt = optim.Adam(self.ac.parameters(), lr=cfg.lr)

    def save(self, path, episode, returns):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "episode": episode,
            "returns": returns,
            "ac_state_dict": self.ac.state_dict(),
            "opt_state_dict": self.opt.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ckpt["ac_state_dict"])
        self.opt.load_state_dict(ckpt["opt_state_dict"])
        return ckpt["episode"], ckpt["returns"]

    def _compute_returns(self, rews, dones, values, gamma):
        n = len(rews); out = [0.0]*n; G = 0.0
        for i in range(n-1, -1, -1):
            G = float(rews[i]) + gamma * G * (1.0 - float(dones[i]))
            out[i] = G
        return torch.tensor(out, dtype=torch.float32).to(self.device)

    def update(self, obs, acts, logps, rews, dones, vals):
        n = len(rews)
        assert n>0 and len(acts)==n and len(logps)==n and len(vals)==n and len(obs)==n and len(dones)==n, \
            f"Buffer mismatch: {len(obs)=} {len(acts)=} {len(logps)=} {len(rews)=} {len(dones)=} {len(vals)=}"
        cfg = self.cfg
        obs = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device)
        acts = torch.tensor(np.array(acts), dtype=torch.int64).to(self.device)
        old_logps = torch.tensor(np.array(logps), dtype=torch.float32).to(self.device)
        vals = torch.tensor(np.array(vals), dtype=torch.float32).to(self.device)
        rets = self._compute_returns(rews, dones, vals, cfg.gamma)
        adv = (rets - vals); adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idx = np.arange(n)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, 1024):
                b = idx[start:start+1024]
                o,a,ol,ad,rt = obs[b], acts[b], old_logps[b], adv[b], rets[b]
                logits = self.ac.actor(o)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(a)
                ratio = (logp - ol).exp()
                clip_adv = torch.clamp(ratio, 1-cfg.clip_coef, 1+cfg.clip_coef) * ad
                pg_loss = -(torch.min(ratio*ad, clip_adv)).mean()
                v = self.ac.critic(o).squeeze(-1)
                v_loss = 0.5 * (rt - v).pow(2).mean() * cfg.vf_coef
                ent = dist.entropy().mean() * cfg.ent_coef
                loss = pg_loss + v_loss - ent
                self.opt.zero_grad(); loss.backward(); self.opt.step()

class TrajectoryRecorder:
    def __init__(self, save_dir: str, env_id: str, env_cfg: dict, global_seed: int, agent_names: list, rec_cfg: Dict = {}):
        self.run_dir = save_dir
        self.sample_rate = int(rec_cfg.get("sample_rate", 1))
        self.pos_xy_idx = rec_cfg.get("pos_xy_idx", None)
        if self.pos_xy_idx:
            assert len(self.pos_xy_idx) == 2, "pos_xy_idx must be a list of two integers"
        os.makedirs(self.run_dir, exist_ok=True)
        self.step_obs: list = []
        self.step_acts: list = []
        self.step_pos: list = []
        self.buffer: list = []

        self.env_id = env_id
        self.env_cfg = env_cfg
        self.global_seed = global_seed
        self.agent_names = agent_names
        self.episode_seeds = []

    def record_step(self, t, agent_id, obs, act, rew, done, info):
        # For JSONL
        self.buffer.append({
            "t": t, "agent_id": agent_id, "obs": obs.tolist(), "act": act,
            "rew": rew, "done": done, "info": info,
        })
        # For NPZ
        self.step_obs.append(obs)
        self.step_acts.append(act)
        if self.pos_xy_idx:
            start, end = self.pos_xy_idx
            self.step_pos.append(obs[start:end])

    def _pad_and_stack(self, obs_list):
        import numpy as np
        max_len = max(int(np.size(o)) for o in obs_list)
        padded = []
        for o in obs_list:
            a = np.asarray(o, dtype=np.float32).ravel()
            if a.size < max_len:
                a = np.concatenate([a, np.zeros(max_len - a.size, dtype=np.float32)], axis=0)
            padded.append(a)
        return np.stack(padded, axis=0).astype(np.float32)

    def save(self, episode_idx: int, episode_seed: int):
        if not self.buffer: return
        self.episode_seeds.append(episode_seed)

        path_base = os.path.join(self.run_dir, f"ep_{episode_idx}")

        # JSONL
        with open(f"{path_base}.jsonl", "w") as f:
            for item in self.buffer: f.write(json.dumps(item) + "\n")

        # NPZ
        obs_mat = self._pad_and_stack(self.step_obs)

        # Reshape actions to (T, A)
        n_agents = len(self.agent_names)
        act_mat = np.array(self.step_acts, dtype=np.int64).reshape(-1, n_agents)
        T = act_mat.shape[0]

        save_payload = {"obs": obs_mat, "act": act_mat, "agent_names": self.agent_names}
        if self.pos_xy_idx:
            pos_mat = np.array(self.step_pos, dtype=np.float32).reshape(T, n_agents, -1)
            assert pos_mat.shape[2] == self.pos_xy_idx[1] - self.pos_xy_idx[0]
            save_payload["pos"] = pos_mat

        np.savez_compressed(f"{path_base}.npz", **save_payload)

        # Write manifest.jsonl
        manifest_path = os.path.join(self.run_dir, "manifest.jsonl")
        with open(manifest_path, "a") as f:
            f.write(json.dumps({
                "episode": episode_idx,
                "n_steps": T,
                "agent_names": self.agent_names,
            }) + "\n")

        # Clear buffers
        self.buffer.clear()
        self.step_obs.clear()
        self.step_acts.clear()
        self.step_pos.clear()

    def save_manifest(self):
        manifest = {
            "env_id": self.env_id,
            "env_cfg": self.env_cfg,
            "global_seed": self.global_seed,
            "episode_seeds": self.episode_seeds,
            "agent_names": self.agent_names,
        }
        with open(os.path.join(self.run_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

def flatten_obs(obs_in: Union[Dict[str, np.ndarray], tuple]) -> Tuple[np.ndarray, List[str]]:
    o = obs_in
    while isinstance(o, tuple):
        if len(o)==0: raise ValueError("Empty tuple observations")
        o = o[0]
    if not isinstance(o, dict):
        raise TypeError(f"Expected dict, got {type(o)}")
    agents = sorted(o.keys())
    vecs = [np.asarray(o[a]).ravel() for a in agents]
    m = max(v.size for v in vecs)
    padded = [np.pad(v, (0, m - v.size)) for v in vecs]
    return np.stack(padded, axis=0), agents
