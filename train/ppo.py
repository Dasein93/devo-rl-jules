"""PPO with GAE, separate-policy support, and trajectory recording.

`PPO` is a single shared-policy learner over a fixed obs/action shape.
`run_train.py` instantiates one PPO per team (predators, prey).
"""
import os, json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Optional
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
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden
        self.actor = MLP(obs_dim, act_dim, hidden)
        self.critic = MLP(obs_dim, 1, hidden)

    def step(self, obs: torch.Tensor, deterministic: bool = False):
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = logits.argmax(-1) if deterministic else dist.sample()
        logp = dist.log_prob(a)
        v = self.critic(obs).squeeze(-1)
        return a, logp, v


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 1024
    hidden: int = 128
    max_grad_norm: float = 0.5


class PPO:
    """Single-policy PPO. Caller is responsible for filtering transitions
    to those belonging to the policy being updated (e.g. one team)."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ac = ActorCritic(obs_dim, act_dim, cfg.hidden).to(device)
        self.opt = optim.Adam(self.ac.parameters(), lr=cfg.lr)

    def save(self, path: str, episode: int, returns: list, extra: Optional[dict] = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "episode": episode,
            "returns": returns,
            "ac_state_dict": self.ac.state_dict(),
            "opt_state_dict": self.opt.state_dict(),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "hidden": self.cfg.hidden,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.ac.load_state_dict(ckpt["ac_state_dict"])
        self.opt.load_state_dict(ckpt["opt_state_dict"])
        return ckpt["episode"], ckpt["returns"]

    @staticmethod
    def _gae(rews, dones, vals, gamma, lam):
        """Returns (advantages, returns). All inputs/outputs are flat 1D over time steps.
        Bootstraps with V=0 at episode end (per-step `dones` already encodes terminal)."""
        n = len(rews)
        adv = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in range(n - 1, -1, -1):
            nonterminal = 1.0 - float(dones[t])
            next_v = float(vals[t + 1]) if t + 1 < n else 0.0
            delta = float(rews[t]) + gamma * next_v * nonterminal - float(vals[t])
            last_gae = delta + gamma * lam * nonterminal * last_gae
            adv[t] = last_gae
        rets = adv + np.asarray(vals, dtype=np.float32)
        return adv, rets

    def update(self, obs, acts, logps, rews, dones, vals) -> Dict[str, float]:
        n = len(obs)
        if n == 0:
            return {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "samples": 0}
        assert len(acts) == n and len(logps) == n and len(vals) == n and len(rews) == n and len(dones) == n, \
            f"Buffer mismatch: {len(obs)=} {len(acts)=} {len(logps)=} {len(rews)=} {len(dones)=} {len(vals)=}"
        cfg = self.cfg

        adv_np, rets_np = self._gae(rews, dones, vals, cfg.gamma, cfg.gae_lambda)
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        acts_t = torch.as_tensor(np.asarray(acts), dtype=torch.int64, device=self.device)
        old_logps = torch.as_tensor(np.asarray(logps), dtype=torch.float32, device=self.device)
        old_vals = torch.as_tensor(np.asarray(vals), dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(adv_np, dtype=torch.float32, device=self.device)
        rets = torch.as_tensor(rets_np, dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idx = np.arange(n)
        pg_acc = v_acc = ent_acc = 0.0
        n_batches = 0
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, cfg.minibatch_size):
                b = idx[start:start + cfg.minibatch_size]
                o, a, ol, ad, rt, ov = obs_t[b], acts_t[b], old_logps[b], adv[b], rets[b], old_vals[b]
                logits = self.ac.actor(o)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(a)
                ratio = (logp - ol).exp()
                clip_adv = torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * ad
                pg_loss = -(torch.min(ratio * ad, clip_adv)).mean()

                v_pred = self.ac.critic(o).squeeze(-1)
                v_clip = ov + (v_pred - ov).clamp(-cfg.clip_coef, cfg.clip_coef)
                v_loss = 0.5 * torch.max((v_pred - rt).pow(2), (v_clip - rt).pow(2)).mean()

                ent = dist.entropy().mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
                self.opt.step()

                pg_acc += pg_loss.item()
                v_acc += v_loss.item()
                ent_acc += ent.item()
                n_batches += 1

        return {
            "pg_loss": pg_acc / max(1, n_batches),
            "v_loss": v_acc / max(1, n_batches),
            "entropy": ent_acc / max(1, n_batches),
            "samples": n,
        }


class TrajectoryRecorder:
    """Records per-episode trajectories to {npz, jsonl} and a run-level manifest.json.

    NPZ layout: obs (T, A, D), act (T, A), pos (T, A, 2) when env supports it, agent_names (A,).
    """

    def __init__(self, save_dir: str, env_id: str, env_cfg: dict, global_seed: int,
                 agent_names: list, rec_cfg: Optional[Dict] = None):
        rec_cfg = rec_cfg or {}
        self.run_dir = save_dir
        self.sample_rate = int(rec_cfg.get("sample_rate", 1))
        os.makedirs(self.run_dir, exist_ok=True)
        self.step_obs: list = []   # list of per-step (A, D_pad) arrays
        self.step_acts: list = []  # list of per-step (A,) arrays
        self.step_pos: list = []   # list of per-step (A, 2) arrays
        self.buffer: list = []     # JSONL records

        self.env_id = env_id
        self.env_cfg = env_cfg
        self.global_seed = global_seed
        self.agent_names = list(agent_names)
        self.episode_seeds: list = []

    def _pad_row(self, vec, target_len):
        v = np.asarray(vec, dtype=np.float32).ravel()
        if v.size < target_len:
            v = np.concatenate([v, np.zeros(target_len - v.size, dtype=np.float32)])
        elif v.size > target_len:
            v = v[:target_len]
        return v

    def record_step(self, t, obs_dict, acts_dict, rewards_dict, done_any, infos):
        for agent_id in self.agent_names:
            self.buffer.append({
                "t": t, "agent_id": agent_id,
                "obs": np.asarray(obs_dict[agent_id]).tolist(),
                "act": int(acts_dict[agent_id]),
                "rew": float(rewards_dict[agent_id]),
                "done": bool(done_any),
                "info": infos.get(agent_id, {}),
            })

        max_d = max(int(np.size(obs_dict[a])) for a in self.agent_names)
        row_obs = np.stack([self._pad_row(obs_dict[a], max_d) for a in self.agent_names], axis=0)
        row_acts = np.array([int(acts_dict[a]) for a in self.agent_names], dtype=np.int64)
        self.step_obs.append(row_obs)
        self.step_acts.append(row_acts)

        if "simple_tag" in self.env_id:
            row_pos = np.stack(
                [np.asarray(obs_dict[a], dtype=np.float32)[2:4] for a in self.agent_names], axis=0
            )
            self.step_pos.append(row_pos)

    def save(self, episode_idx: int, episode_seed: int):
        if not self.buffer:
            return
        self.episode_seeds.append(episode_seed)
        path_base = os.path.join(self.run_dir, f"ep_{episode_idx}")

        with open(f"{path_base}.jsonl", "w") as f:
            for item in self.buffer:
                f.write(json.dumps(item) + "\n")

        max_d = max(r.shape[1] for r in self.step_obs)
        obs_padded = [np.pad(r, ((0, 0), (0, max_d - r.shape[1]))) for r in self.step_obs]
        obs_mat = np.stack(obs_padded, axis=0).astype(np.float32)  # (T, A, D)
        act_mat = np.stack(self.step_acts, axis=0)                 # (T, A)
        payload = {
            "obs": obs_mat,
            "act": act_mat,
            "agent_names": np.asarray(self.agent_names),
        }
        if self.step_pos:
            payload["pos"] = np.stack(self.step_pos, axis=0).astype(np.float32)  # (T, A, 2)

        np.savez_compressed(f"{path_base}.npz", **payload)

        self.buffer.clear()
        self.step_obs.clear()
        self.step_acts.clear()
        self.step_pos.clear()

    def save_manifest(self):
        agent_roles = ["predator" if "adversary" in name else "prey" for name in self.agent_names]
        manifest = {
            "env_id": self.env_id,
            "env_cfg": self.env_cfg,
            "global_seed": self.global_seed,
            "episode_seeds": self.episode_seeds,
            "agent_names": self.agent_names,
            "agent_roles": agent_roles,
        }
        with open(os.path.join(self.run_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)


def flatten_obs(obs_in: Union[Dict[str, np.ndarray], tuple]) -> Tuple[np.ndarray, List[str]]:
    o = obs_in
    while isinstance(o, tuple):
        if len(o) == 0:
            raise ValueError("Empty tuple observations")
        o = o[0]
    if not isinstance(o, dict):
        raise TypeError(f"Expected dict, got {type(o)}")
    agents = sorted(o.keys())
    vecs = [np.asarray(o[a]).ravel() for a in agents]
    m = max(v.size for v in vecs)
    padded = [np.pad(v, (0, m - v.size)) for v in vecs]
    return np.stack(padded, axis=0), agents


def split_teams(agent_names: List[str]) -> Tuple[List[str], List[str]]:
    """Predators ('adversary' in name) vs prey. Order preserved from input."""
    pred = [a for a in agent_names if "adversary" in a]
    prey = [a for a in agent_names if "adversary" not in a]
    return pred, prey
