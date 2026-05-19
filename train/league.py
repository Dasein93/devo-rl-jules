"""League of frozen policy snapshots, sampled as opponents to stabilize co-adapting MARL.

Each team has its own League. When `snapshot_every` episodes have passed, the live
ActorCritic is cloned and persisted to disk. During an episode the trainer asks
the opponent league for a snapshot with probability `opponent_prob`; if returned,
the opponent acts under that frozen policy and its transitions are discarded.
"""
import os
import copy
import glob
from typing import Optional, List

import numpy as np
import torch

from train.ppo import ActorCritic


class League:
    def __init__(self, save_dir: str, team: str, snapshot_every: int = 25,
                 opponent_prob: float = 0.3, max_snapshots: int = 20, device: str = "cpu",
                 rng: Optional[np.random.Generator] = None):
        self.save_dir = save_dir
        self.team = team
        self.snapshot_every = max(1, int(snapshot_every))
        self.opponent_prob = float(opponent_prob)
        self.max_snapshots = int(max_snapshots)
        self.device = device
        self.rng = rng or np.random.default_rng()
        os.makedirs(self.save_dir, exist_ok=True)
        self._cached: List[str] = []
        self._scan()

    def _scan(self):
        self._cached = sorted(glob.glob(os.path.join(self.save_dir, "snap_*.pt")))

    def __len__(self):
        return len(self._cached)

    def maybe_snapshot(self, ac: ActorCritic, episode: int) -> Optional[str]:
        if episode % self.snapshot_every != 0:
            return None
        return self.snapshot(ac, episode)

    def snapshot(self, ac: ActorCritic, episode: int) -> str:
        path = os.path.join(self.save_dir, f"snap_{episode:06d}.pt")
        payload = {
            "ac_state_dict": ac.state_dict(),
            "obs_dim": ac.obs_dim,
            "act_dim": ac.act_dim,
            "hidden": ac.hidden,
            "episode": episode,
        }
        torch.save(payload, path)
        self._cached.append(path)
        self._cached.sort()
        self._evict()
        return path

    def _evict(self):
        while len(self._cached) > self.max_snapshots:
            victim = self._cached.pop(0)
            try:
                os.remove(victim)
            except OSError:
                pass

    def sample(self) -> Optional[ActorCritic]:
        """Return a frozen ActorCritic (eval mode, no_grad), or None if probability or pool fails."""
        if not self._cached or self.opponent_prob <= 0.0:
            return None
        if self.rng.random() > self.opponent_prob:
            return None
        path = self._cached[self.rng.integers(0, len(self._cached))]
        return self._load(path)

    def _load(self, path: str) -> ActorCritic:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        ac = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"], ckpt["hidden"]).to(self.device)
        ac.load_state_dict(ckpt["ac_state_dict"])
        ac.eval()
        for p in ac.parameters():
            p.requires_grad_(False)
        return ac
