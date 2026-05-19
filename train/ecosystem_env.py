"""Custom predator/prey ecosystem env with per-agent HP, energy and mortality.

PettingZoo parallel-API compatible. Agent names follow the simple_tag convention
("adversary_*" for predators, "agent_*" for prey) so `split_teams` and the
league / replay tooling work unchanged.

Phase 1 of the ecosystem expansion: no reproduction, no food field, no genome.
Initial roster is the maximum; agents only leave (death), never join.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EcosystemConfig:
    n_predators_start: int = 10
    n_prey_start: int = 10
    world_size: float = 2.0           # world spans [-world_size/2, world_size/2]^2
    max_steps: int = 1000
    max_neighbors_obs: int = 4        # K nearest visible neighbours in obs

    # Predator
    pred_max_hp: float = 100.0
    pred_max_energy: float = 100.0
    pred_speed: float = 0.05
    pred_sense_radius: float = 0.6

    # Prey
    prey_max_hp: float = 30.0
    prey_max_energy: float = 100.0
    prey_speed: float = 0.065         # prey are slightly faster (Lotka-Volterra ingredient)
    prey_sense_radius: float = 0.6

    # Combat
    attack_damage: float = 15.0
    attack_range: float = 0.075
    eat_energy_gain: float = 20.0

    # Energy decay (Phase 1: predators starve, prey do not yet — Phase 2 adds food field)
    pred_energy_cost: float = 0.5
    prey_energy_cost: float = 0.0

    # Rewards
    reward_per_hit: float = 10.0      # predator gets +reward_per_hit, prey gets -reward_per_hit
    reward_per_death: float = -50.0   # given to the agent that just died this step
    survival_bonus: float = 0.05      # per step alive

    # Boundary (soft penalty when outside [-0.9, 0.9])
    boundary_threshold: float = 0.9
    boundary_penalty: float = 5.0     # multiplied by overshoot^2

    seed: int = 42


# Action mapping (matches simple_tag): 0=no_op, 1=left, 2=right, 3=down, 4=up
_ACTION_VEC = np.array(
    [[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]],
    dtype=np.float32,
)


class _Box:
    """Minimal gymnasium-style Box shim — avoids a hard gymnasium dep."""
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = low
        self.high = high
        self.shape = tuple(shape)
        self.dtype = dtype


class _Discrete:
    def __init__(self, n: int):
        self.n = n


class EcosystemEnv:
    """Parallel-API env. `agents` is the list of *alive* agent names.

    On reset:
        n_predators_start adversaries + n_prey_start prey, randomly placed,
        full HP and energy.
    On step(actions):
        actions dict keyed by alive agent names. Applies movement, then resolves
        predator-prey collisions, then deducts energy, then kills agents whose
        HP or energy hit zero, then builds the next observation and termination
        dicts only for agents still alive.
    """

    metadata = {"name": "ecosystem_v1"}

    def __init__(self, cfg: Optional[EcosystemConfig] = None, **overrides):
        self.cfg = cfg or EcosystemConfig()
        for k, v in overrides.items():
            if not hasattr(self.cfg, k):
                raise ValueError(f"Unknown ecosystem config key: {k}")
            setattr(self.cfg, k, v)

        self._max_pred = int(self.cfg.n_predators_start)
        self._max_prey = int(self.cfg.n_prey_start)
        self._pred_names = [f"adversary_{i}" for i in range(self._max_pred)]
        self._prey_names = [f"agent_{i}" for i in range(self._max_prey)]
        self._all_names = self._pred_names + self._prey_names

        # State arrays sized by max population. Aligned by index: 0..max_pred-1 are
        # predators, max_pred..max_pred+max_prey-1 are prey.
        N = self._max_pred + self._max_prey
        self._N = N
        self._pos = np.zeros((N, 2), dtype=np.float32)
        self._vel = np.zeros((N, 2), dtype=np.float32)
        self._hp = np.zeros(N, dtype=np.float32)
        self._energy = np.zeros(N, dtype=np.float32)
        self._alive = np.zeros(N, dtype=bool)
        self._is_pred = np.zeros(N, dtype=bool)
        self._is_pred[: self._max_pred] = True
        self._t = 0
        self._rng = np.random.default_rng(self.cfg.seed)

        # Per-agent obs dim is fixed.
        self._obs_dim = 6 + 6 * self.cfg.max_neighbors_obs
        self._obs_space = _Box(-np.inf, np.inf, (self._obs_dim,), np.float32)
        self._act_space = _Discrete(5)

    # --- PettingZoo parallel-API surface ---

    @property
    def agents(self) -> List[str]:
        return [self._all_names[i] for i in range(self._N) if self._alive[i]]

    @property
    def possible_agents(self) -> List[str]:
        return list(self._all_names)

    def observation_space(self, agent: str) -> _Box:
        return self._obs_space

    def action_space(self, agent: str) -> _Discrete:
        return self._act_space

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        half = self.cfg.world_size / 2.0
        self._pos[:] = self._rng.uniform(-half * 0.8, half * 0.8, size=(self._N, 2)).astype(np.float32)
        self._vel[:] = 0.0
        self._hp[: self._max_pred] = self.cfg.pred_max_hp
        self._hp[self._max_pred:] = self.cfg.prey_max_hp
        self._energy[: self._max_pred] = self.cfg.pred_max_energy
        self._energy[self._max_pred:] = self.cfg.prey_max_energy
        self._alive[:] = True
        self._t = 0
        obs = self._build_obs_dict()
        infos = {a: {} for a in obs}
        return obs, infos

    def step(self, actions: Dict[str, int]):
        cfg = self.cfg
        half = cfg.world_size / 2.0

        # 1. Apply movement for alive agents only. Velocity = action_dir * speed.
        for name, action in actions.items():
            i = self._index(name)
            if not self._alive[i]:
                continue
            speed = cfg.pred_speed if self._is_pred[i] else cfg.prey_speed
            self._vel[i] = _ACTION_VEC[int(action) % 5] * speed
            self._pos[i] += self._vel[i]
            self._pos[i] = np.clip(self._pos[i], -half, half)

        # 2. Resolve predator-prey collisions.
        rewards = {name: 0.0 for name in self.agents}
        for ip in range(self._max_pred):
            if not self._alive[ip]:
                continue
            for iv in range(self._max_pred, self._N):
                if not self._alive[iv]:
                    continue
                d = float(np.linalg.norm(self._pos[ip] - self._pos[iv]))
                if d <= cfg.attack_range:
                    self._hp[iv] -= cfg.attack_damage
                    self._energy[ip] = min(cfg.pred_max_energy, self._energy[ip] + cfg.eat_energy_gain)
                    rewards[self._all_names[ip]] += cfg.reward_per_hit
                    rewards[self._all_names[iv]] -= cfg.reward_per_hit

        # 3. Energy decay (predators only in Phase 1).
        for i in range(self._N):
            if not self._alive[i]:
                continue
            cost = cfg.pred_energy_cost if self._is_pred[i] else cfg.prey_energy_cost
            self._energy[i] = max(0.0, self._energy[i] - cost)

        # 4. Boundary penalty (continuous penalty applied to reward).
        for i in range(self._N):
            if not self._alive[i]:
                continue
            overshoot = np.maximum(np.abs(self._pos[i]) - cfg.boundary_threshold, 0.0)
            penalty = cfg.boundary_penalty * float(overshoot[0] ** 2 + overshoot[1] ** 2)
            if penalty > 0.0:
                rewards[self._all_names[i]] -= penalty

        # 5. Survival bonus for alive agents.
        for name in rewards:
            rewards[name] += cfg.survival_bonus

        # 6. Resolve deaths (HP ≤ 0 OR energy ≤ 0 for predators).
        terminations = {name: False for name in self.agents}
        for i in range(self._N):
            if not self._alive[i]:
                continue
            died = False
            if self._hp[i] <= 0.0:
                died = True
            elif self._is_pred[i] and self._energy[i] <= 0.0:
                died = True
            if died:
                rewards[self._all_names[i]] += cfg.reward_per_death
                self._alive[i] = False
                terminations[self._all_names[i]] = True

        # 7. Step counter / truncation.
        self._t += 1
        truncate_all = self._t >= cfg.max_steps
        n_pred_alive = int(self._alive[: self._max_pred].sum())
        n_prey_alive = int(self._alive[self._max_pred:].sum())
        team_extinct = (n_pred_alive == 0) or (n_prey_alive == 0)
        do_truncate = truncate_all or team_extinct
        truncations = {name: do_truncate and not terminations.get(name, False)
                       for name in actions.keys() if not terminations.get(name, False)}
        # Re-key truncations to include only still-alive agents that didn't terminate
        truncations = {a: do_truncate for a in self.agents}

        # 8. Build next observations for surviving agents.
        next_obs = self._build_obs_dict()
        infos = {a: {} for a in self.agents}

        # Include populations in infos of one agent for downstream logging.
        if next_obs:
            first = next(iter(next_obs.keys()))
            infos[first] = {"n_pred_alive": n_pred_alive, "n_prey_alive": n_prey_alive}

        return next_obs, rewards, terminations, truncations, infos

    # --- Helpers ---

    def _index(self, name: str) -> int:
        if name.startswith("adversary_"):
            return int(name.split("_")[1])
        if name.startswith("agent_"):
            return self._max_pred + int(name.split("_")[1])
        raise KeyError(name)

    def _build_obs_dict(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        out: Dict[str, np.ndarray] = {}
        alive_idx = np.where(self._alive)[0]
        for i in alive_idx:
            obs = np.zeros(self._obs_dim, dtype=np.float32)
            is_pred = bool(self._is_pred[i])
            max_hp = cfg.pred_max_hp if is_pred else cfg.prey_max_hp
            max_energy = cfg.pred_max_energy if is_pred else cfg.prey_max_energy
            sense_r = cfg.pred_sense_radius if is_pred else cfg.prey_sense_radius

            # Own state.
            obs[0:2] = self._pos[i]
            obs[2:4] = self._vel[i]
            obs[4] = self._hp[i] / max_hp
            obs[5] = self._energy[i] / max_energy

            # Find nearest visible neighbours (by Euclidean distance).
            others = [j for j in alive_idx if j != i]
            if others:
                deltas = self._pos[others] - self._pos[i]
                dists = np.linalg.norm(deltas, axis=1)
                visible = [(d, j, k) for k, (d, j) in enumerate(zip(dists, others)) if d <= sense_r]
                visible.sort(key=lambda x: x[0])
                for slot, (_d, j, k) in enumerate(visible[: cfg.max_neighbors_obs]):
                    off = 6 + 6 * slot
                    obs[off + 0] = deltas[k, 0]
                    obs[off + 1] = deltas[k, 1]
                    obs[off + 2] = self._vel[j, 0]
                    obs[off + 3] = self._vel[j, 1]
                    obs[off + 4] = 1.0 if self._is_pred[j] else 0.0
                    obs[off + 5] = 0.0 if self._is_pred[j] else 1.0
            out[self._all_names[i]] = obs
        return out


def make_ecosystem_env(seed: int = 42, **overrides) -> EcosystemEnv:
    """Factory used by the trainer when env.id == 'ecosystem'."""
    cfg = EcosystemConfig(seed=seed)
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return EcosystemEnv(cfg=cfg)
