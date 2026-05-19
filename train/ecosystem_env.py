"""Custom predator/prey ecosystem env with per-agent HP, energy, mortality,
reproduction, foraging, and heritable genomes.

PettingZoo parallel-API compatible. Agent names follow the simple_tag convention
("adversary_*" for predators, "agent_*" for prey) so `split_teams` and the
league / replay tooling work unchanged.

Phases:
- Phase 1 (mortality only): no reproduction, no food. Population only shrinks.
- Phase 2: reproduction in free slots, food field for prey.
- Phase 3 (this file): heritable genome (speed, HP, sense radius). Children
  inherit the parent's genome with Gaussian mutation.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EcosystemConfig:
    # Starting populations and maxima. Slots = max_predators + max_prey.
    n_predators_start: int = 10
    n_prey_start: int = 10
    max_predators: int = 30
    max_prey: int = 30

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

    # Energy decay
    pred_energy_cost: float = 0.5
    prey_energy_cost: float = 0.3     # Phase 2: prey now starve too if they don't forage

    # Food field (only prey forage; covers world in a coarse grid)
    food_grid_size: int = 16          # NxN cells
    food_cell_max: float = 10.0
    food_regen_rate: float = 0.10     # added to every cell each step (capped at food_cell_max)
    food_eat_rate: float = 5.0        # prey energy gained per step per food unit consumed

    # Reproduction
    repro_pred_energy_thresh: float = 80.0
    repro_prey_energy_thresh: float = 70.0
    repro_min_age: int = 50           # steps an agent must live before it can reproduce
    repro_cooldown: int = 30          # steps after reproducing before it can again
    repro_child_hp_frac: float = 1.0  # child starts at this fraction of max_hp
    repro_child_energy_frac: float = 0.5
    repro_parent_energy_keep: float = 0.5  # parent keeps this fraction after birth

    # Rewards
    reward_per_hit: float = 10.0      # predator gets +reward_per_hit, prey gets -reward_per_hit
    reward_per_death: float = -50.0   # given to the agent that just died this step
    reward_per_birth: float = 5.0     # given to the parent when a child is spawned
    survival_bonus: float = 0.05      # per step alive

    # Boundary (soft penalty when outside [-0.9, 0.9])
    boundary_threshold: float = 0.9
    boundary_penalty: float = 5.0     # multiplied by overshoot^2

    # Genome (Phase 3). Per-agent multipliers on (speed, hp_cap, sense_radius).
    # Children inherit parent's genome plus Gaussian noise (mutation_sigma).
    # All genes are clamped to [genome_min, genome_max].
    genome_enabled: bool = True
    genome_init_sigma: float = 0.08   # std-dev of initial uniform genome variation
    mutation_sigma: float = 0.05
    genome_min: float = 0.5
    genome_max: float = 2.0

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

        # Slot allocation: roster size = max_predators + max_prey. Starting
        # alive count = n_predators_start + n_prey_start; the rest are empty
        # slots that births can occupy.
        self._max_pred = int(max(self.cfg.max_predators, self.cfg.n_predators_start))
        self._max_prey = int(max(self.cfg.max_prey, self.cfg.n_prey_start))
        self._n_start_pred = int(self.cfg.n_predators_start)
        self._n_start_prey = int(self.cfg.n_prey_start)
        self._pred_names = [f"adversary_{i}" for i in range(self._max_pred)]
        self._prey_names = [f"agent_{i}" for i in range(self._max_prey)]
        self._all_names = self._pred_names + self._prey_names

        # State arrays sized by max roster.
        N = self._max_pred + self._max_prey
        self._N = N
        self._pos = np.zeros((N, 2), dtype=np.float32)
        self._vel = np.zeros((N, 2), dtype=np.float32)
        self._hp = np.zeros(N, dtype=np.float32)
        self._energy = np.zeros(N, dtype=np.float32)
        self._age = np.zeros(N, dtype=np.int32)
        self._cooldown = np.zeros(N, dtype=np.int32)  # steps since last reproduction
        self._alive = np.zeros(N, dtype=bool)
        self._is_pred = np.zeros(N, dtype=bool)
        self._is_pred[: self._max_pred] = True
        # Genome: 3 traits per agent — speed_g, hp_g, sense_g — each a multiplier
        # on the team's base stat. Initialised to ~1 with small noise, mutated on
        # reproduction, clamped to [genome_min, genome_max].
        self._genome = np.ones((N, 3), dtype=np.float32)
        self._t = 0
        self._rng = np.random.default_rng(self.cfg.seed)

        # Food field for prey foraging.
        G = self.cfg.food_grid_size
        self._food = np.full((G, G), self.cfg.food_cell_max, dtype=np.float32)

        # Obs: own (6) + K nearest neighbours (6 each) + own-cell food (1)
        # + own genome (3).
        self._obs_dim = 6 + 6 * self.cfg.max_neighbors_obs + 1 + 3
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

        # Empty all slots first.
        self._pos[:] = 0.0
        self._vel[:] = 0.0
        self._hp[:] = 0.0
        self._energy[:] = 0.0
        self._age[:] = 0
        self._cooldown[:] = 0
        self._alive[:] = False

        # Initial genome: uniform random in [1 - σ, 1 + σ] per trait, clamped.
        if self.cfg.genome_enabled:
            self._genome[:] = np.clip(
                1.0 + self._rng.normal(0.0, self.cfg.genome_init_sigma, size=(self._N, 3)),
                self.cfg.genome_min, self.cfg.genome_max,
            ).astype(np.float32)
        else:
            self._genome[:] = 1.0

        # Spawn the starting roster: first n_start_pred predator slots, first n_start_prey prey slots.
        pred_start_idx = np.arange(self._n_start_pred)
        prey_start_idx = np.arange(self._n_start_prey) + self._max_pred
        for idx_arr, is_pred in ((pred_start_idx, True), (prey_start_idx, False)):
            self._alive[idx_arr] = True
            self._pos[idx_arr] = self._rng.uniform(
                -half * 0.8, half * 0.8, size=(len(idx_arr), 2)
            ).astype(np.float32)
            # HP cap scales with hp_g (trait[1]) so genome immediately matters.
            base_hp = self.cfg.pred_max_hp if is_pred else self.cfg.prey_max_hp
            self._hp[idx_arr] = base_hp * self._genome[idx_arr, 1]
            self._energy[idx_arr] = self.cfg.pred_max_energy if is_pred else self.cfg.prey_max_energy

        # Refill food grid.
        self._food[:] = self.cfg.food_cell_max
        self._t = 0
        obs = self._build_obs_dict()
        infos = {a: {} for a in obs}
        return obs, infos

    def step(self, actions: Dict[str, int]):
        cfg = self.cfg
        half = cfg.world_size / 2.0

        # 1. Apply movement for alive agents only. Velocity = action_dir * effective speed,
        # where effective speed = base_speed * genome[i, 0] (speed gene).
        for name, action in actions.items():
            i = self._index(name)
            if not self._alive[i]:
                continue
            base_speed = cfg.pred_speed if self._is_pred[i] else cfg.prey_speed
            speed = base_speed * float(self._genome[i, 0])
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

        # 3a. Prey foraging from the food field.
        for i in range(self._max_pred, self._N):
            if not self._alive[i]:
                continue
            gx, gy = self._world_to_grid(self._pos[i])
            available = float(self._food[gx, gy])
            eaten = min(available, cfg.food_eat_rate)
            self._food[gx, gy] -= eaten
            self._energy[i] = min(self._energy_cap(i), self._energy[i] + eaten)

        # 3b. Food grid regenerates uniformly.
        self._food += cfg.food_regen_rate
        np.clip(self._food, 0.0, cfg.food_cell_max, out=self._food)

        # 3c. Energy decay for all alive agents.
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

        # 6. Resolve deaths. Both teams die at HP<=0 OR energy<=0 (Phase 2:
        # prey can starve too because they now lose energy each step and must
        # forage to refill).
        terminations = {name: False for name in self.agents}
        for i in range(self._N):
            if not self._alive[i]:
                continue
            died = False
            if self._hp[i] <= 0.0:
                died = True
            elif self._energy[i] <= 0.0:
                died = True
            if died:
                rewards[self._all_names[i]] += cfg.reward_per_death
                self._alive[i] = False
                terminations[self._all_names[i]] = True

        # 6b. Age + cooldown bookkeeping, then reproduction.
        for i in range(self._N):
            if not self._alive[i]:
                continue
            self._age[i] += 1
            if self._cooldown[i] > 0:
                self._cooldown[i] -= 1
        births = self._maybe_reproduce(rewards)
        # Births spawned this step are not present in `terminations` yet; we
        # still emit obs for them in the next-obs dict.

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

    def _energy_cap(self, i: int) -> float:
        return self.cfg.pred_max_energy if self._is_pred[i] else self.cfg.prey_max_energy

    def _hp_cap(self, i: int) -> float:
        base = self.cfg.pred_max_hp if self._is_pred[i] else self.cfg.prey_max_hp
        return base * float(self._genome[i, 1])

    def _sense_radius(self, i: int) -> float:
        base = self.cfg.pred_sense_radius if self._is_pred[i] else self.cfg.prey_sense_radius
        return base * float(self._genome[i, 2])

    def _world_to_grid(self, pos: np.ndarray) -> Tuple[int, int]:
        """Map a 2D world coordinate to (row, col) in the food grid."""
        G = self.cfg.food_grid_size
        half = self.cfg.world_size / 2.0
        # Normalize to [0, 1] then bucket into G cells.
        u = (float(pos[0]) + half) / max(1e-6, self.cfg.world_size)
        v = (float(pos[1]) + half) / max(1e-6, self.cfg.world_size)
        gx = int(np.clip(int(u * G), 0, G - 1))
        gy = int(np.clip(int(v * G), 0, G - 1))
        return gx, gy

    def _free_slot(self, is_pred: bool) -> Optional[int]:
        if is_pred:
            for i in range(self._max_pred):
                if not self._alive[i]:
                    return i
        else:
            for i in range(self._max_pred, self._N):
                if not self._alive[i]:
                    return i
        return None

    def _maybe_reproduce(self, rewards: Dict[str, float]) -> List[int]:
        """Check each alive agent against reproduction triggers; spawn children
        in free slots. Returns list of newly-spawned slot indices."""
        cfg = self.cfg
        births: List[int] = []
        half = cfg.world_size / 2.0
        # Snapshot the parents BEFORE any births to avoid the new child instantly
        # qualifying to reproduce in the same step.
        candidates = []
        for i in range(self._N):
            if not self._alive[i]:
                continue
            if self._age[i] < cfg.repro_min_age:
                continue
            if self._cooldown[i] > 0:
                continue
            thresh = cfg.repro_pred_energy_thresh if self._is_pred[i] else cfg.repro_prey_energy_thresh
            if self._energy[i] < thresh:
                continue
            candidates.append(i)

        for i in candidates:
            child_slot = self._free_slot(bool(self._is_pred[i]))
            if child_slot is None:
                continue  # team is at capacity
            # Pay reproduction cost.
            self._energy[i] *= cfg.repro_parent_energy_keep
            self._cooldown[i] = cfg.repro_cooldown

            # Inherit parent's genome with Gaussian mutation, clamped.
            if cfg.genome_enabled:
                mutation = self._rng.normal(0.0, cfg.mutation_sigma, size=3).astype(np.float32)
                self._genome[child_slot] = np.clip(
                    self._genome[i] + mutation, cfg.genome_min, cfg.genome_max,
                )
            else:
                self._genome[child_slot] = 1.0

            # Spawn child near the parent, full HP (per its own genome), partial energy.
            offset = self._rng.uniform(-0.05, 0.05, size=2).astype(np.float32)
            self._pos[child_slot] = np.clip(self._pos[i] + offset, -half, half)
            self._vel[child_slot] = 0.0
            self._hp[child_slot] = self._hp_cap(child_slot) * cfg.repro_child_hp_frac
            self._energy[child_slot] = self._energy_cap(child_slot) * cfg.repro_child_energy_frac
            self._age[child_slot] = 0
            self._cooldown[child_slot] = cfg.repro_cooldown
            self._alive[child_slot] = True
            births.append(child_slot)

            # Reward the parent.
            rewards[self._all_names[i]] = rewards.get(self._all_names[i], 0.0) + cfg.reward_per_birth

        return births

    def mean_genome(self, team: str) -> np.ndarray:
        """Return mean (speed_g, hp_g, sense_g) over currently-alive agents of `team`.
        Returns NaN array if no agents alive."""
        if team == "predator":
            idx = np.arange(self._max_pred)
        elif team == "prey":
            idx = np.arange(self._max_pred, self._N)
        else:
            raise ValueError(team)
        mask = self._alive[idx]
        if not mask.any():
            return np.full(3, np.nan, dtype=np.float32)
        return self._genome[idx][mask].mean(axis=0)

    def _build_obs_dict(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        out: Dict[str, np.ndarray] = {}
        alive_idx = np.where(self._alive)[0]
        for i in alive_idx:
            obs = np.zeros(self._obs_dim, dtype=np.float32)
            is_pred = bool(self._is_pred[i])
            max_hp = cfg.pred_max_hp if is_pred else cfg.prey_max_hp
            max_energy = cfg.pred_max_energy if is_pred else cfg.prey_max_energy
            sense_r = self._sense_radius(i)

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

            # Own-cell food scalar + own genome (last 1+3 dims).
            gx, gy = self._world_to_grid(self._pos[i])
            obs[-4] = float(self._food[gx, gy]) / max(1e-6, cfg.food_cell_max)
            obs[-3:] = self._genome[i]

            out[self._all_names[i]] = obs
        return out


def make_ecosystem_env(seed: int = 42, **overrides) -> EcosystemEnv:
    """Factory used by the trainer when env.id == 'ecosystem'."""
    cfg = EcosystemConfig(seed=seed)
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return EcosystemEnv(cfg=cfg)
