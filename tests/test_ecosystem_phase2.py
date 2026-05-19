"""Phase 2 ecosystem tests: reproduction, food field, dynamic population growth."""
import numpy as np
import pytest

from train.ecosystem_env import make_ecosystem_env


def test_reproduction_fires_when_thresholds_met():
    """An agent with high energy and sufficient age should spawn a child."""
    env = make_ecosystem_env(
        seed=0,
        n_predators_start=1, n_prey_start=0,    # only one predator
        max_predators=2, max_prey=0,            # room for one child
        repro_pred_energy_thresh=10.0,
        repro_min_age=2,
        repro_cooldown=0,
        # Disable everything else that could interfere.
        pred_energy_cost=0.0, prey_energy_cost=0.0,
        attack_range=0.0,
    )
    env.reset(seed=0)
    # Predator at full energy and HP.
    env._energy[0] = 100.0
    env._age[0] = 100  # past min_age
    # Step a few times to let reproduction logic see the thresholds.
    for step in range(3):
        obs, rewards, terms, truncs, infos = env.step({a: 0 for a in env.agents})
        if env._alive[1]:
            break
    assert env._alive[1], "Child predator should have been spawned"
    # Parent paid the reproduction cost (energy halved).
    assert env._energy[0] <= 60.0


def test_reproduction_blocked_when_team_at_capacity():
    env = make_ecosystem_env(
        seed=0,
        n_predators_start=2, n_prey_start=0,
        max_predators=2, max_prey=0,           # already full
        repro_pred_energy_thresh=10.0,
        repro_min_age=1, repro_cooldown=0,
        pred_energy_cost=0.0, attack_range=0.0,
    )
    env.reset(seed=0)
    env._energy[:2] = 100.0
    env._age[:2] = 100
    for _ in range(3):
        env.step({a: 0 for a in env.agents})
    # Pop should still be 2, no growth.
    assert int(env._alive.sum()) == 2


def test_reproduction_cooldown_prevents_back_to_back_births():
    env = make_ecosystem_env(
        seed=0,
        n_predators_start=1, n_prey_start=0,
        max_predators=5, max_prey=0,
        repro_pred_energy_thresh=10.0,
        repro_min_age=1,
        repro_cooldown=999,                    # effectively never lets the parent reproduce twice
        pred_energy_cost=0.0, attack_range=0.0,
    )
    env.reset(seed=0)
    env._energy[0] = 100.0
    env._age[0] = 100
    for _ in range(10):
        env.step({a: 0 for a in env.agents})
    # One birth, no more.
    assert int(env._alive.sum()) == 2


def test_child_starts_at_partial_energy_and_full_hp():
    env = make_ecosystem_env(
        seed=0,
        n_predators_start=1, n_prey_start=0,
        max_predators=2, max_prey=0,
        repro_pred_energy_thresh=10.0,
        repro_min_age=1, repro_cooldown=0,
        repro_child_hp_frac=1.0, repro_child_energy_frac=0.5,
        pred_max_hp=100.0, pred_max_energy=100.0,
        pred_energy_cost=0.0, attack_range=0.0,
        genome_enabled=False,            # pin genome to 1.0 for this test
    )
    env.reset(seed=0)
    env._energy[0] = 100.0
    env._age[0] = 100
    env.step({a: 0 for a in env.agents})
    assert env._alive[1]
    assert env._hp[1] == pytest.approx(100.0)
    assert env._energy[1] == pytest.approx(50.0)


def test_food_grid_regenerates_after_being_eaten():
    env = make_ecosystem_env(
        seed=0,
        n_predators_start=0, n_prey_start=1,
        max_predators=0, max_prey=1,
        prey_energy_cost=0.0,
        food_regen_rate=2.0, food_cell_max=10.0, food_eat_rate=10.0,
    )
    env.reset(seed=0)
    # Force prey position so we know which grid cell to inspect.
    env._pos[env._max_pred] = np.array([0.0, 0.0], dtype=np.float32)
    gx, gy = env._world_to_grid(env._pos[env._max_pred])
    # Step once → prey eats from this cell.
    env.step({"agent_0": 0})
    # Reset food at that cell to 0 to isolate the regen test.
    env._food[gx, gy] = 0.0
    # Move prey away so it doesn't keep eating this cell.
    env._pos[env._max_pred] = np.array([0.8, 0.8], dtype=np.float32)
    for _ in range(3):
        env.step({"agent_0": 0})
    assert env._food[gx, gy] > 0.0, "Food cell should regenerate when no one eats it"


def test_prey_starves_without_food():
    env = make_ecosystem_env(
        seed=0,
        n_predators_start=0, n_prey_start=1,
        max_predators=0, max_prey=1,
        prey_max_energy=3.0, prey_energy_cost=1.0,
        food_regen_rate=0.0, food_cell_max=0.0,   # no food anywhere
    )
    env.reset(seed=0)
    for _ in range(10):
        obs, rewards, terms, truncs, infos = env.step({a: 0 for a in env.agents})
        if not env._alive[env._max_pred]:
            break
    assert not env._alive[env._max_pred], "Prey should starve when no food is available"


def test_possible_agents_lists_full_roster():
    env = make_ecosystem_env(
        seed=0, n_predators_start=2, n_prey_start=3,
        max_predators=5, max_prey=4,
    )
    env.reset(seed=0)
    possible = env.possible_agents
    assert len(possible) == 5 + 4
    assert all(n.startswith("adversary_") or n.startswith("agent_") for n in possible)
    # Only the starting subset is alive immediately after reset.
    assert len(env.agents) == 2 + 3


def test_reproduction_pays_parent_reward():
    env = make_ecosystem_env(
        seed=0,
        n_predators_start=1, n_prey_start=0,
        max_predators=2, max_prey=0,
        repro_pred_energy_thresh=10.0,
        repro_min_age=1, repro_cooldown=0,
        reward_per_birth=7.5,
        pred_energy_cost=0.0, attack_range=0.0, survival_bonus=0.0, boundary_penalty=0.0,
    )
    env.reset(seed=0)
    env._energy[0] = 100.0
    env._age[0] = 100
    obs, rewards, _, _, _ = env.step({a: 0 for a in env.agents})
    assert rewards["adversary_0"] == pytest.approx(7.5)
