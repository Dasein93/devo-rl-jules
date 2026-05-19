"""Tests for the Phase-1 ecosystem env: mortality, deterministic mechanics, API shape."""
import numpy as np
import pytest

from train.ecosystem_env import EcosystemConfig, EcosystemEnv, make_ecosystem_env


def _stationary_actions(env):
    """All alive agents take action 0 (no_op)."""
    return {a: 0 for a in env.agents}


def test_reset_returns_full_roster():
    env = make_ecosystem_env(seed=42, n_predators_start=3, n_prey_start=4)
    obs, infos = env.reset(seed=42)
    assert sorted(obs.keys()) == sorted([f"adversary_{i}" for i in range(3)] + [f"agent_{i}" for i in range(4)])
    assert len(env.agents) == 7
    for v in obs.values():
        assert v.shape == (env._obs_dim,)


def test_obs_dim_consistent_across_agents():
    env = make_ecosystem_env(seed=0, n_predators_start=2, n_prey_start=2, max_neighbors_obs=4)
    obs, _ = env.reset(seed=0)
    dims = {v.shape[0] for v in obs.values()}
    assert len(dims) == 1
    assert next(iter(dims)) == 6 + 6 * 4


def test_action_and_observation_space_shapes():
    env = make_ecosystem_env(seed=0, n_predators_start=2, n_prey_start=2)
    env.reset(seed=0)
    a0 = env.agents[0]
    assert env.action_space(a0).n == 5
    assert env.observation_space(a0).shape == (env._obs_dim,)


def test_prey_dies_after_enough_hits():
    """Place a predator on top of a prey so they collide every step. Prey
    should die when accumulated damage exceeds prey_max_hp."""
    env = make_ecosystem_env(
        seed=0, n_predators_start=1, n_prey_start=1,
        prey_max_hp=30.0, attack_damage=15.0, attack_range=10.0,  # huge range, always hits
    )
    env.reset(seed=0)
    # Force both agents to the origin so they always collide.
    env._pos[:] = 0.0
    # Step until prey dies. Damage 15/step, HP 30 → dies on the 2nd hit.
    for _ in range(5):
        obs, rewards, terms, truncs, infos = env.step(_stationary_actions(env))
        if "agent_0" in terms and terms["agent_0"]:
            break
    assert "agent_0" not in env.agents, "Prey should have been removed after enough damage"
    assert "adversary_0" in env.agents, "Predator should still be alive"


def test_team_extinction_terminates_episode():
    env = make_ecosystem_env(
        seed=0, n_predators_start=1, n_prey_start=1,
        prey_max_hp=10.0, attack_damage=20.0, attack_range=10.0,
        max_steps=1000,
    )
    env.reset(seed=0)
    env._pos[:] = 0.0
    # One hit kills the prey at this damage/HP setting.
    obs, rewards, terms, truncs, infos = env.step(_stationary_actions(env))
    # Prey terminated; survivor (predator) gets truncated=True (episode-level end).
    assert terms["agent_0"] is True
    assert truncs.get("adversary_0", False) is True


def test_predator_starves_when_energy_depleted():
    env = make_ecosystem_env(
        seed=0, n_predators_start=1, n_prey_start=1,
        pred_max_energy=2.0, pred_energy_cost=1.0,
        attack_range=0.0,  # no collisions → can't refuel
    )
    env.reset(seed=0)
    # Place agents far apart so predator can't eat.
    env._pos[0] = np.array([-0.5, 0.0], dtype=np.float32)
    env._pos[1] = np.array([0.5, 0.0], dtype=np.float32)
    # 2 energy, cost 1/step → starves on step 2 (energy hits 0 on step 2, dies same step).
    for _ in range(5):
        obs, rewards, terms, truncs, infos = env.step({a: 0 for a in env.agents})
        if not env._alive[0]:
            break
    assert not env._alive[0], "Predator should starve with no food"


def test_collision_rewards_match_config():
    env = make_ecosystem_env(
        seed=0, n_predators_start=1, n_prey_start=1,
        reward_per_hit=10.0, attack_damage=1.0, attack_range=10.0,
        survival_bonus=0.0, boundary_penalty=0.0,
    )
    env.reset(seed=0)
    env._pos[:] = 0.0
    obs, rewards, _, _, _ = env.step(_stationary_actions(env))
    # Predator inside attack range of prey → +10 / -10
    assert rewards["adversary_0"] == pytest.approx(10.0)
    assert rewards["agent_0"] == pytest.approx(-10.0)


def test_dead_agent_not_in_subsequent_obs():
    env = make_ecosystem_env(
        seed=0, n_predators_start=1, n_prey_start=2,
        prey_max_hp=5.0, attack_damage=50.0, attack_range=0.05,  # small range
    )
    env.reset(seed=0)
    env._pos[0] = np.array([0.0, 0.0], dtype=np.float32)   # predator
    env._pos[1] = np.array([0.001, 0.0], dtype=np.float32) # agent_0 in range → dies
    env._pos[2] = np.array([0.5, 0.5], dtype=np.float32)   # agent_1 out of range → survives
    obs, rewards, terms, truncs, infos = env.step({a: 0 for a in env.agents})
    assert "agent_0" not in obs, "Dead prey should not appear in next obs"
    assert "agent_1" in obs, "Surviving prey should still be observable"


def test_episode_max_steps_truncation():
    env = make_ecosystem_env(
        seed=0, n_predators_start=2, n_prey_start=2,
        max_steps=5, attack_range=0.0,  # no kills so episode can hit max_steps
        pred_energy_cost=0.0,
    )
    env.reset(seed=0)
    for step in range(5):
        obs, rewards, terms, truncs, infos = env.step({a: 0 for a in env.agents})
    # After 5 steps the env should signal episode truncation for all alive agents.
    assert any(truncs.values()), "max_steps should produce truncation"


def test_per_agent_step_count_in_info():
    env = make_ecosystem_env(seed=0, n_predators_start=2, n_prey_start=2)
    env.reset(seed=0)
    obs, rewards, terms, truncs, infos = env.step({a: 0 for a in env.agents})
    # First-agent info should carry population counts.
    first_info = infos[next(iter(infos.keys()))]
    assert "n_pred_alive" in first_info
    assert "n_prey_alive" in first_info
