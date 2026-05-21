"""Tests for the obstacle layer: movement collision, line-of-sight blocking, obs slot."""
import numpy as np
import pytest

from train.ecosystem_env import make_ecosystem_env


def test_obstacles_count_zero_by_default():
    env = make_ecosystem_env(seed=0, n_predators_start=2, n_prey_start=2)
    env.reset(seed=0)
    assert env._n_obstacles == 0


def test_obstacles_spawn_at_reset_and_dont_overlap():
    env = make_ecosystem_env(
        seed=0, n_predators_start=2, n_prey_start=2,
        obstacles_count=5,
        obstacles_radius_min=0.05, obstacles_radius_max=0.08,
        world_size=2.0,
    )
    env.reset(seed=0)
    assert env._obs_centers.shape == (5, 2)
    assert env._obs_radii.shape == (5,)
    assert (env._obs_radii >= 0.05).all() and (env._obs_radii <= 0.08).all()
    # No pair of obstacle centres should overlap (within sum of radii + small margin).
    for i in range(5):
        for j in range(i + 1, 5):
            d = float(np.linalg.norm(env._obs_centers[i] - env._obs_centers[j]))
            assert d >= float(env._obs_radii[i] + env._obs_radii[j])


def test_obstacle_pushes_agent_out_on_movement_collision():
    env = make_ecosystem_env(
        seed=0, n_predators_start=1, n_prey_start=0,
        max_predators=1, max_prey=0,
        obstacles_count=1,
        obstacles_radius_min=0.2, obstacles_radius_max=0.2,
    )
    env.reset(seed=0)
    # Place the obstacle at origin and the predator just outside on the +x side.
    env._obs_centers[0] = np.array([0.0, 0.0], dtype=np.float32)
    env._obs_radii[0] = 0.2
    env._pos[0] = np.array([0.25, 0.0], dtype=np.float32)
    # Action 1 = move LEFT. The predator should bump into the obstacle and stop at its perimeter.
    env.step({"adversary_0": 1})
    d = float(np.linalg.norm(env._pos[0] - env._obs_centers[0]))
    assert d >= 0.2 - 1e-3, f"Predator should be pushed to the obstacle perimeter, got d={d}"


def test_line_of_sight_blocked_by_obstacle():
    """A predator and a prey on opposite sides of an obstacle should not see each other."""
    env = make_ecosystem_env(
        seed=0, n_predators_start=1, n_prey_start=1,
        max_predators=1, max_prey=1,
        obstacles_count=1,
        obstacles_radius_min=0.15, obstacles_radius_max=0.15,
        pred_sense_radius=1.0, prey_sense_radius=1.0,
        max_neighbors_obs=4, genome_init_sigma=0.0,
    )
    env.reset(seed=0)
    env._obs_centers[0] = np.array([0.0, 0.0], dtype=np.float32)
    env._obs_radii[0] = 0.15
    env._pos[0] = np.array([-0.4, 0.0], dtype=np.float32)
    env._pos[1] = np.array([0.4, 0.0], dtype=np.float32)
    obs = env._build_obs_dict()
    # First neighbour slot's is_prey flag is at offset 6 + 5 = 11.
    assert obs["adversary_0"][11] == 0.0, "Predator should NOT see prey through the obstacle"


def test_line_of_sight_not_blocked_when_obstacle_is_off_axis():
    """Same setup but the obstacle is shifted off the line of sight — agents should see each other."""
    env = make_ecosystem_env(
        seed=0, n_predators_start=1, n_prey_start=1,
        max_predators=1, max_prey=1,
        obstacles_count=1,
        obstacles_radius_min=0.1, obstacles_radius_max=0.1,
        pred_sense_radius=1.0, prey_sense_radius=1.0,
        max_neighbors_obs=4, genome_init_sigma=0.0,
    )
    env.reset(seed=0)
    env._obs_centers[0] = np.array([0.0, 0.5], dtype=np.float32)  # well above the agents' line
    env._obs_radii[0] = 0.1
    env._pos[0] = np.array([-0.4, 0.0], dtype=np.float32)
    env._pos[1] = np.array([0.4, 0.0], dtype=np.float32)
    obs = env._build_obs_dict()
    assert obs["adversary_0"][11] == 1.0, "Predator should see prey when obstacle is off the LOS"


def test_obstacles_included_in_observation():
    env = make_ecosystem_env(
        seed=0, n_predators_start=1, n_prey_start=0,
        max_predators=1, max_prey=0,
        obstacles_count=3, obstacles_in_obs=3,
        obstacles_radius_min=0.12, obstacles_radius_max=0.12,
        genome_init_sigma=0.0,
    )
    env.reset(seed=0)
    env._pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    obs = env._build_obs_dict()
    obs_vec = obs["adversary_0"]
    # Layout: 6 own + 6*K neighbours + 1 food + 3 genome + 3*M obstacles
    M = 3
    K = env.cfg.max_neighbors_obs
    base = 6 + 6 * K + 1 + 3
    radii_slots = obs_vec[base + 2::3][:M]
    assert (radii_slots > 0).all(), "All obstacle radius slots should be populated"
    # Each pair (rel_x, rel_y) should equal the obstacle's centre minus agent's position (here origin).
    for slot in range(M):
        rel = obs_vec[base + 3 * slot : base + 3 * slot + 2]
        # The obstacles are sorted by distance from agent; we only check the slot is some real obstacle.
        matches = np.any(np.all(np.isclose(env._obs_centers, rel, atol=1e-4), axis=1))
        assert matches, f"Obs slot {slot} relative pos {rel} doesn't match any obstacle centre"


def test_obs_dim_grows_with_obstacles_in_obs():
    no_obs = make_ecosystem_env(seed=0, n_predators_start=1, n_prey_start=1)
    no_obs.reset(seed=0)
    base_dim = no_obs._obs_dim
    with_obs = make_ecosystem_env(seed=0, n_predators_start=1, n_prey_start=1,
                                   obstacles_count=4, obstacles_in_obs=4)
    with_obs.reset(seed=0)
    assert with_obs._obs_dim == base_dim + 4 * 3
