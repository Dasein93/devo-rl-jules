"""Phase 3 ecosystem tests: heritable genome + mutation on reproduction."""
import numpy as np

from train.ecosystem_env import make_ecosystem_env


def test_genome_initialised_near_one():
    env = make_ecosystem_env(seed=0, n_predators_start=5, n_prey_start=5, genome_init_sigma=0.1)
    env.reset(seed=0)
    # All initialised genes within a few sigma of 1.0.
    assert np.all(env._genome > 0.5)
    assert np.all(env._genome < 1.5)


def test_genome_disabled_means_all_ones():
    env = make_ecosystem_env(seed=0, n_predators_start=2, n_prey_start=2, genome_enabled=False)
    env.reset(seed=0)
    assert np.allclose(env._genome, 1.0)


def test_child_inherits_parent_genome_with_mutation():
    """Spawn a child and verify its genome is parent's + small Gaussian noise."""
    env = make_ecosystem_env(
        seed=0, n_predators_start=1, n_prey_start=0,
        max_predators=2, max_prey=0,
        repro_pred_energy_thresh=10.0, repro_min_age=1, repro_cooldown=0,
        pred_energy_cost=0.0, attack_range=0.0,
        mutation_sigma=0.05, genome_min=0.0, genome_max=10.0,
    )
    env.reset(seed=0)
    # Force a specific parent genome.
    parent_genome = np.array([1.2, 0.9, 1.3], dtype=np.float32)
    env._genome[0] = parent_genome
    env._energy[0] = 100.0
    env._age[0] = 100
    env.step({"adversary_0": 0})
    assert env._alive[1], "Child should have been spawned"
    child_genome = env._genome[1]
    # Child differs from parent by something within a reasonable mutation range.
    diff = child_genome - parent_genome
    assert np.all(np.abs(diff) < 0.5), f"Mutation should be small: {diff}"
    # Mutation is non-zero with overwhelming probability — at least one trait
    # should differ from the parent.
    assert not np.allclose(diff, 0.0), "Child genome should differ from parent"


def test_speed_gene_scales_movement():
    """An agent with speed_g=2.0 should move twice as fast as one with speed_g=1.0."""
    env = make_ecosystem_env(
        seed=0, n_predators_start=2, n_prey_start=0,
        max_predators=2, max_prey=0,
        attack_range=0.0, pred_energy_cost=0.0,
    )
    env.reset(seed=0)
    env._pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    env._pos[1] = np.array([0.0, 0.0], dtype=np.float32)
    env._genome[0] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    env._genome[1] = np.array([2.0, 1.0, 1.0], dtype=np.float32)
    # Both take action 2 (move right).
    env.step({"adversary_0": 2, "adversary_1": 2})
    moved_slow = env._pos[0, 0]
    moved_fast = env._pos[1, 0]
    assert moved_fast > moved_slow * 1.5, f"Fast agent should move much further: slow={moved_slow}, fast={moved_fast}"


def test_hp_gene_scales_starting_hp():
    """An agent with hp_g=2.0 should start with twice the HP of one with hp_g=1.0."""
    env = make_ecosystem_env(
        seed=0, n_predators_start=2, n_prey_start=0,
        max_predators=2, max_prey=0,
        pred_max_hp=100.0, genome_init_sigma=0.0,
    )
    env.reset(seed=0)
    env._genome[0] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    env._genome[1] = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    # Re-apply HP based on genome (reset already did, but with genome=1 default).
    env._hp[0] = env._hp_cap(0)
    env._hp[1] = env._hp_cap(1)
    assert env._hp[1] == 2.0 * env._hp[0]


def test_sense_gene_scales_neighbour_visibility():
    """An agent with sense_g=2.0 should see neighbours at twice the radius."""
    env_narrow = make_ecosystem_env(
        seed=0, n_predators_start=1, n_prey_start=1,
        max_predators=1, max_prey=1,
        pred_sense_radius=0.3, max_neighbors_obs=4, genome_init_sigma=0.0,
    )
    env_narrow.reset(seed=0)
    env_narrow._pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    env_narrow._pos[1] = np.array([0.4, 0.0], dtype=np.float32)
    env_narrow._genome[0] = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # narrow
    obs_narrow = env_narrow._build_obs_dict()

    env_wide = make_ecosystem_env(
        seed=0, n_predators_start=1, n_prey_start=1,
        max_predators=1, max_prey=1,
        pred_sense_radius=0.3, max_neighbors_obs=4, genome_init_sigma=0.0,
    )
    env_wide.reset(seed=0)
    env_wide._pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    env_wide._pos[1] = np.array([0.4, 0.0], dtype=np.float32)
    env_wide._genome[0] = np.array([1.0, 1.0, 2.0], dtype=np.float32)    # wide
    obs_wide = env_wide._build_obs_dict()

    # The first neighbour slot's is_prey flag is at offset 6 + 5 = 11.
    assert obs_narrow["adversary_0"][6 + 5] == 0.0, "Narrow-sensed predator should not see prey at 0.4"
    assert obs_wide["adversary_0"][6 + 5] == 1.0, "Wide-sensed predator should see prey at 0.4"


def test_mean_genome_helper_returns_per_team_means():
    env = make_ecosystem_env(seed=0, n_predators_start=3, n_prey_start=2, genome_init_sigma=0.0)
    env.reset(seed=0)
    env._genome[0:3] = np.array([[1.5, 1.0, 1.0], [1.0, 1.0, 1.0], [0.5, 1.0, 1.0]], dtype=np.float32)
    env._genome[env._max_pred:env._max_pred + 2] = np.array([[1.0, 1.0, 2.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    pred_mean = env.mean_genome("predator")
    prey_mean = env.mean_genome("prey")
    np.testing.assert_allclose(pred_mean, [1.0, 1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(prey_mean, [1.0, 1.0, 1.5], atol=1e-6)


def test_genome_in_obs_last_three_dims():
    env = make_ecosystem_env(seed=0, n_predators_start=1, n_prey_start=0, max_predators=1, max_prey=0)
    env.reset(seed=0)
    env._genome[0] = np.array([1.23, 0.77, 1.55], dtype=np.float32)
    obs = env._build_obs_dict()
    np.testing.assert_allclose(obs["adversary_0"][-3:], [1.23, 0.77, 1.55], atol=1e-5)
