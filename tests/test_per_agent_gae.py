"""Tests for per-agent rollout GAE and update_precomputed."""
import numpy as np
import torch

from train.ppo import PPO, PPOConfig


def test_update_precomputed_matches_update_for_single_sequence():
    """For a single sequence (no agent boundary issues), update_precomputed with
    GAE-computed advantages should produce the same outcome as update."""
    cfg = PPOConfig(lr=1e-3, update_epochs=1, minibatch_size=16, hidden=16)
    obs_dim, act_dim = 4, 3

    rng = np.random.default_rng(0)
    n = 32
    obs = rng.standard_normal((n, obs_dim)).astype(np.float32).tolist()
    acts = rng.integers(0, act_dim, size=n).tolist()
    rews = rng.standard_normal(n).tolist()
    dones = [0.0] * (n - 1) + [1.0]
    vals = rng.standard_normal(n).tolist()
    logps = rng.standard_normal(n).tolist()

    torch.manual_seed(7)
    ppo1 = PPO(obs_dim, act_dim, cfg, device="cpu")
    sd = {k: v.clone() for k, v in ppo1.ac.state_dict().items()}

    torch.manual_seed(123); np.random.seed(123)
    stats1 = ppo1.update(obs, acts, logps, rews, dones, vals)

    ppo2 = PPO(obs_dim, act_dim, cfg, device="cpu")
    ppo2.ac.load_state_dict(sd)
    adv, rets = PPO._gae(rews, dones, vals, cfg.gamma, cfg.gae_lambda)
    torch.manual_seed(123); np.random.seed(123)
    stats2 = ppo2.update_precomputed(obs, acts, logps, vals, adv.tolist(), rets.tolist())

    # update() = compute GAE + update_precomputed, so they must agree numerically.
    assert abs(stats1["pg_loss"] - stats2["pg_loss"]) < 1e-5
    assert abs(stats1["v_loss"] - stats2["v_loss"]) < 1e-5
    assert abs(stats1["entropy"] - stats2["entropy"]) < 1e-5


def test_per_agent_gae_no_bleed_across_agents():
    """Concatenating two independent sequences and running _gae over the
    concatenation should give the same advantages as computing _gae per sequence
    — provided each sequence ends with done=1."""
    rews_a = [1.0, 1.0, 1.0]
    dones_a = [0.0, 0.0, 1.0]
    vals_a = [0.5, 0.5, 0.5]
    rews_b = [2.0, 2.0]
    dones_b = [0.0, 1.0]
    vals_b = [0.3, 0.3]

    adv_a, rets_a = PPO._gae(rews_a, dones_a, vals_a, gamma=0.99, lam=0.95)
    adv_b, rets_b = PPO._gae(rews_b, dones_b, vals_b, gamma=0.99, lam=0.95)

    cat_rews = rews_a + rews_b
    cat_dones = dones_a + dones_b
    cat_vals = vals_a + vals_b
    adv_cat, rets_cat = PPO._gae(cat_rews, cat_dones, cat_vals, gamma=0.99, lam=0.95)

    expected_adv = np.concatenate([adv_a, adv_b])
    expected_rets = np.concatenate([rets_a, rets_b])

    # With dones=[0,0,1,0,1], the terminal at the end of sequence A correctly
    # resets last_gae, so concatenated GAE equals separate GAEs.
    np.testing.assert_allclose(adv_cat, expected_adv, atol=1e-6)
    np.testing.assert_allclose(rets_cat, expected_rets, atol=1e-6)


def test_per_agent_gae_avoids_interleaved_contamination():
    """If two agents' transitions are *interleaved* row-by-row within a timestep,
    GAE deltas mix values across agents. Verify the per-sequence path avoids that."""
    # 2 agents, 3 timesteps, shared reward = 1.0 per step, episode ends at t=2.
    # Per-agent: agent A vals are [0, 0, 0]; agent B vals are [10, 10, 10].
    rews_a = [1.0, 1.0, 1.0]
    dones_a = [0.0, 0.0, 1.0]
    vals_a = [0.0, 0.0, 0.0]
    adv_a_only, _ = PPO._gae(rews_a, dones_a, vals_a, gamma=0.99, lam=0.95)

    # Interleaved layout: [t0_A, t0_B, t1_A, t1_B, t2_A, t2_B]
    interleaved_rews = [1.0] * 6
    interleaved_dones = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    interleaved_vals = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0]
    adv_inter, _ = PPO._gae(interleaved_rews, interleaved_dones, interleaved_vals,
                            gamma=0.99, lam=0.95)
    # The interleaved GAE for agent A's rows (0, 2, 4) is contaminated by agent B's
    # values at neighbouring rows — it does NOT match the clean per-agent advantage.
    assert not np.allclose(adv_inter[::2], adv_a_only, atol=1e-3), \
        "Interleaved GAE happened to match per-agent — test setup is degenerate"
