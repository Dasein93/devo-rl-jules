"""Tests for MAPPO-style centralised critic."""
import numpy as np
import torch

from train.ppo import PPO, PPOConfig, ActorCritic


def test_actor_critic_state_dim_defaults_to_obs_dim():
    ac = ActorCritic(obs_dim=8, act_dim=3, hidden=16)
    assert ac.state_dim == 8


def test_actor_critic_centralised_critic_uses_state_dim():
    ac = ActorCritic(obs_dim=4, act_dim=3, hidden=16, state_dim=12)
    # critic accepts (B, 12); rejects (B, 4)
    out = ac.value(torch.zeros(2, 12))
    assert out.shape == (2,)


def test_actor_critic_act_only_does_not_touch_critic():
    """act() should work even if state_dim differs from obs_dim."""
    ac = ActorCritic(obs_dim=4, act_dim=3, hidden=16, state_dim=12)
    a, logp = ac.act(torch.zeros(2, 4))
    assert a.shape == (2,)
    assert logp.shape == (2,)


def test_ppo_save_load_preserves_state_dim(tmpdir):
    cfg = PPOConfig(hidden=16)
    ppo = PPO(obs_dim=4, act_dim=3, cfg=cfg, device="cpu", state_dim=12)
    path = str(tmpdir / "ckpt.pt")
    ppo.save(path, episode=1, returns=[0.0])

    ckpt = torch.load(path, weights_only=False)
    assert ckpt["state_dim"] == 12

    # Reload into a fresh PPO with same state_dim
    ppo2 = PPO(obs_dim=4, act_dim=3, cfg=cfg, device="cpu", state_dim=12)
    ppo2.load(path)
    # Weights match
    for p1, p2 in zip(ppo.ac.parameters(), ppo2.ac.parameters()):
        assert torch.allclose(p1, p2)


def test_ppo_update_with_centralised_states():
    torch.manual_seed(0); np.random.seed(0)
    obs_dim, state_dim, act_dim = 4, 12, 3
    cfg = PPOConfig(lr=1e-3, update_epochs=1, minibatch_size=8, hidden=16)
    ppo = PPO(obs_dim, act_dim, cfg, device="cpu", state_dim=state_dim)

    n = 16
    obs = np.random.randn(n, obs_dim).astype(np.float32)
    states = np.random.randn(n, state_dim).astype(np.float32)
    acts = np.random.randint(0, act_dim, size=n).tolist()
    with torch.no_grad():
        a, logp = ppo.ac.act(torch.from_numpy(obs))
        v = ppo.ac.value(torch.from_numpy(states))
    logps = logp.numpy().tolist()
    vals = v.numpy().tolist()
    rews = np.random.randn(n).tolist()
    dones = [0.0] * (n - 1) + [1.0]

    stats = ppo.update(obs.tolist(), acts, logps, rews, dones, vals, states=states.tolist())
    for key in ("pg_loss", "v_loss", "entropy"):
        assert np.isfinite(stats[key])
    assert stats["samples"] == n


def test_ppo_update_states_length_mismatch_raises():
    cfg = PPOConfig(hidden=16)
    ppo = PPO(4, 3, cfg, device="cpu", state_dim=12)
    # 16 obs but only 8 states → assertion
    obs = np.random.randn(16, 4).astype(np.float32)
    states = np.random.randn(8, 12).astype(np.float32)
    try:
        ppo.update(obs.tolist(), [0]*16, [0.0]*16, [0.0]*16, [0.0]*16, [0.0]*16, states=states.tolist())
    except AssertionError:
        return
    raise AssertionError("expected length-mismatch AssertionError")
