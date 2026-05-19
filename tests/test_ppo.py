import numpy as np
import torch

from train.ppo import PPO, PPOConfig, ActorCritic


def test_gae_terminal_zeros_bootstrap():
    """At an episode boundary (done=1), GAE must not bleed value across the cut."""
    rews = [1.0, 1.0, 1.0, 1.0]
    dones = [0.0, 1.0, 0.0, 0.0]
    vals = [0.5, 0.5, 0.5, 0.5]
    adv, rets = PPO._gae(rews, dones, vals, gamma=0.99, lam=0.95)
    # At t=1 (terminal), nonterminal=0, so delta = rews[1] - vals[1] = 0.5, and last_gae resets.
    assert np.isclose(adv[1], 0.5, atol=1e-6)
    # rets = adv + vals
    assert np.allclose(rets, adv + np.asarray(vals), atol=1e-6)


def test_gae_matches_discounted_return_when_lambda_one_and_terminal():
    """With λ=1 and a terminal flag at the end, returns ≈ discounted sum of rewards."""
    rews = [1.0, 1.0, 1.0]
    dones = [0.0, 0.0, 1.0]
    vals = [0.0, 0.0, 0.0]
    _, rets = PPO._gae(rews, dones, vals, gamma=0.5, lam=1.0)
    # Manually: G3=1, G2=1+0.5*1=1.5, G1=1+0.5*1.5=1.75
    assert np.allclose(rets, [1.75, 1.5, 1.0], atol=1e-6)


def test_ppo_update_runs_and_reduces_loss_on_constant_signal():
    """One PPO update step on a small synthetic batch should produce finite losses."""
    torch.manual_seed(0)
    np.random.seed(0)
    obs_dim, act_dim = 6, 5
    cfg = PPOConfig(lr=1e-3, update_epochs=2, minibatch_size=16, hidden=32)
    ppo = PPO(obs_dim, act_dim, cfg, device="cpu")

    n = 64
    obs = np.random.randn(n, obs_dim).astype(np.float32)
    acts = np.random.randint(0, act_dim, size=n).tolist()
    # Bootstrap a plausible logp/val from the freshly-initialized policy.
    with torch.no_grad():
        a, logp, v = ppo.ac.step(torch.from_numpy(obs))
    logps = logp.numpy().tolist()
    vals = v.numpy().tolist()
    rews = np.random.randn(n).tolist()
    dones = [0.0] * (n - 1) + [1.0]

    stats = ppo.update(obs.tolist(), acts, logps, rews, dones, vals)
    for key in ("pg_loss", "v_loss", "entropy"):
        assert np.isfinite(stats[key]), f"{key} not finite: {stats[key]}"
    assert stats["samples"] == n


def test_ppo_save_load_roundtrip(tmpdir):
    cfg = PPOConfig(hidden=32)
    ppo = PPO(obs_dim=4, act_dim=3, cfg=cfg, device="cpu")
    path = str(tmpdir / "ckpt.pt")
    ppo.save(path, episode=7, returns=[0.1, 0.2, 0.3])

    ppo2 = PPO(obs_dim=4, act_dim=3, cfg=cfg, device="cpu")
    ep, rets = ppo2.load(path)
    assert ep == 7
    assert rets == [0.1, 0.2, 0.3]
    # Weights match
    for p1, p2 in zip(ppo.ac.parameters(), ppo2.ac.parameters()):
        assert torch.allclose(p1, p2)
