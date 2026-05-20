"""Tests for the GRU-based RecurrentActorCritic and PPO.update_recurrent."""
import numpy as np
import pytest
import torch

from train.ppo import PPO, PPOConfig, RecurrentActorCritic


def test_recurrent_ac_init_hidden_shape():
    ac = RecurrentActorCritic(obs_dim=8, act_dim=5, hidden=32)
    h = ac.init_hidden(batch_size=3)
    assert h.shape == (1, 3, 32)


def test_recurrent_ac_forward_step_shapes():
    ac = RecurrentActorCritic(obs_dim=8, act_dim=5, hidden=32)
    obs = torch.zeros(3, 8)
    h = ac.init_hidden(3)
    logits, v, h_new = ac.forward_step(obs, h)
    assert logits.shape == (3, 5)
    assert v.shape == (3,)
    assert h_new.shape == (1, 3, 32)


def test_recurrent_ac_forward_seq_shapes():
    ac = RecurrentActorCritic(obs_dim=4, act_dim=3, hidden=16)
    obs_seq = torch.zeros(2, 7, 4)
    h0 = ac.init_hidden(2)
    logits, v, h_final = ac.forward_seq(obs_seq, h0)
    assert logits.shape == (2, 7, 3)
    assert v.shape == (2, 7)
    assert h_final.shape == (1, 2, 16)


def test_recurrent_ac_step_samples_action_and_advances_hidden():
    ac = RecurrentActorCritic(obs_dim=4, act_dim=5, hidden=16)
    obs = torch.randn(2, 4)
    a, logp, v, h_new = ac.step(obs)
    assert a.shape == (2,)
    assert logp.shape == (2,)
    assert v.shape == (2,)
    assert h_new.shape == (1, 2, 16)
    assert int(a.max()) < 5 and int(a.min()) >= 0


def test_recurrent_ac_act_returns_hidden_without_v():
    ac = RecurrentActorCritic(obs_dim=4, act_dim=3, hidden=8)
    obs = torch.zeros(1, 4)
    a, logp, h_new = ac.act(obs)
    assert a.shape == (1,)
    assert h_new.shape == (1, 1, 8)


def test_ppo_recurrent_flag_uses_recurrent_ac():
    cfg = PPOConfig(hidden=16)
    ppo = PPO(obs_dim=4, act_dim=3, cfg=cfg, device="cpu", recurrent=True)
    assert isinstance(ppo.ac, RecurrentActorCritic)


def test_ppo_recurrent_rejects_state_dim():
    cfg = PPOConfig(hidden=16)
    with pytest.raises(ValueError, match="recurrent"):
        PPO(obs_dim=4, act_dim=3, cfg=cfg, device="cpu", recurrent=True, state_dim=12)


def test_ppo_update_recurrent_runs_and_changes_weights():
    torch.manual_seed(0); np.random.seed(0)
    cfg = PPOConfig(lr=1e-3, update_epochs=1, hidden=16)
    ppo = PPO(obs_dim=4, act_dim=3, cfg=cfg, device="cpu", recurrent=True)
    w0 = next(ppo.ac.parameters()).clone().detach()

    sequences = []
    for _ in range(3):
        T = 8
        obs = np.random.randn(T, 4).astype(np.float32)
        acts = np.random.randint(0, 3, size=T).tolist()
        rng_vals = np.random.randn(T).tolist()
        sequences.append({
            "obs": obs,
            "acts": acts,
            "logps": rng_vals,
            "vals": rng_vals,
            "advs": np.random.randn(T).tolist(),
            "rets": np.random.randn(T).tolist(),
        })

    stats = ppo.update_recurrent(sequences)
    assert stats["samples"] == 3 * 8
    for k in ("pg_loss", "v_loss", "entropy"):
        assert np.isfinite(stats[k])

    w1 = next(ppo.ac.parameters())
    assert not torch.allclose(w0, w1), "Update should have moved the weights"


def test_ppo_update_recurrent_rejects_on_non_recurrent():
    cfg = PPOConfig(hidden=8)
    ppo = PPO(obs_dim=4, act_dim=3, cfg=cfg, device="cpu", recurrent=False)
    with pytest.raises(RuntimeError, match="non-recurrent"):
        ppo.update_recurrent([])


def test_ppo_save_load_recurrent_roundtrip(tmpdir):
    cfg = PPOConfig(hidden=16)
    ppo = PPO(obs_dim=4, act_dim=3, cfg=cfg, device="cpu", recurrent=True)
    path = str(tmpdir / "rec.pt")
    ppo.save(path, episode=5, returns=[0.1, 0.2])

    ppo2 = PPO(obs_dim=4, act_dim=3, cfg=cfg, device="cpu", recurrent=True)
    ep, rets = ppo2.load(path)
    assert ep == 5
    assert rets == [0.1, 0.2]
    for p1, p2 in zip(ppo.ac.parameters(), ppo2.ac.parameters()):
        assert torch.allclose(p1, p2)
