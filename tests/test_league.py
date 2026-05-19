import os
import numpy as np
import torch

from train.ppo import ActorCritic
from train.league import League


def test_snapshot_persists_and_sampling_returns_frozen_actor(tmpdir):
    save_dir = str(tmpdir / "lg")
    league = League(save_dir, team="predator", snapshot_every=1, opponent_prob=1.0,
                    max_snapshots=5, device="cpu", rng=np.random.default_rng(0))
    assert len(league) == 0
    assert league.sample() is None  # empty pool

    ac = ActorCritic(obs_dim=4, act_dim=3, hidden=16)
    p = league.snapshot(ac, episode=1)
    assert os.path.isfile(p)
    assert len(league) == 1

    sampled = league.sample()
    assert sampled is not None
    # Returned actor is in eval mode and frozen.
    assert not sampled.training
    for param in sampled.parameters():
        assert not param.requires_grad


def test_snapshot_eviction(tmpdir):
    save_dir = str(tmpdir / "lg")
    league = League(save_dir, team="prey", snapshot_every=1, opponent_prob=0.0,
                    max_snapshots=3, device="cpu", rng=np.random.default_rng(0))
    ac = ActorCritic(obs_dim=4, act_dim=3, hidden=16)
    for ep in range(1, 6):
        league.snapshot(ac, episode=ep)
    assert len(league) == 3
    # The earliest two should have been deleted.
    remaining = sorted(os.listdir(save_dir))
    assert remaining == ["snap_000003.pt", "snap_000004.pt", "snap_000005.pt"]


def test_opponent_prob_zero_never_samples(tmpdir):
    save_dir = str(tmpdir / "lg")
    league = League(save_dir, team="prey", snapshot_every=1, opponent_prob=0.0,
                    max_snapshots=5, device="cpu", rng=np.random.default_rng(0))
    ac = ActorCritic(obs_dim=4, act_dim=3, hidden=16)
    league.snapshot(ac, episode=1)
    for _ in range(20):
        assert league.sample() is None


def test_maybe_snapshot_respects_cadence(tmpdir):
    save_dir = str(tmpdir / "lg")
    league = League(save_dir, team="predator", snapshot_every=3, opponent_prob=0.0,
                    max_snapshots=5, device="cpu", rng=np.random.default_rng(0))
    ac = ActorCritic(obs_dim=4, act_dim=3, hidden=16)
    assert league.maybe_snapshot(ac, episode=1) is None
    assert league.maybe_snapshot(ac, episode=2) is None
    assert league.maybe_snapshot(ac, episode=3) is not None
    assert league.maybe_snapshot(ac, episode=6) is not None
    assert len(league) == 2
