# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Predator–prey co-evolution sandbox on PettingZoo MPE `simple_tag_v3`:

- **Per-team PPO** (`train/ppo.py`) — one policy per team, separate optimizers/replay.
- **Optional MAPPO-style centralised critic** (`train.centralized_critic` flag) —
  each team's critic sees the concatenation of all its own agents' observations.
- **League of frozen snapshots** (`train/league.py`) — periodic snapshots of each
  team's policy are sampled as opponents during rollouts to stabilise co-adaptation.
- **GAE** with value-loss clipping and gradient clipping.
- **Trajectory recording + replay** to MP4 (heatmap, positions+trails, or ecosystem split-panel).
- **Two envs**: `mpe.simple_tag_v3` (PettingZoo) and `ecosystem` (`train/ecosystem_env.py`).
  Choice driven by `env.id` in the config.

## Commands

```bash
pip install -r requirements.txt

# Train (entrypoint is run_train.py).
python run_train.py --config configs/base.yaml --episodes 200 --device cpu
python run_train.py --config configs/base.yaml --episodes 1000 --device cuda

# Resume: --resume_from points at an existing artifacts/run_* dir; it auto-loads
# the newest pred_*.pt and prey_*.pt by ctime and appends to metrics.csv.
python run_train.py --config configs/base.yaml --episodes 2000 --resume_from artifacts/run_YYYYMMDD_HHMMSS

# Evaluate (deterministic by default). Glob is expanded; latest match wins.
python tools/eval.py \
  --pred 'artifacts/run_*/checkpoints/pred_*.pt' \
  --prey 'artifacts/run_*/checkpoints/prey_*.pt' \
  --episodes 20
# Either side can be omitted to use a random-init baseline.

# Tournament: round-robin between all league snapshots → captures heatmap + Elo.
python tools/tournament.py --run artifacts/run_YYYYMMDD_HHMMSS --episodes 3

# Compare ablation runs (overlay metric across multiple runs).
python tools/compare_runs.py --runs artifacts/run_A artifacts/run_B \
  --labels A B --metric pred_return --out compare.png

# Ecosystem-specific diagnostics (need genome columns / pop columns in metrics.csv).
python tools/genome_trace.py --run artifacts/run_YYYYMMDD_HHMMSS --out genome.png
python tools/phase_portrait.py --run artifacts/run_YYYYMMDD_HHMMSS --out phase.png

# Tests — use `python -m pytest`, not `pytest`, because the system-wide `pytest`
# is a separate uv-managed install that does not see the project requirements.
python -m pytest -q
python -m pytest tests/test_ppo.py::test_gae_terminal_zeros_bootstrap -v

# Train on the custom ecosystem env (mortality, reproduction, food field).
python run_train.py --config configs/ecosystem.yaml --episodes 300 --device cpu

# Replay. `in` is a POSITIONAL argument; do NOT pass it as --in.
# Multi-episode directories are sorted by integer ep index, not alphabetically.
python tools/replay.py artifacts/run_*/traj --out video.mp4 --mode heatmap
python tools/replay.py artifacts/run_*/traj --out video.mp4 --mode positions --frameskip 2 --trail 20
# Ecosystem split panel (2D scene + population-over-time curve, agents wink in/out):
python tools/replay.py artifacts/run_*/traj --out video.mp4 --mode ecosystem --frameskip 2 --trail 20
```

`tests/test_replay_positions.py` invokes `run_train.py` and `tools/replay.py` via
`subprocess` against the real `configs/base.yaml`, so it requires `ffmpeg` (from
`imageio[ffmpeg]`) and is the slow test (~30 s). Skip with `-k 'not replay'` if iterating.

## Architecture

**Per-team policies.** `run_train.py` uses `train.ppo.split_teams` to partition
agents into predators (`"adversary" in name`) and prey, instantiates one PPO per
team with potentially different `obs_dim`, and runs them in parallel. Per-step
rewards are aggregated as the per-team mean and replicated across that team's
agents before going into the buffer; only the team's own samples update its
policy.

**Centralised critic (MAPPO).** Toggled by `train.centralized_critic: true`.
`ActorCritic.__init__` takes a `state_dim` parameter; when set, the critic
takes a flattened concatenation of all team agents' observations (shape
`team_size * pad_obs_dim`) instead of per-agent obs. The actor is unchanged.
During rollouts the trainer builds the team state via `_team_state(team_obs)`
and passes it to `_act_team(..., state=...)`, which returns a *shared* value
replicated across the team's agents. The buffer carries a `states` list
alongside `obs`; `PPO.update(..., states=...)` uses it to recompute v_pred.
**Opponents (league snapshots) are routed through `_act_team_actor_only`**
because their saved critic may have a different `state_dim` than the live
trainer — calling `.step()` on them would crash with a shape mismatch.
Checkpoints persist `state_dim`; `ActorCritic` and `League._load` honour it.

**League / "digital evolution".** `train/league.py` holds a FIFO pool of
`ActorCritic` state dicts on disk under `league/{predator,prey}/snap_*.pt`. At
each episode start, each league is asked for a sample: with probability
`opponent_prob` it returns a randomly chosen frozen actor (eval mode, no grad).
That team then plays the episode under the snapshot and its transitions are NOT
collected — only the live policy of the other team is updated. Snapshots are
pushed every `snapshot_every` episodes (cadence applies to the *episode index*,
not the number of trained episodes). Eviction is oldest-first past
`max_snapshots`.

**Env step compatibility.** `_step()` in `run_train.py` handles both the 4-tuple
(`obs, rew, done, info`) and 5-tuple (`obs, rew, term, trunc, info`) PettingZoo
return shapes and collapses `done_any = any(...)` across agents. An episode ends
when *any* agent terminates/truncates.

**PPO update API.** There are two entry points:
- `PPO.update(obs, acts, logps, rews, dones, vals, states=None)` — computes
  GAE from the supplied `(rews, dones, vals)` and then runs the update. Use for
  a single contiguous rollout sequence.
- `PPO.update_precomputed(obs, acts, logps, vals, advs, rets, states=None)` —
  caller supplies the advantages and returns. Use when rollouts span multiple
  independent sequences (e.g. per-agent buffers) that must each be GAE'd
  separately to avoid bleed across boundaries.

**GAE.** `PPO._gae` is a pure static function — easy to unit-test (see
`tests/test_ppo.py`). It bootstraps with `V=0` past the buffer end, so the last
timestep of each rollout effectively assumes terminal; combined with the
per-step `done` mask this gives the standard GAE behaviour for episodic
rollouts.

**Per-agent rollouts.** `run_train.py` collects each agent's transitions into
its own dict in `pred_buf` / `prey_buf` (`{agent_name: {"obs": [...], ...}}`),
not into a single flat list. Before the PPO update, `_update_per_agent`
computes GAE separately on each agent's sequence, then concatenates the
results and calls `PPO.update_precomputed`. This avoids the previous
GAE-bleed bug where the interleaved per-(time, agent) layout caused
within-step deltas to mix values across agents (especially degenerate with
MAPPO's shared per-step values). Per-agent GAE produces ~3-4× faster early
learning at fixed seed compared to the interleaved layout.

**Trajectory recording.** `TrajectoryRecorder` writes per-episode
`ep_<i>.{jsonl,npz}` and a run-level `manifest.json`. NPZ layout is now
`obs (T, A, D)`, `act (T, A)`, optional `pos (T, A, 2)`, plus `agent_names (A,)`
— consistent across keys (previous flat `(T*A, D)` layout for `obs` was changed
along with `tests/test_recorder.py`).

**Position extraction is env-specific.** `record_step` only emits `pos` when
`"simple_tag" in env_id` and reads `obs[2:4]` — the agent's `(x, y)` slot in
simple_tag's observation layout. Other envs would need their own extractor.

**Roles by name.** Predator vs prey is inferred from agent id substring:
`"adversary" in name` → predator, else prey. simple_tag agent ids follow this
convention; new envs would need a different scheme (and `split_teams` would
need to be generalised).

**Ecosystem env (`train/ecosystem_env.py`).** Custom PettingZoo-parallel-API
env. Per-agent HP, energy, age, reproduction cooldown. Predators eat prey
(prey HP -= damage, predator energy += food); both teams starve at energy 0
and die at HP 0. Prey forage from a coarse food grid (`food_grid_size`)
that regenerates each step. Reproduction triggers on (age > min, cooldown
== 0, energy ≥ team threshold) and spawns a child in a free slot. The
roster has `max_predators + max_prey` slots total; `n_*_start` agents are
alive at reset, the rest fill in via births. Agent names follow the
simple_tag convention (`adversary_*`, `agent_*`) so `split_teams`, league,
tournament, and replay code all work unchanged. The trainer uses
`env.possible_agents` to know the full slot roster; `done_any` in
`_step()` is now driven by `truncations` (episode-level) rather than
`terminations` (per-agent death) — this distinction is what lets the
ecosystem env have agents die mid-episode without ending the episode.

**Artifacts layout.**

```
artifacts/run_<UTC timestamp>/
├── checkpoints/{pred,prey}_<ep>.pt     # ActorCritic + Adam state, dims
├── league/{predator,prey}/snap_*.pt    # frozen snapshots
├── plots/return.png                    # per-team return curves
├── traj/ep_*.{npz,jsonl} + manifest.json
│                                       # NPZ now also has `alive (T,A) bool`;
│                                       # ecosystem episodes include `pos (T,A,2)`
│                                       # from obs[0:2], with NaN for dead slots.
├── metrics.csv                         # ep, returns, captures, ep_steps,
│                                       # pred_pop_end/_mean, prey_pop_end/_mean,
│                                       # losses, league sizes
└── tournament/                         # written by tools/tournament.py
    ├── scores.csv                      # pair-wise captures, episode length, returns
    ├── ratings.csv                     # Elo per snapshot per team
    └── heatmap.png
```

**Tournament / "evolutionary diagnostic".** `tools/tournament.py` loads every
`league/{predator,prey}/snap_*.pt` and runs each (pred_i, prey_j) pair for K
episodes. `_elo()` does a few sweeps of standard Elo with score = 1 if mean
captures > 0, 0.5 if 0, else 0. Asymmetric (predators and prey have separate
rating scales). Top Elo snapshots are often NOT the latest snapshot — that's
the diagnostic signal: cycling means later snapshots got specialised to
counter the league pool, not absolute strength.

**Captures metric.** Logged in `metrics.csv` as `captures` per episode and
computed in two places: (1) in `run_train.py`'s inner loop, by counting per-
step prey rewards ≤ -10 (simple_tag awards -10 to prey / +10 to predator per
collision), and (2) in `tools/tournament.py` the same way. Both rely on the
simple_tag reward convention; new envs would need a different detector.

`.gitignore` excludes `artifacts/`, `*.mp4`, `*.npz`, `*.csv` — keep generated
files out of commits.

## Config

`configs/base.yaml` is the single source of truth for hyperparameters. CLI flags
(`--episodes`, `--device`, `--save_dir`, `--resume_from`) override the
corresponding config values; everything else (PPO coefficients, env shape,
plot/checkpoint cadence, league, recording) only changes via the YAML.

## CI

`.github/workflows/ci.yml` runs `pytest -q` on push to `main` and on PRs against
Python 3.11. The replay test is included, so the workflow requires
`imageio[ffmpeg]` from `requirements.txt` (already present).
