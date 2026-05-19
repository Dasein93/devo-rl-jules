# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Predator–prey digital evolution + RL sandbox. PPO trained on PettingZoo MPE `simple_tag_v3` (parallel API), with shared-policy multi-agent training, trajectory recording, and MP4 replays.

## Commands

```bash
pip install -r requirements.txt

# Train (entrypoint is run_train.py; the notebook references run_cpu.py which does not exist).
python run_train.py --config configs/base.yaml --episodes 50 --device cpu
python run_train.py --config configs/base.yaml --episodes 200 --device cuda

# Resume: --resume_from points at an existing artifacts/run_* dir; it auto-loads the
# newest checkpoints/*.pt by ctime and continues appending to metrics.csv.
python run_train.py --config configs/base.yaml --episodes 100 --resume_from artifacts/run_YYYYMMDD_HHMM

# Tests
pytest -q
pytest tests/test_recorder.py::test_recorder_ragged_obs -v

# Replay an entire traj/ dir or a single .npz. The input is a POSITIONAL arg
# (the README's `--in` example is wrong — argparse treats `in` as positional).
python tools/replay.py artifacts/run_*/traj --out video.mp4 --mode heatmap
python tools/replay.py artifacts/run_*/traj --out video.mp4 --mode positions
```

`tests/test_replay_positions.py` invokes `run_train.py` and `tools/replay.py` via `subprocess` against the real `configs/base.yaml`, so it requires ffmpeg and runs a short PettingZoo training — it's slow and the slowest part of the suite.

## Architecture

Single shared policy across all agents. `run_train.py` is the orchestrator; `train/ppo.py` holds PPO + the recorder; `tools/replay.py` is an offline visualizer.

**Observation handling.** Predators and prey have different observation sizes in `simple_tag_v3`. `flatten_obs()` in `train/ppo.py` sorts agent ids, ravels each obs vector, and right-pads them all to the max size so the shared MLP sees a uniform `(A, D)` matrix. The same padding happens again inside `TrajectoryRecorder._pad_and_stack` when saving. Anything that consumes raw obs must tolerate this padding.

**Env step compatibility.** `_step()` in `run_train.py` handles both the 4-tuple (`obs, rew, done, info`) and 5-tuple (`obs, rew, term, trunc, info`) PettingZoo return shapes and collapses `done_any = any(...)` across agents. An episode ends when *any* agent terminates/truncates.

**PPO buffer layout (subtle).** The training loop appends `obs`/`acts`/`logps`/`vals` per-(step, agent) but `rews`/`dones` per-step (using `mean(rewards.values())`). Before `ppo.update()`, the per-step arrays are expanded by `rep = n_total // n_steps` to match the per-agent length. If you change how rewards are aggregated or how many entries get appended per step, fix this expansion too or the `assert` in `PPO.update` will fire.

**Trajectory recording.** Enabled via `recording.enabled: true` in the config. `TrajectoryRecorder` writes both `ep_<i>.jsonl` (full per-step records) and `ep_<i>.npz` (`obs`, `act`, `agent_names`, and `pos` when available) per episode, plus a single `manifest.json` at the end (env_cfg, seeds, agent_names, agent_roles). Saved `obs` is shape `(T*A, D)` flat while `act` is `(T, A)` — known asymmetry, tests rely on `obs.shape[0] % act.shape[0] == 0`.

**Position extraction is env-specific.** `record_step_positions` only runs when `"simple_tag" in env_id` and reads `obs[2:4]` — the agent's `(x, y)` slot in the simple_tag observation layout. Other envs need their own extraction.

**Roles by name.** Predator vs prey is inferred from agent id substring: `"adversary" in name` → predator, else prey. Used in both `manifest.json` and `tools/replay.py` (red predators, green prey). simple_tag agent ids follow this convention; new envs would need a different scheme.

**Artifacts layout.** Each run lives in `artifacts/run_<UTC timestamp>/` with `checkpoints/ckpt_<ep>.pt`, `plots/return.png`, `traj/ep_*.{npz,jsonl}` + `manifest.json`, and `metrics.csv`. `.gitignore` excludes `artifacts/`, `*.mp4`, `*.npz`, `*.csv` — keep generated files out of commits.

## Config

`configs/base.yaml` is the single source of truth for hyperparameters. CLI flags (`--episodes`, `--device`, `--save_dir`, `--resume_from`) override the corresponding config values; everything else (PPO coefficients, env shape, plot/checkpoint cadence, recording) only changes via the YAML.
