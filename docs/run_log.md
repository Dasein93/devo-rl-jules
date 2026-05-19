# Run Log

| date (UTC) | episodes | pred return (last 10) | prey return (last 10) | config | notes |
|---|---:|---:|---:|---|---|
| 2025-08-26 | — | — | — | configs/base.yaml | Colab quickstart bootstrap |
| 2026-05-19 | 60 | +23.0 | -503.1 | configs/base.yaml | Smoke test of per-team PPO + league (snapshot_every=25) |
| 2026-05-19 | 150 | +87.0 | -262.0 | configs/base.yaml | Preview run; 6 league snaps/team; tournament Elo cycle: top pred = ep25, pred_ep150 vs prey_ep25 → 0 captures |
| 2026-05-19 | 50  | +38.0 | -178.6 | configs/base.yaml (centralized_critic: true) | MAPPO smoke; ~4× faster pred return at ep50 vs decentralised (+38 vs +10) |
| 2026-05-19 | 600 | TBD   | TBD    | configs/base.yaml (centralized_critic: true) | MAPPO long run with 20-snapshot league; intermediate: +145 pred @ ep350, oscillation suggests league pressure |
