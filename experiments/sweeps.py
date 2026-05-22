"""Run an OFAT (one-factor-at-a-time) experiment sweep over training/env specs.

Each cell is a small YAML override patch applied to a base config. We emit one
run per cell into artifacts/experiments/<exp_id>/, then analyse.py aggregates
the metrics.csv files across runs.

Goal: identify which specs matter most for learning quality in the simple_tag
and ecosystem envs.
"""
from __future__ import annotations
import argparse, copy, json, os, subprocess, sys, time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import yaml


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_ROOT = os.path.join(REPO, "artifacts", "experiments")
CFG_ROOT = os.path.join(EXP_ROOT, "_configs")


@dataclass
class Cell:
    exp_id: str
    base: str               # path to base yaml (simple_tag or ecosystem)
    episodes: int
    overrides: Dict[str, Any] = field(default_factory=dict)
    group: str = ""         # which sweep group this cell belongs to
    notes: str = ""


def _deep_merge(dst: dict, src: dict) -> dict:
    """Merge src into dst in-place, recursing into sub-dicts."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _materialise_config(cell: Cell) -> str:
    """Write the patched YAML to disk and return its path."""
    with open(cell.base, "r") as f:
        cfg = yaml.safe_load(f)
    _deep_merge(cfg, cell.overrides)
    os.makedirs(CFG_ROOT, exist_ok=True)
    out = os.path.join(CFG_ROOT, f"{cell.exp_id}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def _run_dir(exp_id: str) -> str:
    return os.path.join(EXP_ROOT, exp_id)


def run_cell(cell: Cell, device: str = "cpu", dry_run: bool = False) -> Dict[str, Any]:
    """Execute one cell. Returns a record with wall-clock and exit status."""
    cfg_path = _materialise_config(cell)
    out_dir = _run_dir(cell.exp_id)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, "run_train.py",
        "--config", cfg_path,
        "--episodes", str(cell.episodes),
        "--save_dir", out_dir,
        "--device", device,
    ]
    rec = {
        "exp_id": cell.exp_id, "group": cell.group, "base": os.path.basename(cell.base),
        "episodes": cell.episodes, "overrides": cell.overrides, "notes": cell.notes,
        "cmd": " ".join(cmd), "config": cfg_path, "out_dir": out_dir,
    }
    if dry_run:
        rec.update({"status": "dry_run", "wall_s": 0.0})
        return rec
    log_path = os.path.join(out_dir, "train.log")
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
    rec["wall_s"] = round(time.time() - t0, 2)
    rec["status"] = "ok" if proc.returncode == 0 else f"fail({proc.returncode})"
    rec["log"] = log_path
    return rec


# -------- Experiment matrix ---------------------------------------------------

def matrix() -> List[Cell]:
    base_tag = os.path.join(REPO, "configs", "base.yaml")
    base_eco = os.path.join(REPO, "configs", "ecosystem.yaml")
    cells: List[Cell] = []

    # Reference baseline + seed-noise check. 200 eps matches base.yaml default.
    cells.append(Cell("tag_baseline", base_tag, 200,
                      overrides={"seed": 42},
                      group="baseline", notes="reference: 2v2, 200 steps, MAPPO critic"))
    cells.append(Cell("tag_baseline_seed7", base_tag, 200,
                      overrides={"seed": 7},
                      group="baseline", notes="alternate seed to gauge noise floor"))

    # A. Episode length sweep
    for steps in (50, 100, 400):
        cells.append(Cell(f"tag_steps_{steps}", base_tag, 200,
                          overrides={"env": {"max_steps": steps}},
                          group="episode_length",
                          notes=f"max_steps={steps}"))

    # B. Predator count sweep (prey held at 2)
    for n in (1, 4, 6):
        cells.append(Cell(f"tag_pred_{n}", base_tag, 200,
                          overrides={"env": {"n_predators": n, "n_prey": 2}},
                          group="predator_count",
                          notes=f"{n} predators vs 2 prey"))

    # C. Prey count sweep (pred held at 2)
    for n in (1, 4, 6):
        cells.append(Cell(f"tag_prey_{n}", base_tag, 200,
                          overrides={"env": {"n_predators": 2, "n_prey": n}},
                          group="prey_count",
                          notes=f"2 predators vs {n} prey"))

    # D. Symmetric team size
    for n in (3, 4):
        cells.append(Cell(f"tag_{n}v{n}", base_tag, 200,
                          overrides={"env": {"n_predators": n, "n_prey": n}},
                          group="team_size",
                          notes=f"{n}v{n} balanced"))

    # E. Network capacity
    for h in (64, 256):
        cells.append(Cell(f"tag_hidden_{h}", base_tag, 200,
                          overrides={"train": {"hidden": h}},
                          group="hidden",
                          notes=f"hidden={h}"))

    # F. Decentralised critic
    cells.append(Cell("tag_decentralised", base_tag, 200,
                      overrides={"train": {"centralized_critic": False}},
                      group="critic",
                      notes="MAPPO off — per-agent critic only"))

    # G. Long training reference
    cells.append(Cell("tag_long_400", base_tag, 400,
                      overrides={"seed": 42},
                      group="duration",
                      notes="400-episode reference run"))

    # H. Ecosystem env: world size variation. Keep step budget modest so
    # the sweep finishes in a reasonable time (ecosystem is ~25x slower
    # than simple_tag at default settings).
    for ws, tag in ((1.5, "small"), (2.0, "med"), (3.0, "large")):
        cells.append(Cell(f"eco_world_{tag}", base_eco, 60,
                          overrides={
                              "env": {
                                  "max_steps": 400,
                                  "n_predators": 8, "n_prey": 8,
                                  "ecosystem": {"world_size": ws,
                                                "max_predators": 24,
                                                "max_prey": 24},
                              },
                          },
                          group="eco_world",
                          notes=f"world_size={ws}, 8 start each, 400-step eps"))

    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None,
                    help="Run only these cell ids or groups (substring match)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip a cell if its out_dir already contains a metrics.csv")
    a = ap.parse_args()

    cells = matrix()
    if a.only:
        keep = []
        for c in cells:
            if any(s in c.exp_id or s in c.group for s in a.only):
                keep.append(c)
        cells = keep
        if not cells:
            print("No cells matched --only filter")
            return 1

    os.makedirs(EXP_ROOT, exist_ok=True)
    manifest_path = os.path.join(EXP_ROOT, "sweep_manifest.json")
    records: List[Dict[str, Any]] = []
    t_start = time.time()
    print(f"Running {len(cells)} cells -> {EXP_ROOT}")
    for i, c in enumerate(cells, 1):
        existing = os.path.join(_run_dir(c.exp_id))
        already = False
        if a.skip_existing and os.path.isdir(existing):
            # Look for any nested metrics.csv under a run_* dir.
            for sub in os.listdir(existing):
                m = os.path.join(existing, sub, "metrics.csv")
                if os.path.isfile(m):
                    already = True
                    break
        tag = f"[{i}/{len(cells)}] {c.exp_id} ({c.group})"
        if already:
            print(tag, "SKIP (existing)")
            records.append({"exp_id": c.exp_id, "group": c.group, "status": "skipped"})
            continue
        print(tag, "->", c.notes)
        sys.stdout.flush()
        rec = run_cell(c, device=a.device, dry_run=a.dry_run)
        records.append(rec)
        print("    ", rec.get("status"), f"({rec.get('wall_s')}s)")
        with open(manifest_path, "w") as f:
            json.dump({"started": t_start, "elapsed": time.time() - t_start, "records": records}, f, indent=2)

    print(f"Done in {(time.time() - t_start) / 60:.1f} min")
    print("Manifest:", manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
