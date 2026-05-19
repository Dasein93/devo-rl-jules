#!/usr/bin/env python3
"""Plot mean genome traits over training for an ecosystem run.

Reads metrics.csv (columns pred_speed_g, pred_hp_g, pred_sense_g,
prey_speed_g, prey_hp_g, prey_sense_g) and produces a 2x3 grid plot:
- rows: predator (top), prey (bottom)
- columns: speed, HP, sense traits

Usage:
    python tools/genome_trace.py --run artifacts/run_YYYYMMDD_HHMMSS --out genome.png
"""
from __future__ import annotations
import argparse
import csv, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_TRAITS = ["speed_g", "hp_g", "sense_g"]
_TRAIT_NAMES = ["speed", "HP", "sense radius"]


def _smooth(xs: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(xs) < w:
        return xs
    return np.convolve(xs, np.ones(w) / w, mode="valid")


def plot_genome(run_dir: str, out_path: str, window: int = 10, dpi: int = 130):
    metrics = os.path.join(run_dir, "metrics.csv")
    with open(metrics) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty metrics: {metrics}")
    needed = [f"{team}_{trait}" for team in ("pred", "prey") for trait in _TRAITS]
    for col in needed:
        if col not in rows[0]:
            raise ValueError(f"Column '{col}' missing from {metrics}. "
                             f"Did you train with the ecosystem env (genome enabled)?")

    eps = np.array([int(r["episode"]) for r in rows])

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), dpi=dpi, sharex=True)
    colors = {"pred": "tab:red", "prey": "tab:green"}
    for row, team in enumerate(["pred", "prey"]):
        for col, trait in enumerate(_TRAITS):
            ax = axes[row, col]
            raw = np.array([float(r[f"{team}_{trait}"]) for r in rows])
            ax.plot(eps, raw, color=colors[team], alpha=0.25)
            sm = _smooth(raw, window)
            if len(sm) >= 2:
                ax.plot(eps[len(eps) - len(sm):], sm, color=colors[team], linewidth=2)
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
            ax.set_title(f"{team} {_TRAIT_NAMES[col]} (mean genome)")
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel(team)
            if row == 1:
                ax.set_xlabel("episode")
    fig.suptitle(f"Genome trait drift over training — {os.path.basename(run_dir.rstrip('/'))}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to artifacts/run_* directory")
    ap.add_argument("--out", default=None, help="Output PNG path (default: <run>/plots/genome.png)")
    ap.add_argument("--window", type=int, default=10, help="MA smoothing window")
    ap.add_argument("--dpi", type=int, default=130)
    args = ap.parse_args()
    out = args.out or os.path.join(args.run, "plots", "genome.png")
    plot_genome(args.run, out, window=args.window, dpi=args.dpi)


if __name__ == "__main__":
    main()
