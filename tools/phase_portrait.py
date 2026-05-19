#!/usr/bin/env python3
"""Plot the (prey_pop, pred_pop) phase portrait of an ecosystem run.

Reads metrics.csv pred_pop_mean / prey_pop_mean per episode and draws the
trajectory through the 2D phase space, colour-coded by episode index. Classic
Lotka-Volterra dynamics show as closed (or roughly closed) loops; a system
stuck at an equilibrium degenerates to a point cloud.

Usage:
    python tools/phase_portrait.py --run artifacts/run_YYYYMMDD_HHMMSS --out phase.png
"""
from __future__ import annotations
import argparse
import csv, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot_phase_portrait(run_dir: str, out_path: str, smooth_window: int = 10, dpi: int = 130):
    metrics = os.path.join(run_dir, "metrics.csv")
    with open(metrics) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty metrics: {metrics}")
    for col in ("pred_pop_mean", "prey_pop_mean", "episode"):
        if col not in rows[0]:
            raise ValueError(f"Column '{col}' missing from {metrics}")

    eps = np.array([int(r["episode"]) for r in rows])
    pred = np.array([float(r["pred_pop_mean"]) for r in rows])
    prey = np.array([float(r["prey_pop_mean"]) for r in rows])

    def _ma(x, w):
        if w <= 1 or len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    pred_s = _ma(pred, smooth_window)
    prey_s = _ma(prey, smooth_window)
    if len(pred_s) < len(pred):
        eps_s = eps[len(eps) - len(pred_s):]
    else:
        eps_s = eps

    fig, ax = plt.subplots(figsize=(7.5, 7.0), dpi=dpi)

    # Build a coloured trajectory: each segment between consecutive points gets a
    # colour determined by the midpoint episode.
    points = np.column_stack([prey_s, pred_s])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(eps_s[0], eps_s[-1])
    lc = LineCollection(segments, cmap="viridis", norm=norm, linewidth=1.5, alpha=0.8)
    lc.set_array(eps_s[:-1])
    ax.add_collection(lc)

    # Raw scatter underneath, more transparent.
    ax.scatter(prey, pred, c=eps, cmap="viridis", s=8, alpha=0.25)

    # Mark start and end.
    ax.scatter([prey_s[0]], [pred_s[0]], marker="o", s=160, facecolors="none",
               edgecolors="black", linewidths=2, label="start")
    ax.scatter([prey_s[-1]], [pred_s[-1]], marker="*", s=300, color="black", label="end")

    ax.set_xlabel("prey mean population")
    ax.set_ylabel("predator mean population")
    ax.set_title(f"Phase portrait — {os.path.basename(run_dir.rstrip('/'))}\n"
                 f"(MA{smooth_window} smoothed; raw points faded)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    cb = fig.colorbar(lc, ax=ax, label="episode")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")
    return {
        "pred_min": float(pred.min()), "pred_max": float(pred.max()),
        "prey_min": float(prey.min()), "prey_max": float(prey.max()),
        "pred_range": float(pred.max() - pred.min()),
        "prey_range": float(prey.max() - prey.min()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to artifacts/run_* directory")
    ap.add_argument("--out", default=None, help="Output PNG (default: <run>/plots/phase.png)")
    ap.add_argument("--smooth", type=int, default=10, help="Moving-average window for trajectory")
    ap.add_argument("--dpi", type=int, default=130)
    args = ap.parse_args()
    out = args.out or os.path.join(args.run, "plots", "phase.png")
    stats = plot_phase_portrait(args.run, out, smooth_window=args.smooth, dpi=args.dpi)
    print(f"pop ranges: predator [{stats['pred_min']:.1f}, {stats['pred_max']:.1f}]  "
          f"prey [{stats['prey_min']:.1f}, {stats['prey_max']:.1f}]")


if __name__ == "__main__":
    main()
