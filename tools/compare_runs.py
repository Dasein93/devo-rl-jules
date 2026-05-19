#!/usr/bin/env python3
"""Overlay metrics.csv curves from multiple runs.

Useful for ablations — e.g. MAPPO vs decentralised PPO, league on vs off,
or different hyperparameters.

Usage:
    python tools/compare_runs.py \\
        --runs artifacts/run_mappo artifacts/run_dec \\
        --labels MAPPO Decentralised \\
        --metric pred_return \\
        --out compare_pred.png
"""
from __future__ import annotations
import argparse
import csv, os
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_metric(run_dir: str, metric: str):
    path = os.path.join(run_dir, "metrics.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty")
    if metric not in rows[0]:
        raise ValueError(f"Metric '{metric}' not in {path}. Available: {list(rows[0])}")
    eps = np.asarray([int(r["episode"]) for r in rows])
    vals = np.asarray([float(r[metric]) for r in rows])
    return eps, vals


def _smooth(xs: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(xs) < w:
        return xs
    return np.convolve(xs, np.ones(w) / w, mode="valid")


def compare(runs: List[str], labels: List[str], metric: str, out: str,
            window: int, dpi: int):
    if labels and len(labels) != len(runs):
        raise ValueError(f"--labels count ({len(labels)}) must match --runs count ({len(runs)})")
    if not labels:
        labels = [os.path.basename(r.rstrip("/")) for r in runs]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)
    cmap = plt.get_cmap("tab10")

    for i, (run, label) in enumerate(zip(runs, labels)):
        eps, vals = _load_metric(run, metric)
        color = cmap(i % 10)
        ax.plot(eps, vals, color=color, alpha=0.25)
        sm = _smooth(vals, window)
        if len(sm) >= 2:
            ax.plot(eps[len(eps) - len(sm):], sm, color=color, label=f"{label} (MA{window})", linewidth=2)
        else:
            ax.plot(eps, vals, color=color, label=label, linewidth=1.5)

    ax.set_xlabel("episode")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} across {len(runs)} run(s)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="Run directories to compare")
    ap.add_argument("--labels", nargs="+", default=None, help="Label per run (default: dir basename)")
    ap.add_argument("--metric", default="pred_return", help="Column from metrics.csv")
    ap.add_argument("--out", default="compare.png")
    ap.add_argument("--window", type=int, default=25, help="Moving-average window")
    ap.add_argument("--dpi", type=int, default=130)
    args = ap.parse_args()
    compare(args.runs, args.labels, args.metric, args.out, args.window, args.dpi)


if __name__ == "__main__":
    main()
