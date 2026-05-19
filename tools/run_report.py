#!/usr/bin/env python3
"""Auto-generate a markdown report from a single training run.

Reads metrics.csv + plots/ + tournament/ (if present) and produces a self-
contained markdown report. Useful for sharing a run's results without
hand-curating a writeup.

Usage:
    python tools/run_report.py --run artifacts/run_YYYYMMDD_HHMMSS \\
        [--out artifacts/run_YYYYMMDD_HHMMSS/report.md]
"""
from __future__ import annotations
import argparse
import csv, os
from typing import Optional

import numpy as np


def _safe_float(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _slice_stats(values, sl):
    arr = np.asarray([v for v in values[sl] if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def _has_genome_columns(header) -> bool:
    return all(col in header for col in
               ("pred_speed_g", "pred_hp_g", "pred_sense_g",
                "prey_speed_g", "prey_hp_g", "prey_sense_g"))


def _has_pop_columns(header) -> bool:
    return all(col in header for col in
               ("pred_pop_mean", "prey_pop_mean", "captures", "ep_steps"))


def generate(run_dir: str, out_path: Optional[str] = None) -> str:
    metrics_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"No metrics.csv in {run_dir}")

    with open(metrics_path) as f:
        rows = list(csv.DictReader(f))
    header = list(rows[0].keys()) if rows else []
    n = len(rows)
    if n == 0:
        raise ValueError(f"Empty metrics.csv at {metrics_path}")

    name = os.path.basename(run_dir.rstrip("/"))
    lines = [f"# Run report — `{name}`", ""]

    eps = np.array([int(r["episode"]) for r in rows])
    lines.append(f"- Episodes: **{n}** (ep {eps[0]} → ep {eps[-1]})")

    pred_ret = [_safe_float(r["pred_return"]) for r in rows]
    prey_ret = [_safe_float(r["prey_return"]) for r in rows]
    early = slice(0, max(1, n // 10))
    late = slice(max(0, n - n // 10), n)
    pe, _ = _slice_stats(pred_ret, early)
    pl, _ = _slice_stats(pred_ret, late)
    ye, _ = _slice_stats(prey_ret, early)
    yl, _ = _slice_stats(prey_ret, late)
    lines += [
        f"- Predator return: early {pe:+.2f} → late {pl:+.2f}",
        f"- Prey return:     early {ye:+.2f} → late {yl:+.2f}",
        "",
    ]

    plots_dir = os.path.join(run_dir, "plots")
    plot_rel = os.path.relpath(plots_dir, run_dir)
    if os.path.exists(os.path.join(plots_dir, "return.png")):
        lines += ["## Per-team learning curves", "",
                  f"![return]({plot_rel}/return.png)", ""]

    if _has_pop_columns(header):
        caps = np.array([int(_safe_float(r["captures"])) for r in rows])
        steps = np.array([int(_safe_float(r["ep_steps"])) for r in rows])
        pred_pop = np.array([_safe_float(r["pred_pop_mean"]) for r in rows])
        prey_pop = np.array([_safe_float(r["prey_pop_mean"]) for r in rows])

        lines += ["## Population dynamics", ""]
        lines += [
            f"- predator mean population: early {pred_pop[early].mean():.2f}"
            f" → late {pred_pop[late].mean():.2f}"
            f" (range across run: [{pred_pop.min():.1f}, {pred_pop.max():.1f}])",
            f"- prey mean population:     early {prey_pop[early].mean():.2f}"
            f" → late {prey_pop[late].mean():.2f}"
            f" (range across run: [{prey_pop.min():.1f}, {prey_pop.max():.1f}])",
            f"- total captures across all episodes: **{int(caps.sum())}**",
            f"- episode length: early {steps[early].mean():.0f} steps → late {steps[late].mean():.0f} steps",
            "",
        ]
        for fname, caption in [("pop_dynamics.png", "Per-episode populations / captures / lengths"),
                               ("phase.png", "Predator vs. prey phase portrait")]:
            full = os.path.join(run_dir, fname)
            if os.path.exists(full):
                lines += [f"### {caption}", "", f"![{fname}]({fname})", ""]

    if _has_genome_columns(header):
        lines += ["## Genome trait drift", ""]
        traits = [("pred_speed_g", "predator speed"),
                  ("pred_hp_g", "predator HP"),
                  ("pred_sense_g", "predator sense"),
                  ("prey_speed_g", "prey speed"),
                  ("prey_hp_g", "prey HP"),
                  ("prey_sense_g", "prey sense")]
        lines.append("| trait | early | late | Δ |")
        lines.append("|---|---:|---:|---:|")
        for col, label in traits:
            vals = [_safe_float(r[col]) for r in rows]
            v_early = [v for v in vals[early] if v > 0.0]
            v_late = [v for v in vals[late] if v > 0.0]
            e_val = np.mean(v_early) if v_early else float("nan")
            l_val = np.mean(v_late) if v_late else float("nan")
            d = l_val - e_val if (np.isfinite(e_val) and np.isfinite(l_val)) else float("nan")
            sign = "↑" if d > 0 else "↓"
            d_str = f"{sign} {abs(d):.4f}" if np.isfinite(d) else "n/a"
            lines.append(f"| {label} | {e_val:.4f} | {l_val:.4f} | {d_str} |")
        lines.append("")
        gn = os.path.join(run_dir, "genome.png")
        if os.path.exists(gn):
            lines += ["![genome](genome.png)", ""]

    league_pred = os.path.join(run_dir, "league", "predator")
    if os.path.isdir(league_pred):
        n_snaps = len([p for p in os.listdir(league_pred) if p.endswith(".pt")])
        lines += [f"## League", "",
                  f"- snapshots per team: **{n_snaps}**", ""]

    tourney_dir = os.path.join(run_dir, "tournament")
    if os.path.isdir(tourney_dir):
        lines += ["## Tournament", ""]
        ratings = os.path.join(tourney_dir, "ratings.csv")
        if os.path.exists(ratings):
            with open(ratings) as f:
                trows = list(csv.DictReader(f))
            top_pred = max((r for r in trows if r["team"] == "predator"),
                           key=lambda r: float(r["elo"]), default=None)
            top_prey = max((r for r in trows if r["team"] == "prey"),
                           key=lambda r: float(r["elo"]), default=None)
            if top_pred:
                lines.append(f"- top Elo predator: **{top_pred['snapshot']}** ({top_pred['elo']})")
            if top_prey:
                lines.append(f"- top Elo prey:     **{top_prey['snapshot']}** ({top_prey['elo']})")
            lines.append("")
        heatmap = os.path.join(tourney_dir, "heatmap.png")
        if os.path.exists(heatmap):
            rel = os.path.relpath(heatmap, run_dir)
            lines += [f"![tournament heatmap]({rel})", ""]

    eco_mp4 = os.path.join(run_dir, "ecosystem.mp4")
    if os.path.exists(eco_mp4):
        lines += ["## Replay", "",
                  "Split-panel replay (2D scene on the left, population over training on the right):",
                  "",
                  "`ecosystem.mp4`", ""]

    report = "\n".join(lines)
    if out_path is None:
        out_path = os.path.join(run_dir, "report.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Wrote {out_path}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to artifacts/run_* directory")
    ap.add_argument("--out", default=None, help="Output report path (default: <run>/report.md)")
    args = ap.parse_args()
    generate(args.run, args.out)


if __name__ == "__main__":
    main()
