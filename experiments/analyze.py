"""Aggregate metrics.csv from every experiment cell, build a summary table,
comparison plots per group, and a markdown report.

Run after sweeps.py finishes:
  python experiments/analyze.py
"""
from __future__ import annotations
import glob, json, os, sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_ROOT = os.path.join(REPO, "artifacts", "experiments")
ANALYSIS_DIR = os.path.join(EXP_ROOT, "_analysis")


def _find_metrics(exp_id: str) -> Optional[str]:
    """Each cell writes to artifacts/experiments/<exp_id>/run_<ts>/metrics.csv."""
    cands = sorted(glob.glob(os.path.join(EXP_ROOT, exp_id, "run_*", "metrics.csv")))
    return cands[-1] if cands else None


def load_manifest() -> List[Dict]:
    p = os.path.join(EXP_ROOT, "sweep_manifest.json")
    if not os.path.isfile(p):
        return []
    with open(p) as f:
        return json.load(f).get("records", [])


def load_all() -> Dict[str, Dict]:
    """exp_id -> {df, overrides, group, episodes, wall_s, notes}."""
    out: Dict[str, Dict] = {}
    for rec in load_manifest():
        exp_id = rec.get("exp_id")
        if not exp_id:
            continue
        mp = _find_metrics(exp_id)
        if not mp:
            continue
        df = pd.read_csv(mp)
        out[exp_id] = {
            "df": df, "group": rec.get("group", ""),
            "overrides": rec.get("overrides", {}),
            "episodes": rec.get("episodes"),
            "wall_s": rec.get("wall_s"),
            "notes": rec.get("notes", ""),
        }
    return out


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def summarise(exp_id: str, info: Dict) -> Dict:
    df = info["df"]
    # Use last 25% of episodes as the "settled" window — captures asymptote, not noise.
    n = len(df)
    tail = max(1, n // 4)
    last = df.tail(tail)
    rec = {
        "exp_id": exp_id,
        "group": info["group"],
        "episodes": n,
        "wall_s": info["wall_s"],
        "pred_return_final": last["pred_return"].mean(),
        "prey_return_final": last["prey_return"].mean(),
        "captures_final": last["captures"].mean(),
        "captures_total": df["captures"].sum(),
        "ep_steps_mean": df["ep_steps"].mean(),
        "pred_entropy_final": last["pred_entropy"].mean(),
        "prey_entropy_final": last["prey_entropy"].mean(),
    }
    if "pred_pop_mean" in df.columns:
        rec["pred_pop_mean"] = last["pred_pop_mean"].mean()
        rec["prey_pop_mean"] = last["prey_pop_mean"].mean()
        # Growth: end pop relative to start (only meaningful for ecosystem).
        rec["pred_pop_end"] = last["pred_pop_end"].mean()
        rec["prey_pop_end"] = last["prey_pop_end"].mean()
    # Slope of pred_return over training (cheap learning-speed proxy).
    if n >= 10:
        x = np.arange(n)
        slope_pred = np.polyfit(x, df["pred_return"].values, 1)[0]
        slope_capt = np.polyfit(x, df["captures"].values, 1)[0]
        rec["pred_return_slope"] = slope_pred
        rec["captures_slope"] = slope_capt
    return rec


def make_group_plot(group: str, items: List[Tuple[str, Dict]], out_path: str):
    """Overlay pred_return and captures curves for all cells in a group."""
    if not items:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for exp_id, info in items:
        df = info["df"]
        # MA window relative to series length, capped.
        w = max(1, min(10, len(df) // 10))
        x_smooth = np.arange(w, len(df) + 1)
        if len(df) >= w:
            axes[0].plot(x_smooth, smooth(df["pred_return"].values, w), label=exp_id, alpha=0.85)
            axes[1].plot(x_smooth, smooth(df["captures"].values, w), label=exp_id, alpha=0.85)
        else:
            axes[0].plot(df.index + 1, df["pred_return"], label=exp_id, alpha=0.85)
            axes[1].plot(df.index + 1, df["captures"], label=exp_id, alpha=0.85)
    axes[0].set_title(f"{group}: predator return (smoothed)")
    axes[0].set_xlabel("episode"); axes[0].set_ylabel("pred return")
    axes[1].set_title(f"{group}: captures per episode (smoothed)")
    axes[1].set_xlabel("episode"); axes[1].set_ylabel("captures")
    for ax in axes:
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    runs = load_all()
    if not runs:
        print("No runs found. Did sweeps.py finish?")
        return 1

    # Per-cell summary
    summary_rows = [summarise(k, v) for k, v in runs.items()]
    summary = pd.DataFrame(summary_rows).sort_values(["group", "exp_id"])
    csv_path = os.path.join(ANALYSIS_DIR, "summary.csv")
    summary.to_csv(csv_path, index=False, float_format="%.4f")
    print("Wrote:", csv_path)

    # Group plots
    groups: Dict[str, List[Tuple[str, Dict]]] = {}
    for k, v in runs.items():
        groups.setdefault(v["group"], []).append((k, v))
    for g, items in groups.items():
        out = os.path.join(ANALYSIS_DIR, f"group_{g}.png")
        make_group_plot(g, items, out)
        print("Wrote:", out)

    # Per-group best/worst extraction
    def _best_worst(g_df: pd.DataFrame, metric: str, higher_is_better: bool = True):
        s = g_df.dropna(subset=[metric])
        if s.empty:
            return None, None
        idx_b = s[metric].idxmax() if higher_is_better else s[metric].idxmin()
        idx_w = s[metric].idxmin() if higher_is_better else s[metric].idxmax()
        return s.loc[idx_b], s.loc[idx_w]

    # Markdown report
    report_path = os.path.join(ANALYSIS_DIR, "REPORT.md")
    with open(report_path, "w") as f:
        f.write("# Experiment sweep report\n\n")
        f.write(
            "Predator-prey co-evolution sweep across episode length, agent counts, "
            "team size, network capacity, critic style, training duration, and "
            "ecosystem world size. Each cell is a single run; the baseline is replicated "
            "under two seeds to gauge the noise floor.\n\n"
        )
        f.write(f"Total cells: **{len(runs)}**.\n\n")

        f.write("## Summary table\n\n")
        f.write("Final metrics are means over the **last 25%** of each run's episodes "
                "(asymptote, not noise). Slopes are linear fits over the whole run.\n\n")
        cols = ["exp_id", "group", "episodes", "wall_s",
                "pred_return_final", "prey_return_final", "captures_final",
                "captures_slope", "pred_return_slope",
                "ep_steps_mean", "pred_entropy_final"]
        if "pred_pop_mean" in summary.columns:
            cols += ["pred_pop_mean", "prey_pop_mean"]
        keep = [c for c in cols if c in summary.columns]
        f.write(summary[keep].to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n")

        f.write("## Per-group best/worst\n\n")
        f.write("For each sweep dimension, the cell with the highest mean captures "
                "(final 25%) vs. the lowest. Captures is the cleanest learning signal "
                "in simple_tag — it counts predator-prey collisions.\n\n")
        f.write("| group | best (captures) | captures_final | worst (captures) | captures_final |\n")
        f.write("|---|---|---|---|---|\n")
        for g in sorted(summary["group"].unique()):
            gdf = summary[summary["group"] == g]
            b, w = _best_worst(gdf, "captures_final", higher_is_better=True)
            if b is None:
                continue
            f.write(f"| {g} | `{b['exp_id']}` | {b['captures_final']:.2f} | "
                    f"`{w['exp_id']}` | {w['captures_final']:.2f} |\n")
        f.write("\n")

        f.write("## Group comparison plots\n\n")
        for g in sorted(groups):
            f.write(f"### {g}\n\n![{g}](group_{g}.png)\n\n")

        f.write("## Notes\n\n")
        f.write("- `captures_final` is the headline metric. For predators, higher = better "
                "policy; for prey it's the inverse (lower captures = better evasion).\n")
        f.write("- `pred_return_slope` ≈ learning speed (per-episode return slope).\n")
        f.write("- `pred_entropy_final` near 0 = degenerate (collapsed) policy; "
                "near `ln(act_dim)` = uniform random.\n")
        f.write("- For the ecosystem env, `pred_pop_mean`/`prey_pop_mean` capture "
                "whether the populations actually coexist over the episode.\n")
    print("Wrote:", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
