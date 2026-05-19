"""Smoke tests for tools/phase_portrait.py."""
import csv
import os

import numpy as np
import pytest

from tools.phase_portrait import plot_phase_portrait


def _write_metrics(path: str, n: int = 100):
    """Synthetic Lotka-Volterra-like wobbles around (15, 10)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "pred_pop_mean", "prey_pop_mean"])
        for ep in range(1, n + 1):
            # Two interlocked sinusoids → closed loops in phase space.
            t = ep * 0.15
            pred = 10.0 + 4.0 * np.sin(t)
            prey = 15.0 + 5.0 * np.sin(t - 0.6)
            w.writerow([ep, f"{pred:.4f}", f"{prey:.4f}"])


def test_plot_phase_portrait_writes_png(tmpdir):
    run = tmpdir / "run"
    run.mkdir()
    _write_metrics(str(run / "metrics.csv"), n=150)
    out = str(tmpdir / "phase.png")
    stats = plot_phase_portrait(str(run), out, smooth_window=5, dpi=80)
    assert os.path.getsize(out) > 0
    # Sinusoid setup has pred amplitude 4, prey amplitude 5 → ranges ≈ 8 and 10.
    assert 6.0 < stats["pred_range"] < 10.0
    assert 8.0 < stats["prey_range"] < 12.0


def test_plot_phase_portrait_missing_column_raises(tmpdir):
    run = tmpdir / "run"
    run.mkdir()
    with open(run / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "pred_return"])
        w.writerow([1, 0.5])
    with pytest.raises(ValueError, match="missing"):
        plot_phase_portrait(str(run), str(tmpdir / "p.png"))
