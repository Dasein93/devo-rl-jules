"""Smoke tests for tools/genome_trace.py."""
import csv
import os

import pytest

from tools.genome_trace import plot_genome


def _write_fake_metrics(path: str, n: int = 30):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "episode",
            "pred_speed_g", "pred_hp_g", "pred_sense_g",
            "prey_speed_g", "prey_hp_g", "prey_sense_g",
        ])
        for ep in range(1, n + 1):
            # Pretend predator sense radius drifts upward, prey speed drifts upward.
            w.writerow([
                ep,
                f"{1.0:.4f}",
                f"{1.0:.4f}",
                f"{1.0 + 0.01 * ep:.4f}",
                f"{1.0 + 0.01 * ep:.4f}",
                f"{1.0:.4f}",
                f"{1.0:.4f}",
            ])


def test_plot_genome_writes_png(tmpdir):
    run = tmpdir / "run"
    run.mkdir()
    _write_fake_metrics(str(run / "metrics.csv"), n=40)
    out = str(tmpdir / "genome.png")
    plot_genome(str(run), out, window=5, dpi=80)
    assert os.path.getsize(out) > 0


def test_plot_genome_raises_when_columns_missing(tmpdir):
    run = tmpdir / "run"
    run.mkdir()
    with open(run / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "pred_return"])
        w.writerow([1, 0.5])
    with pytest.raises(ValueError, match="missing"):
        plot_genome(str(run), str(tmpdir / "g.png"))
