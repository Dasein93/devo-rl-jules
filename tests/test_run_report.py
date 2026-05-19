"""Smoke tests for tools/run_report.py."""
import csv
import os

import pytest

from tools.run_report import generate


def _write_metrics(path: str, n: int = 50, with_genome: bool = True):
    base = ["episode", "pred_return", "prey_return", "captures", "ep_steps",
            "pred_pop_end", "prey_pop_end", "pred_pop_mean", "prey_pop_mean"]
    genome = ["pred_speed_g", "pred_hp_g", "pred_sense_g",
              "prey_speed_g", "prey_hp_g", "prey_sense_g"]
    cols = base + (genome if with_genome else [])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for ep in range(1, n + 1):
            row = [ep, ep * 0.1, -ep * 0.05, 5, 200, 10, 10, 10.0, 10.0]
            if with_genome:
                # Trait drift: pred speed up, prey speed down.
                row += [1.0 + ep * 0.001, 1.0, 1.0, 1.0 - ep * 0.001, 1.0, 1.0]
            w.writerow(row)


def test_generate_minimal(tmpdir):
    run = tmpdir / "run"
    run.mkdir()
    _write_metrics(str(run / "metrics.csv"), n=30, with_genome=True)
    report = generate(str(run))
    assert os.path.exists(str(run / "report.md"))
    assert "Population dynamics" in report
    assert "Genome trait drift" in report
    assert "predator speed" in report


def test_generate_without_genome_columns(tmpdir):
    """Should still produce a report when genome columns aren't present."""
    run = tmpdir / "run"
    run.mkdir()
    _write_metrics(str(run / "metrics.csv"), n=20, with_genome=False)
    report = generate(str(run))
    assert "Genome trait drift" not in report
    assert "Population dynamics" in report


def test_generate_raises_when_metrics_missing(tmpdir):
    run = tmpdir / "run"
    run.mkdir()
    with pytest.raises(FileNotFoundError):
        generate(str(run))


def test_generate_emits_tournament_section_when_present(tmpdir):
    run = tmpdir / "run"
    run.mkdir()
    _write_metrics(str(run / "metrics.csv"), n=20)
    t = run / "tournament"
    t.mkdir()
    with open(t / "ratings.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["team", "snapshot", "elo"])
        w.writerow(["predator", "ep100", "1750.5"])
        w.writerow(["predator", "ep125", "1500.0"])
        w.writerow(["prey", "ep75", "1450.2"])
    report = generate(str(run))
    assert "top Elo predator: **ep100**" in report
    assert "top Elo prey:     **ep75**" in report
