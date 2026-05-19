import os
import csv
import pytest

from tools.compare_runs import compare, _load_metric, _smooth
import numpy as np


def _make_run(path: str, n: int, pred_offset: float = 0.0):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "pred_return", "prey_return"])
        for ep in range(1, n + 1):
            w.writerow([ep, f"{pred_offset + ep * 0.1:.4f}", f"{-ep * 0.05:.4f}"])


def test_compare_two_runs_writes_png(tmpdir):
    run_a = os.path.join(tmpdir, "run_a")
    run_b = os.path.join(tmpdir, "run_b")
    _make_run(run_a, 30, pred_offset=0.0)
    _make_run(run_b, 30, pred_offset=2.0)
    out = os.path.join(tmpdir, "compare.png")
    compare([run_a, run_b], ["A", "B"], metric="pred_return", out=out, window=5, dpi=80)
    assert os.path.getsize(out) > 0


def test_load_metric_unknown_column_raises(tmpdir):
    run = os.path.join(tmpdir, "run")
    _make_run(run, 5)
    with pytest.raises(ValueError, match="not in"):
        _load_metric(run, "nonexistent")


def test_smooth_window_one_is_identity():
    xs = np.array([1.0, 2.0, 3.0])
    out = _smooth(xs, 1)
    np.testing.assert_array_equal(out, xs)


def test_smooth_short_series_returns_input():
    xs = np.array([1.0, 2.0])
    out = _smooth(xs, 5)
    np.testing.assert_array_equal(out, xs)


def test_label_count_mismatch_raises(tmpdir):
    run = os.path.join(tmpdir, "run")
    _make_run(run, 5)
    with pytest.raises(ValueError, match="count"):
        compare([run], ["A", "B"], metric="pred_return", out=str(tmpdir / "x.png"), window=2, dpi=80)
