"""Проверки формул значимости. Запуск: python -m pytest tests/ -q  (или python tests/test_metrics.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import ams, z0, best_threshold, weighted_counts  # noqa: E402


def test_z0_matches_closed_form():
    s, b = 10.0, 100.0
    expected = np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))
    assert abs(z0(s, b) - expected) < 1e-12
    # при s << b приближение s/sqrt(b) работает с точностью ~ s/b
    assert abs(z0(1.0, 10000.0) - 0.01) < 1e-4


def test_ams_regularisation():
    assert ams(0.0, 100.0) == 0.0
    assert ams(10.0, 100.0) < z0(10.0, 100.0)  # b_r = 10 слегка уменьшает AMS
    assert abs(ams(10.0, 100.0) - np.sqrt(2 * (120 * np.log(1 + 10 / 110) - 10))) < 1e-12


def test_weighted_counts():
    y = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 1, 0])
    w = np.array([2.0, 3.0, 5.0, 7.0])
    assert weighted_counts(y, pred, w) == (2.0, 5.0)


def test_best_threshold_prefers_pure_signal():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(500), np.zeros(5000)]
    scores = np.r_[rng.normal(0.8, 0.1, 500), rng.normal(0.3, 0.15, 5000)]
    w = np.r_[np.full(500, 1.0), np.full(5000, 100.0)]
    res = best_threshold(y, scores, w, metric="ams")
    assert 0.5 < res["threshold"] < 1.0
    assert res["significance"] > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
