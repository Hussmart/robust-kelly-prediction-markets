"""Tests for the calibration mathematics: parameter recovery, monotonicity, metrics."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.calibration.isotonic import IsotonicCalibrator, pava
from src.calibration.platt_scaling import PlattCalibrator, _logit, _sigmoid
from src.calibration.reliability_diagrams import (
    brier_score, cluster_bootstrap_predict, expected_calibration_error, log_loss, reliability_table,
)


def _simulate(n: int, a: float, b: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.03, 0.97, n)
    y = (rng.uniform(size=n) < _sigmoid(a * _logit(p) + b)).astype(float)
    return p, y


# ---------------------------------------------------------------- Platt
@pytest.mark.parametrize("a, b", [(1.0, 0.0), (0.6, 0.2), (1.5, -0.3)])
def test_platt_recovers_true_parameters(a, b):
    p, y = _simulate(60_000, a, b)
    cal = PlattCalibrator().fit(p, y)
    assert cal.a_ == pytest.approx(a, abs=0.05)
    assert cal.b_ == pytest.approx(b, abs=0.05)


def test_platt_matches_sklearn_logistic_regression():
    p, y = _simulate(5_000, 0.7, 0.1, seed=3)
    ours = PlattCalibrator(ridge=0.0).fit(p, y)
    ref = LogisticRegression(C=1e10, tol=1e-10, max_iter=1000).fit(_logit(p)[:, None], y)
    assert ours.a_ == pytest.approx(ref.coef_[0, 0], abs=1e-4)
    assert ours.b_ == pytest.approx(ref.intercept_[0], abs=1e-4)


def test_platt_is_monotone_when_a_positive():
    p, y = _simulate(2_000, 0.8, 0.0)
    q = PlattCalibrator().fit(p, y).predict(np.linspace(0.01, 0.99, 200))
    assert np.all(np.diff(q) > 0)


def test_platt_perfectly_separable_stays_finite():
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    y = np.array([0, 0, 0, 1, 1, 1.0])
    cal = PlattCalibrator().fit(p, y)
    assert np.isfinite([cal.a_, cal.b_]).all()


def test_platt_requires_two_classes():
    with pytest.raises(ValueError):
        PlattCalibrator().fit(np.array([0.2, 0.4]), np.array([1.0, 1.0]))


# ---------------------------------------------------------------- isotonic
def test_pava_matches_sklearn_and_is_monotone():
    rng = np.random.default_rng(1)
    y = rng.uniform(size=200)
    w = rng.uniform(0.5, 2.0, size=200)
    ours = pava(y, w)
    ref = IsotonicRegression().fit_transform(np.arange(200), y, sample_weight=w)
    np.testing.assert_allclose(ours, ref, atol=1e-10)
    assert np.all(np.diff(ours) >= -1e-12)


def test_pava_already_monotone_is_identity():
    y = np.array([0.1, 0.2, 0.2, 0.9])
    np.testing.assert_allclose(pava(y, np.ones(4)), y)


def test_isotonic_calibrator_monotone_and_bounded():
    p, y = _simulate(3_000, 0.6, 0.0, seed=5)
    cal = IsotonicCalibrator(floor=0.005, ceil=0.995).fit(p, y)
    q = cal.predict(np.linspace(0, 1, 500))
    assert np.all(np.diff(q) >= -1e-12)
    assert q.min() >= 0.005 and q.max() <= 0.995


def test_isotonic_handles_tied_prices():
    p = np.array([0.5, 0.5, 0.5, 0.9, 0.9])
    y = np.array([0, 1, 0, 1, 1.0])
    cal = IsotonicCalibrator().fit(p, y)
    assert cal.predict(np.array([0.5]))[0] == pytest.approx(1 / 3)


# ---------------------------------------------------------------- metrics
def test_brier_known_values():
    assert brier_score(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0
    assert brier_score(np.array([0.5, 0.5]), np.array([1, 0])) == 0.25


def test_log_loss_known_value():
    assert log_loss(np.array([0.5]), np.array([1])) == pytest.approx(np.log(2))


def test_ece_small_for_calibrated_large_for_miscalibrated():
    rng = np.random.default_rng(2)
    q = rng.uniform(size=50_000)
    y = (rng.uniform(size=q.size) < q).astype(float)
    assert expected_calibration_error(q, y) < 0.01
    assert expected_calibration_error(np.clip(q + 0.2, 0, 1), y) > 0.1


def test_reliability_bins_partition_all_samples():
    q = np.linspace(0, 1, 103)
    t = reliability_table(q, (q > 0.5).astype(float), n_bins=10)
    assert t["count"].sum() == 103


def test_calibration_reduces_ece_out_of_sample():
    p_tr, y_tr = _simulate(5_000, 0.5, 0.3, seed=7)
    p_te, y_te = _simulate(5_000, 0.5, 0.3, seed=8)
    raw = expected_calibration_error(p_te, y_te)
    for cal in (PlattCalibrator(), IsotonicCalibrator()):
        assert expected_calibration_error(cal.fit(p_tr, y_tr).predict(p_te), y_te) < raw / 2


# ---------------------------------------------------------------- bootstrap
def test_cluster_bootstrap_interval_covers_truth_and_orders():
    p, y = _simulate(4_000, 0.7, 0.0, seed=11)
    groups = np.arange(len(p)) // 20
    grid = np.array([0.2, 0.5, 0.8])
    med, lo, hi = cluster_bootstrap_predict(PlattCalibrator, p, y, groups, grid, n_boot=200, seed=1)
    truth = _sigmoid(0.7 * _logit(grid))
    assert np.all(lo <= med) and np.all(med <= hi)
    assert np.all((lo - 0.03 <= truth) & (truth <= hi + 0.03))


def test_cluster_bootstrap_wider_with_fewer_clusters():
    p, y = _simulate(2_000, 0.7, 0.0, seed=13)
    few = cluster_bootstrap_predict(PlattCalibrator, p, y, np.arange(2000) // 500, np.array([0.5]), 200)
    many = cluster_bootstrap_predict(PlattCalibrator, p, y, np.arange(2000) // 5, np.array([0.5]), 200)
    assert (few[2] - few[1])[0] > (many[2] - many[1])[0]


# ---------------------------------------------------------------- Laplace (Bayesian Platt) interval
def test_laplace_interval_brackets_prediction_and_widens_with_less_data():
    p_big, y_big = _simulate(5_000, 0.8, 0.0, seed=21)
    p_small, y_small = p_big[:100], y_big[:100]
    grid = np.array([0.1, 0.5, 0.9])
    widths = []
    for p, y in ((p_big, y_big), (p_small, y_small)):
        cal = PlattCalibrator().fit(p, y)
        lo, hi = cal.predict_interval(grid, 0.10)
        q = cal.predict(grid)
        assert np.all(lo < q) and np.all(q < hi)
        widths.append(hi - lo)
    assert np.all(widths[1] > widths[0])


def test_laplace_interval_has_roughly_nominal_coverage():
    a, b = 0.7, 0.2
    grid = np.array([0.15, 0.5, 0.85])
    truth = _sigmoid(a * _logit(grid) + b)
    hits, reps = 0.0, 300
    for seed in range(reps):
        p, y = _simulate(400, a, b, seed=1000 + seed)
        lo, hi = PlattCalibrator(ridge=1e-3).fit(p, y).predict_interval(grid, 0.10)
        hits += np.mean((lo <= truth) & (truth <= hi))
    assert 0.85 <= hits / reps <= 0.96


def test_laplace_interval_does_not_collapse_when_no_failures():
    """All favourites win -> a bootstrap sees no variation, the Bayesian interval stays wide."""
    p = np.repeat([0.9, 0.95, 0.97, 0.05, 0.03], 6)
    y = np.repeat([1.0, 1.0, 1.0, 0.0, 0.0], 6)
    cal = PlattCalibrator(ridge=1.0).fit(p, y)
    lo, _ = cal.predict_interval(np.array([0.93]), 0.10)
    assert cal.predict(np.array([0.93]))[0] - lo[0] > 0.01
    groups = np.repeat(np.arange(5), 6)
    _, blo, _ = cluster_bootstrap_predict(lambda: PlattCalibrator(ridge=1.0), p, y, groups, np.array([0.93]), 200)
    assert blo[0] > lo[0]     # the bootstrap lower bound is (spuriously) tighter


def test_sample_weights_change_the_fit():
    p, y = _simulate(500, 0.6, 0.0, seed=2)
    w = np.where(np.arange(500) < 250, 5.0, 1.0)
    assert PlattCalibrator().fit(p, y, w).a_ != PlattCalibrator().fit(p, y).a_
