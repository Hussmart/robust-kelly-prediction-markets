"""Tests for the broad-universe cohorts, the universe backtest and the paired bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest_engine as be
from src.backtest import metrics as mt
from src.backtest import universe_backtest as ub
from src.collectors import universe as uni

DAY = 86_400


# ---------------------------------------------------------------- cohort assignment
def _markets(end_days: list[float], closed_days: list[float] | None = None) -> pd.DataFrame:
    end = [uni.T0 + pd.Timedelta(days=d) for d in end_days]
    closed = [uni.T0 + pd.Timedelta(days=d) for d in (closed_days or end_days)]
    return pd.DataFrame({"end_date": end, "closed_time": closed, "market_id": [str(i) for i in range(len(end))]})


def test_cohort_window_starts_three_days_after_the_decision_date():
    m = uni.assign_cohorts(_markets([2.0, 3.0, 13.9, 14.0, 16.5, 27.9, 28.0]))
    # 2.0 and 16.5 (= 2.5 mod 14) end within 3 days of a decision date; 14.0 / 28.0 sit
    # exactly on the next decision date: all dropped.
    assert m.market_id.tolist() == ["1", "2", "5"]
    assert m.cohort.tolist() == [0, 0, 1]
    assert m.horizon_days.between(3, 14, inclusive="left").all()


def test_cohorts_use_only_the_scheduled_end_date_not_the_actual_resolution():
    """An early-resolving market must be assigned by its scheduled end, never by when it closed."""
    a = uni.assign_cohorts(_markets([40.0, 50.0], closed_days=[33.0, 51.0]))
    # market 0 is scheduled to end in cohort 2 (day 40 = 28 + 12) although it closed early, on day 33
    assert a.loc[a.market_id == "0", "cohort"].tolist() == [2]


def test_markets_already_closed_at_the_decision_date_are_excluded():
    m = uni.assign_cohorts(_markets([10.0, 10.0], closed_days=[-1.0, 9.0]))
    assert m.market_id.tolist() == ["1"]                    # market 0 had closed before T_0
    m = uni.assign_cohorts(_markets([24.0], closed_days=[13.0]))   # cohort 1, closed at day 13 < T_1 = 14
    assert m.empty


def test_decision_precedes_scheduled_end_and_cohorts_do_not_overlap():
    m = uni.assign_cohorts(_markets(list(np.arange(0.0, 200.0, 0.5))))
    end_ts = (m.end_date - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    assert (m.decision_ts < end_ts).all()
    assert (end_ts < m.decision_ts + uni.COHORT_DAYS * DAY).all()


def test_late_resolution_is_flagged_as_overlap():
    m = uni.assign_cohorts(_markets([10.0, 10.0], closed_days=[10.0, 20.0]))
    assert m.settles_after_next.tolist() == [False, True]


# ---------------------------------------------------------------- universe backtest
def _universe(n_cohorts: int = 14, per_cohort: int = 60, seed: int = 0) -> pd.DataFrame:
    """Markets whose YES probability is 0.7 * price + 0.15 (so the market is mis-calibrated)."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_cohorts):
        for i in range(per_cohort):
            price = float(rng.uniform(0.05, 0.95))
            true_p = 0.15 + 0.7 * price
            rows.append(dict(market_id=f"{k}-{i}", event_key=f"e{k}-{i // 2}", cohort=k,
                             decision_ts=int(uni.T0.timestamp()) + k * 14 * DAY, price=price,
                             outcome=float(rng.uniform() < true_p), horizon_days=10.0, volume=1e6, question="q"))
    return pd.DataFrame(rows)


def test_rounds_start_after_burn_in_and_respect_caps():
    cfg = ub.UniverseConfig(min_train_cohorts=6, max_bets=5)
    rounds = ub.build_universe_rounds(_universe(), cfg)
    assert rounds[0].meeting == "C06" and rounds[0].n_train_meetings == 6
    for r in rounds:
        assert len(r.p_hat) <= 5
        assert np.all(r.p_hat > r.price) and np.all((r.price > 0.01) & (r.price < 0.99)) and np.all(r.d >= 0)


def test_at_most_one_bet_per_event():
    rounds = ub.build_universe_rounds(_universe(), ub.UniverseConfig(min_train_cohorts=6, max_bets=60))
    u = _universe().set_index("market_id")
    for r in rounds:
        events = [u.loc[m, "event_key"] for m in r.pair_ids]
        assert len(events) == len(set(events))


def test_future_outcomes_do_not_change_past_rounds():
    """Flipping the outcomes of the last cohort must leave every earlier round unchanged."""
    u = _universe()
    cfg = ub.UniverseConfig(min_train_cohorts=6)
    base = ub.build_universe_rounds(u, cfg)
    flipped = u.copy()
    last = flipped.cohort == flipped.cohort.max()
    flipped.loc[last, "outcome"] = 1 - flipped.loc[last, "outcome"]
    alt = ub.build_universe_rounds(flipped, cfg)
    for a, b in zip(base[:-1], alt[:-1], strict=True):
        np.testing.assert_allclose(a.p_hat, b.p_hat)
        np.testing.assert_allclose(a.d, b.d)
        assert a.pair_ids == b.pair_ids and a.wins.tolist() == b.wins.tolist()


def test_calibration_learns_the_planted_miscalibration():
    """Prices are mis-calibrated by construction; walk-forward Platt must beat the raw price."""
    preds = ub.expanding_window_predictions(_universe(n_cohorts=20, per_cohort=150), ub.UniverseConfig(min_train_cohorts=6))
    from src.calibration.reliability_diagrams import brier_score

    y = preds.outcome.to_numpy()
    assert brier_score(preds.platt.to_numpy(), y) < brier_score(preds.raw.to_numpy(), y)


def test_run_backtest_on_universe_rounds_is_consistent():
    rounds = ub.build_universe_rounds(_universe(n_cohorts=10), ub.UniverseConfig(min_train_cohorts=6))
    res = be.run_backtest(rounds, be.BacktestConfig(budget=0.35, f_max=0.10), gammas=(0.0, 1.0), n_random=10)
    assert len(res.returns) == len(rounds)
    from src.optimization.naive_kelly import growth, kelly_fractions
    from src.optimization.robust_kelly import solve_robust_kelly

    for r in rounds:          # Gamma = 0 reproduces naive Kelly in *model* growth (realised P&L
        if len(r.p_hat) == 0:  # differs slightly because the MILP grid moves the stakes a little)
            continue
        naive = growth(kelly_fractions(r.p_hat, r.price, 0.35, 0.10), r.p_hat, r.price)
        rob = growth(solve_robust_kelly(r.p_hat, r.price, r.d, 0.0, 0.35, 0.10).f, r.p_hat, r.price)
        assert rob == pytest.approx(naive, abs=1e-4)


# ---------------------------------------------------------------- paired bootstrap
def test_paired_bootstrap_detects_a_clear_difference():
    rng = np.random.default_rng(0)
    b = rng.normal(0.0, 0.02, 60)
    out = mt.paired_bootstrap_diff(b + 0.05, b, n_boot=2000)
    assert out["observed"] == pytest.approx(60 * 0.05)
    assert out["lo"] > 0 and out["prob_positive"] == 1.0


def test_paired_bootstrap_interval_contains_zero_for_identical_strategies():
    a = np.random.default_rng(1).normal(0.01, 0.03, 40)
    out = mt.paired_bootstrap_diff(a, a.copy(), n_boot=500)
    assert out["lo"] == 0.0 == out["hi"] and out["observed"] == 0.0


def test_paired_bootstrap_validates_input():
    with pytest.raises(ValueError):
        mt.paired_bootstrap_diff(np.array([1.0, 2.0]), np.array([1.0]))


# ---------------------------------------------------------------- ex-ante selection
def test_sample_is_deterministic_and_ignores_outcomes():
    m = pd.DataFrame({"market_id": [str(i) for i in range(1000)], "outcome": np.arange(1000) % 2})
    a, b = uni.sample_markets(m, 100, seed=3), uni.sample_markets(m, 100, seed=3)
    assert a.market_id.tolist() == b.market_id.tolist() and len(a) == 100
    flipped = m.assign(outcome=1 - m.outcome)
    assert uni.sample_markets(flipped, 100, seed=3).market_id.tolist() == a.market_id.tolist()
    assert len(uni.sample_markets(m, 5000)) == 1000


class _FakePM:
    """Stub with a fixed price history; records the requested window."""

    def __init__(self, hist: pd.DataFrame) -> None:
        self.hist, self.calls = hist, []

    def price_history(self, token: str, start: int, end: int, fidelity: int = 60) -> pd.DataFrame:
        self.calls.append((start, end))
        return self.hist


def test_entry_features_use_only_data_up_to_the_decision_date():
    t = 1_000_000
    hist = pd.DataFrame({"ts": [t - 7200, t - 3600, t, t + 3600, t + 7200], "price": [0.2, 0.25, 0.3, 0.9, 0.95]})
    f = uni.entry_features(_FakePM(hist), "tok", t)
    assert f["price"] == 0.3 and f["price_age_h"] == 0.0 and f["n_changes"] == 2      # later points ignored


def test_activity_is_measured_by_price_changes_not_by_the_number_of_grid_points():
    """A dead market repeats its price on every hourly grid point; it must look inactive."""
    t = 1_000_000
    grid = np.arange(t - 143 * 3600, t + 1, 3600)
    dead = pd.DataFrame({"ts": grid, "price": np.full(len(grid), 0.4)})
    f = uni.entry_features(_FakePM(dead), "tok", t, lookback_h=144)
    assert f["n_changes"] == 0 and f["price_age_h"] == 144.0
    moved = dead.assign(price=np.where(np.arange(len(grid)) < 100, 0.4, 0.45))
    g = uni.entry_features(_FakePM(moved), "tok", t, lookback_h=144)
    assert g["n_changes"] == 1 and g["price_age_h"] == pytest.approx((len(grid) - 100) * 1.0, abs=1)


def test_entry_features_without_history_are_missing_not_zero():
    f = uni.entry_features(_FakePM(pd.DataFrame({"ts": [], "price": []})), "tok", 1_000_000)
    assert np.isnan(f["price"]) and f["n_changes"] == 0.0


def test_entry_features_request_a_single_window_ending_at_the_decision_date():
    pm = _FakePM(pd.DataFrame({"ts": [5], "price": [0.5]}))
    uni.entry_features(pm, "tok", 1_000_000, lookback_h=144)
    assert pm.calls == [(1_000_000 - 144 * 3600, 1_000_000)]
