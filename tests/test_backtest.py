"""Tests for metrics and the walk-forward backtest engine (synthetic data, no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest_engine as be
from src.backtest import metrics as mt


# ---------------------------------------------------------------- metrics
def test_wealth_curve_and_cumulative_growth():
    r = np.log([2.0, 0.5, 1.5])
    np.testing.assert_allclose(mt.wealth_curve(r), [1.0, 2.0, 1.0, 1.5])
    assert mt.cumulative_log_growth(r) == pytest.approx(np.log(1.5))


def test_max_drawdown_known_paths():
    assert mt.max_drawdown(np.log([2.0, 1.0, 1.0])) == 0.0
    assert mt.max_drawdown(np.log([2.0, 0.5, 2.0])) == pytest.approx(0.5)
    assert mt.max_drawdown(np.log([0.5, 0.5])) == pytest.approx(0.75)


def test_sharpe_like_edge_cases():
    assert np.isnan(mt.sharpe_like(np.array([0.1])))
    assert np.isnan(mt.sharpe_like(np.array([0.1, 0.1, 0.1])))
    r = np.array([0.1, -0.05, 0.08, 0.02])
    assert mt.sharpe_like(r, 1.0) == pytest.approx(r.mean() / r.std(ddof=1))
    assert mt.sharpe_like(r, 4.0) == pytest.approx(2 * mt.sharpe_like(r, 1.0))


def test_summarize_keys_and_empty():
    s = mt.summarize(np.array([0.1, -0.1, 0.2]))
    assert s["win_rate"] == pytest.approx(2 / 3) and s["worst_round"] == -0.1
    assert mt.summarize(np.array([]))["rounds"] == 0.0


# ---------------------------------------------------------------- settlement
def _round(p=(0.7, 0.6), price=(0.5, 0.5), wins=(True, False), d=(0.05, 0.05)) -> be.Round:
    return be.Round("M", ["a", "b"], ["YES", "YES"], np.array(p), np.array(d), np.array(price),
                    np.array(wins), decision_ts=0)


def test_settle_payoffs():
    r = _round()                                    # b = 1 for both bets
    assert be.settle(r, np.array([0.2, 0.1])) == pytest.approx(np.log(1 + 0.2 - 0.1))
    assert be.settle(r, np.array([0.0, 0.0])) == 0.0
    assert be.settle(_round(wins=(False, False)), np.array([0.3, 0.3])) == pytest.approx(np.log(0.4))
    assert be.settle(be.Round("M", [], [], *(np.array([]),) * 4, 0), np.array([])) == 0.0


def test_equal_weight_respects_cap_and_budget():
    np.testing.assert_allclose(be.equal_weight(4, 0.6, 0.3), 0.15)
    np.testing.assert_allclose(be.equal_weight(1, 0.6, 0.3), 0.3)
    assert len(be.equal_weight(0, 0.6, 0.3)) == 0


def test_run_backtest_structure_and_gamma_zero_matches_naive():
    rounds = [_round(), _round(wins=(True, True)), _round(p=(0.55, 0.9), wins=(False, True))]
    for i, r in enumerate(rounds):
        r.meeting = f"M{i}"
    res = be.run_backtest(rounds, be.BacktestConfig(), gammas=(0.0, 1.0, 2.0), n_random=20)
    assert list(res.returns.columns) == ["naive_kelly", "robust_G0", "robust_G1", "robust_G2", "equal_weight", "random"]
    assert len(res.returns) == 3
    np.testing.assert_allclose(res.returns["naive_kelly"], res.returns["robust_G0"], atol=5e-3)
    assert set(res.summary().columns) >= {"cum_log_growth", "max_drawdown", "sharpe_like"}


def test_empty_round_gives_zero_return_for_all_strategies():
    empty = be.Round("M", [], [], *(np.array([]),) * 4, 0)
    res = be.run_backtest([empty], be.BacktestConfig(), gammas=(1.0,), n_random=5)
    assert (res.returns.to_numpy() == 0).all()


# ---------------------------------------------------------------- walk-forward, no look-ahead
def _synthetic_features(n_meetings: int = 12, seed: int = 0) -> pd.DataFrame:
    """3 buckets per meeting; snapshots at 30/24/18 h before close; venue B is noisy."""
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    for m in range(n_meetings):
        t_event = t0 + pd.Timedelta(days=45 * m)
        win = rng.integers(0, 3)
        for k in range(3):
            truth = 0.8 if k == win else 0.1
            for h in (30, 24, 18):
                pa = float(np.clip(truth + rng.normal(0, 0.03), 0.02, 0.98))
                pb = float(np.clip(truth + rng.normal(0, 0.08), 0.02, 0.98))
                rows.append(dict(
                    pair_id=f"m{m}-{k}", meeting=f"m{m}", bucket=k, ts=int(t_event.timestamp() - h * 3600),
                    hours_to_close=float(h), p_poly=pa, p_kalshi=pb, outcome=int(k == win), event_time=t_event,
                    logit_div=float(np.log(pa / (1 - pa)) - np.log(pb / (1 - pb))), spread_kalshi=0.02,
                    volume_poly=1000.0, volume_kalshi=800.0, stale_poly_h=1.0, stale_kalshi_h=1.0,
                    roll_spread_poly=0.01, ofi_poly=0.0, ofi_kalshi=0.0, vol_poly=0.1, vol_kalshi=0.1,
                    abs_div=abs(pa - pb),
                ))
    return pd.DataFrame(rows)


def test_build_rounds_skips_burn_in_and_uses_only_past():
    feats = _synthetic_features()
    cfg = be.BacktestConfig(min_train_meetings=6, n_boot=30)
    rounds = be.build_rounds(feats, cfg)
    assert [r.meeting for r in rounds][0] == "m6"          # first 6 meetings are burn-in
    assert all(r.n_train_meetings >= 6 for r in rounds)
    for r in rounds:
        assert r.decision_ts < feats[feats.meeting == r.meeting].event_time.iloc[0].timestamp()


def test_future_outcomes_do_not_change_past_decisions():
    """Flipping the outcomes of the LAST meeting must not change any earlier round."""
    feats = _synthetic_features()
    cfg = be.BacktestConfig(n_boot=30)
    base = be.build_rounds(feats, cfg)
    flipped = feats.copy()
    last = flipped.meeting == "m11"
    flipped.loc[last, "outcome"] = 1 - flipped.loc[last, "outcome"]
    alt = be.build_rounds(flipped, cfg)
    for a, b in zip(base[:-1], alt[:-1]):
        np.testing.assert_allclose(a.p_hat, b.p_hat)
        np.testing.assert_allclose(a.d, b.d)
        assert a.pair_ids == b.pair_ids


def test_candidates_have_positive_edge_and_valid_prices():
    rounds = be.build_rounds(_synthetic_features(), be.BacktestConfig(n_boot=30))
    for r in rounds:
        assert np.all(r.p_hat > r.price) and np.all((r.price > 0) & (r.price < 1))
        assert np.all(r.d >= 0)
