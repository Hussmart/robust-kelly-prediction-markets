"""Tests for feature engineering: edge cases (missing data, zero volume) and no look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import feature_engineering as fe

H = fe.HOUR
T0 = 1_700_000_000
T_END = T0 + 10 * 24 * H


def _pair() -> pd.Series:
    return pd.Series({"pair_id": "X-H0", "meeting": "25DEC", "bucket": "hold", "outcome": 1})


def _prices(p: float = 0.6, start: int = T0, end: int = T_END) -> pd.DataFrame:
    ts = np.arange(start, end, H)
    return pd.DataFrame({"ts": ts, "price": np.full(len(ts), p)})


def _candles(bid: float = 0.55, ask: float = 0.57, start: int = T0, end: int = T_END) -> pd.DataFrame:
    ts = np.arange(start, end, H)
    n = len(ts)
    return pd.DataFrame({"ts": ts, "price": np.full(n, 0.56), "yes_bid": np.full(n, bid),
                         "yes_ask": np.full(n, ask), "volume": np.ones(n), "open_interest": np.ones(n)})


def _trades(start: int = T0, end: int = T_END, every_h: int = 1, price: float = 0.6) -> pd.DataFrame:
    ts = np.arange(start, end, every_h * H)
    n = len(ts)
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    return pd.DataFrame({"ts": ts, "yes_price": np.full(n, price), "size": np.full(n, 10.0),
                         "notional": np.full(n, 6.0), "yes_flow": 6.0 * sign})


EMPTY_TRADES = pd.DataFrame({c: pd.Series(dtype=float) for c in ["ts", "yes_price", "size", "notional", "yes_flow"]})


def _panel(**kw) -> pd.DataFrame:
    args = dict(pair=_pair(), start_ts=T0, end_ts=T_END, poly_prices=_prices(), poly_trades=_trades(),
                kalshi_candles=_candles(), kalshi_trades=_trades())
    args.update(kw)
    return fe.build_pair_panel(**args)


# ---------------------------------------------------------------- primitives
def test_logit_is_clipped_and_antisymmetric():
    assert np.isfinite(fe.logit(0.0)) and np.isfinite(fe.logit(1.0))
    np.testing.assert_allclose(fe.logit(0.3), -fe.logit(0.7))


def test_asof_respects_max_age_and_skips_nan():
    ts = np.array([0, 10 * H])
    vals = np.array([0.4, np.nan])
    out = fe.asof(np.array([-1, 5 * H, 30 * H]), ts, vals, max_age_h=24)
    assert np.isnan(out[0])            # before the first observation
    assert out[1] == 0.4               # NaN at 10h is skipped
    assert np.isnan(out[2])            # 30h old -> too stale


def test_kalshi_invalid_quote_falls_back_to_last_trade():
    c = pd.DataFrame({"ts": [1, 2], "price": [0.3, 0.3], "yes_bid": [0.0, 0.29],
                      "yes_ask": [1.0, 0.31], "volume": [0, 1], "open_interest": [1, 1]})
    q = fe.kalshi_implied_price(c)
    assert q.p.tolist() == pytest.approx([0.3, 0.30]) and np.isnan(q.spread[0]) and q.spread[1] == pytest.approx(0.02)


def test_roll_spread_recovers_known_spread():
    """Simulate bid-ask bounce around a random walk; Roll's estimator should recover s."""
    rng = np.random.default_rng(0)
    s = 0.04
    efficient = 0.5 + np.cumsum(rng.normal(0, 0.002, 20_000))
    trades = efficient + rng.choice([-1, 1], size=efficient.size) * s / 2
    assert fe.roll_spread(trades) == pytest.approx(s, rel=0.1)


def test_roll_spread_edge_cases():
    assert np.isnan(fe.roll_spread(np.array([0.5, 0.51])))          # too few trades
    assert fe.roll_spread(np.linspace(0.1, 0.9, 50) ** 2) == 0.0      # positive autocov


# ---------------------------------------------------------------- panel edge cases
def test_zero_volume_gives_zero_not_nan():
    panel = _panel(poly_trades=EMPTY_TRADES)
    assert (panel.volume_poly == 0).all() and (panel.ofi_poly == 0).all()
    assert panel.stale_poly_h.isna().all()
    assert np.isfinite(panel.log_volume_min).all()


def test_missing_prices_rows_are_dropped():
    # Polymarket market only starts trading on day 6: earlier snapshots must be dropped.
    late = T0 + 6 * 24 * H
    panel = _panel(poly_prices=_prices(start=late))
    assert len(panel) > 0
    assert (panel.ts >= late).all()
    assert panel[["p_poly", "p_kalshi"]].notna().all().all()


def test_stale_prices_are_dropped():
    # Poly prices stop 3 days before the end -> older than 48h -> rows dropped.
    panel = _panel(poly_prices=_prices(end=T_END - 3 * 24 * H))
    assert panel.hours_to_close.min() > 24


def test_divergence_and_ofi_values():
    panel = _panel()
    assert panel["div"].iloc[-1] == pytest.approx(0.6 - 0.56)
    assert panel.spread_kalshi.iloc[-1] == pytest.approx(0.02)
    assert panel.ofi_poly.abs().max() <= 1.0 / 24 + 1e-9  # alternating signs cancel out


def test_no_lookahead():
    """Perturbing data after snapshot t must not change any feature at or before t."""
    base = _panel()
    cut = int(base.ts.iloc[len(base) // 2])
    tr = _trades()
    tr.loc[tr.ts > cut, "notional"] *= 50
    tr.loc[tr.ts > cut, "yes_price"] = 0.9
    px = _prices()
    px.loc[px.ts > cut, "price"] = 0.95
    moved = _panel(poly_prices=px, poly_trades=tr)
    cols = [c for c in base.columns if c not in ("pair_id", "meeting", "bucket")]
    before = base.ts <= cut
    pd.testing.assert_frame_equal(base.loc[before, cols], moved.loc[before, cols])


def test_persistence_label():
    ts = np.arange(0, 10) * 6 * H
    div = np.array([0.05, 0.04, 0.04, 0.04, 0.05, 0.001, 0.0, -0.05, 0.0, 0.0])
    panel = pd.DataFrame({"pair_id": "a", "ts": ts, "div": div})
    lab = fe.add_persistence_label(panel, horizon_h=24, min_div=0.02)["persistent"].to_numpy()
    assert lab[0] == 1.0          # 0.05 -> 0.05 a day later, same sign
    assert lab[1] == 0.0          # 0.04 -> 0.001
    assert np.isnan(lab[5])       # below min_div
    assert np.isnan(lab[7])       # no snapshot 24h ahead


def test_kalshi_zero_bid_quote_is_valid_but_empty_book_is_not():
    """A (0, 0.02) quote on a long shot is real (mid 1c); the empty (0, 1) book is not."""
    c = pd.DataFrame({"ts": [1, 2, 3], "price": [0.05, 0.05, 0.05], "yes_bid": [0.0, 0.0, 0.03],
                      "yes_ask": [0.02, 1.0, 0.03], "volume": [0, 0, 0], "open_interest": [1, 1, 1]})
    q = fe.kalshi_implied_price(c)
    assert q.p[0] == pytest.approx(0.01) and q.spread[0] == pytest.approx(0.02)
    assert q.p[1] == pytest.approx(0.05) and np.isnan(q.spread[1])       # empty book -> last trade
    assert q.p[2] == pytest.approx(0.05) and np.isnan(q.spread[2])       # crossed/locked -> last trade
