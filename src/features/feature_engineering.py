"""Per-snapshot features: implied probability, spread, order-flow imbalance, volume, time-to-close, volatility.

For every mapped pair we build a panel of snapshots on a regular grid
``t_k = t_end - k * step`` over the ``window_days`` before the event. Every feature at
``t_k`` uses only data with timestamp ``<= t_k`` (no look-ahead). Notation follows
``docs/methodology.md`` Section 2:

* ``p_poly``, ``p_kalshi``: implied YES probability on each venue. Kalshi uses the quote
  mid ``(bid + ask) / 2`` when a valid two-sided quote exists, otherwise its last trade.
  Polymarket uses the CLOB price series.
* ``div = p_poly - p_kalshi`` and ``logit_div = logit(p_poly) - logit(p_kalshi)``.
* ``spread_kalshi``: quoted spread ``ask - bid``.
* ``roll_spread_{poly,kalshi}``: Roll (1984) effective-spread estimate from trailing
  trades, ``s = 2 * sqrt(-Cov(dp_t, dp_{t-1}))`` (0 when the covariance is positive).
  Polymarket has no public historical order book, so this is our only spread measure
  there. On Kalshi it can be validated against the quoted spread.
* ``volume_{venue}``: trailing ``flow_window_h`` taker notional (USD).
* ``ofi_{venue}``: order-flow imbalance ``sum(signed notional) / sum(notional)`` in [-1, 1],
  defined as 0 when there is no volume.
* ``vol_{venue}``: realised volatility, the std of hourly logit-price changes over
  ``vol_window_h``.
* ``hours_to_close`` and ``stale_{venue}_h`` (hours since the last trade).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 0.005  # probability clip for logits; 0.5c is below one tick on both venues
HOUR = 3600


def logit(p: np.ndarray | pd.Series | float, eps: float = EPS) -> np.ndarray:
    """Clipped log-odds ``log(p / (1 - p))`` with ``p`` clipped to ``[eps, 1 - eps]``."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def snapshot_grid(start_ts: int, end_ts: int, step_h: int) -> np.ndarray:
    """Snapshot times ``end_ts - k*step`` that are ``>= start_ts``, in ascending order."""
    n = (end_ts - start_ts) // (step_h * HOUR)
    return end_ts - np.arange(n, -1, -1) * step_h * HOUR


def asof(grid: np.ndarray, ts: np.ndarray, values: np.ndarray, max_age_h: float) -> np.ndarray:
    """Last ``values`` observed at or before each grid time; NaN if older than ``max_age_h``.

    ``ts`` must be sorted ascending. NaN entries in ``values`` are skipped, so the result
    is the last *valid* observation.
    """
    ts = np.asarray(ts, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    ok = ~np.isnan(values)
    ts, values = ts[ok], values[ok]
    out = np.full(len(grid), np.nan)
    if len(ts) == 0:
        return out
    idx = np.searchsorted(ts, grid, side="right") - 1
    has = idx >= 0
    age = np.where(has, (grid - ts[np.maximum(idx, 0)]) / HOUR, np.inf)
    keep = has & (age <= max_age_h)
    out[keep] = values[idx[keep]]
    return out


def kalshi_implied_price(candles: pd.DataFrame, max_spread: float = 0.5) -> pd.DataFrame:
    """Kalshi YES probability per candle: quote mid if a valid 2-sided quote exists, else last trade.

    A quote is valid when ``0 < bid < ask < 1`` and ``ask - bid <= max_spread``. Returns
    ``ts, p, spread`` where ``spread`` is NaN for invalid quotes.
    """
    bid, ask = candles["yes_bid"].to_numpy(float), candles["yes_ask"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        valid = (bid > 0) & (ask < 1) & (ask > bid) & (ask - bid <= max_spread)
    mid = np.where(valid, (bid + ask) / 2.0, candles["price"].to_numpy(float))
    return pd.DataFrame({
        "ts": candles["ts"].to_numpy(np.int64),
        "p": mid,
        "spread": np.where(valid, ask - bid, np.nan),
    })


def roll_spread(prices: np.ndarray) -> float:
    """Roll (1984) effective-spread estimator ``2*sqrt(-Cov(dp_t, dp_{t-1}))``.

    Bid-ask bounce makes consecutive trade-price changes negatively autocorrelated. With
    a constant half-spread ``s/2`` and uncorrelated efficient-price innovations,
    ``Cov(dp_t, dp_{t-1}) = -s^2/4``. Returns 0 when the sample covariance is
    non-negative and NaN when fewer than 3 price changes are available.
    """
    dp = np.diff(np.asarray(prices, dtype=float))
    if len(dp) < 3:
        return np.nan
    cov = np.cov(dp[1:], dp[:-1])[0, 1]
    return 2.0 * np.sqrt(-cov) if cov < 0 else 0.0


def trailing_trade_features(
    grid: np.ndarray, trades: pd.DataFrame, window_h: float, roll_max_trades: int = 2_000,
) -> pd.DataFrame:
    """Trailing-window trade statistics at every grid time.

    Returns columns ``volume`` (taker notional), ``ofi`` (order-flow imbalance, 0 without
    volume), ``n_trades``, ``roll_spread`` (from the most recent ``roll_max_trades``
    trades in the window) and ``stale_h`` (hours since the last trade, NaN if none yet).
    """
    ts = trades["ts"].to_numpy(np.int64)
    notional = trades["notional"].to_numpy(float)
    flow = trades["yes_flow"].to_numpy(float)
    price = trades["yes_price"].to_numpy(float)
    cum_n = np.concatenate([[0.0], np.cumsum(notional)])
    cum_f = np.concatenate([[0.0], np.cumsum(flow)])
    hi = np.searchsorted(ts, grid, side="right")
    lo = np.searchsorted(ts, grid - int(window_h * HOUR), side="right")
    volume = cum_n[hi] - cum_n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        ofi = np.where(volume > 0, (cum_f[hi] - cum_f[lo]) / volume, 0.0)
    roll = np.array([roll_spread(price[max(l, h - roll_max_trades):h]) for l, h in zip(lo, hi)])
    stale = np.full(len(grid), np.nan)
    has = hi > 0
    stale[has] = (grid[has] - ts[hi[has] - 1]) / HOUR
    return pd.DataFrame({
        "volume": volume, "ofi": np.clip(ofi, -1.0, 1.0), "n_trades": hi - lo,
        "roll_spread": roll, "stale_h": stale,
    })


def realised_vol(grid: np.ndarray, ts: np.ndarray, p: np.ndarray, window_h: float) -> np.ndarray:
    """Std of hourly logit-price changes over the trailing ``window_h`` at each grid time."""
    hourly_grid = np.arange(grid[0] - int(window_h * HOUR), grid[-1] + 1, HOUR)
    hourly = pd.Series(logit(asof(hourly_grid, ts, p, max_age_h=np.inf)), index=hourly_grid)
    dl = hourly.diff()
    w = int(window_h)
    rolled = dl.rolling(w, min_periods=max(3, w // 3)).std()
    return rolled.reindex(grid).to_numpy()


def build_pair_panel(
    pair: pd.Series,
    start_ts: int,
    end_ts: int,
    poly_prices: pd.DataFrame,
    poly_trades: pd.DataFrame,
    kalshi_candles: pd.DataFrame,
    kalshi_trades: pd.DataFrame,
    step_h: int = 6,
    flow_window_h: float = 24.0,
    vol_window_h: float = 72.0,
    max_price_age_h: float = 48.0,
) -> pd.DataFrame:
    """Build the snapshot panel for one pair.

    Rows where either venue has no price newer than ``max_price_age_h`` are dropped (the
    market did not exist yet, or quotes were stale). Venues with no trading in a window
    get ``volume = 0`` and ``ofi = 0`` rather than NaN.
    """
    grid = snapshot_grid(start_ts + int(vol_window_h * HOUR), end_ts - HOUR, step_h)
    kq = kalshi_implied_price(kalshi_candles) if len(kalshi_candles) else pd.DataFrame(
        {"ts": np.array([], np.int64), "p": [], "spread": []})
    p_poly = asof(grid, poly_prices["ts"].to_numpy(), poly_prices["price"].to_numpy(), max_price_age_h)
    p_kalshi = asof(grid, kq["ts"].to_numpy(), kq["p"].to_numpy(), max_price_age_h)
    panel = pd.DataFrame({
        "pair_id": pair["pair_id"],
        "meeting": pair["meeting"],
        "bucket": pair["bucket"],
        "ts": grid,
        "hours_to_close": (end_ts - grid) / HOUR,
        "p_poly": p_poly,
        "p_kalshi": p_kalshi,
        "spread_kalshi": asof(grid, kq["ts"].to_numpy(), kq["spread"].to_numpy(), step_h),
        "vol_poly": realised_vol(grid, poly_prices["ts"].to_numpy(), poly_prices["price"].to_numpy(), vol_window_h),
        "vol_kalshi": realised_vol(grid, kq["ts"].to_numpy(), kq["p"].to_numpy(), vol_window_h),
        "outcome": pair["outcome"],
    })
    for venue, trades in (("poly", poly_trades), ("kalshi", kalshi_trades)):
        tf = trailing_trade_features(grid, trades, flow_window_h)
        panel[f"volume_{venue}"] = tf["volume"].to_numpy()
        panel[f"ofi_{venue}"] = tf["ofi"].to_numpy()
        panel[f"n_trades_{venue}"] = tf["n_trades"].to_numpy()
        panel[f"roll_spread_{venue}"] = tf["roll_spread"].to_numpy()
        panel[f"stale_{venue}_h"] = tf["stale_h"].to_numpy()
    panel["div"] = panel["p_poly"] - panel["p_kalshi"]
    panel["abs_div"] = panel["div"].abs()
    panel["logit_div"] = logit(panel["p_poly"]) - logit(panel["p_kalshi"])
    panel["log_volume_min"] = np.log1p(np.minimum(panel["volume_poly"], panel["volume_kalshi"]))
    return panel.dropna(subset=["p_poly", "p_kalshi"]).reset_index(drop=True)


def add_persistence_label(
    panel: pd.DataFrame, horizon_h: float = 24.0, min_div: float = 0.02, keep_frac: float = 0.5,
) -> pd.DataFrame:
    """Label whether a cross-venue divergence is *real* (persistent) or *noise* (transient).

    For a snapshot with ``|div_t| >= min_div``, the divergence is labelled persistent
    (``1``) when, ``horizon_h`` later, it still has the same sign and at least
    ``keep_frac`` of its size: ``sign(div_{t+h}) = sign(div_t)`` and
    ``|div_{t+h}| >= keep_frac * |div_t|``. Snapshots below ``min_div`` or without a
    snapshot ``horizon_h`` ahead get NaN. The label looks forward and is only ever used as
    a *target*, never as a feature.
    """
    out = panel.copy()
    out["persistent"] = np.nan
    for _, g in out.groupby("pair_id"):
        future = g.set_index("ts")["div"]
        fut = future.reindex(g["ts"].to_numpy() + int(horizon_h * HOUR)).to_numpy()
        cur = g["div"].to_numpy()
        eligible = (np.abs(cur) >= min_div) & ~np.isnan(fut)
        lab = (np.sign(fut) == np.sign(cur)) & (np.abs(fut) >= keep_frac * np.abs(cur))
        out.loc[g.index, "persistent"] = np.where(eligible, lab.astype(float), np.nan)
    return out
