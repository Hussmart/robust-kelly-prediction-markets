"""A broad universe of resolved Polymarket binary markets with a look-ahead-safe entry price.

Purpose: the FOMC panel has ~1 bet per round, which cannot compare allocation methods. This
universe has thousands of resolved markets across topics, so a backtest can hold ~10
concurrent, roughly independent bets per round and actually experience losses.

Cohort construction (fixed calendar, and only ex-ante information):

* Decision dates ``T_k = T_0 + k * 14 days``.
* A market belongs to cohort ``k`` iff its scheduled ``end_date`` satisfies
  ``T_k + 3d <= end_date < T_k + 14d``. The scheduled end date is known when the bet is
  placed. The *actual* resolution time is not: markets that resolve early are
  disproportionately YES, so selecting on it would leak the outcome into the universe.
* The market must still be open at ``T_k`` (``closed_time > T_k``, which is observable at
  ``T_k``), must have traded actively (an ex-ante activity filter on the number of price changes
  before ``T_k``) and must not have an extreme entry price (``[price_lo, price_hi]``; the outcome
  is usually already known). Nothing is filtered on lifetime volume: it is future information,
  and conditioning on it selects markets that later resolve YES (a cheap market that goes on to
  resolve YES trades at high prices and so accumulates more dollar volume).
* The population is *all* resolved binary markets ending in the study period, from which a
  uniform random sample (fixed seed, independent of outcomes) is drawn to bound API cost.
* Bets settle at the actual resolution time. That is usually before ``T_{k+1}``; the share
  that resolves later (a capital overlap the backtest ignores) is reported as
  ``settles_after_next``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.collectors.polymarket import PolymarketCollector

logger = logging.getLogger(__name__)

DAY = 86_400
T0 = pd.Timestamp("2024-07-01", tz="UTC")
COHORT_DAYS = 14
MIN_HORIZON_DAYS = 3


def assign_cohorts(markets: pd.DataFrame) -> pd.DataFrame:
    """Add ``cohort``, ``decision_ts``, ``horizon_days`` and ``settles_after_next`` to ``markets``.

    Cohorts use the scheduled ``end_date`` only. Markets whose scheduled end falls in the
    first three days after a decision date, that had already closed by the decision date, or that
    have no end date are dropped.
    """
    m = markets.copy()
    m["end_date"] = pd.to_datetime(m["end_date"], utc=True)
    m["closed_time"] = pd.to_datetime(m["closed_time"], utc=True)
    m = m.dropna(subset=["end_date"])
    days = (m["end_date"] - T0) / pd.Timedelta(days=1)
    m["cohort"] = np.floor(days / COHORT_DAYS).astype(int)
    in_window = (days % COHORT_DAYS) >= MIN_HORIZON_DAYS  # ends 3-14 days after T_k
    m = m[(m["cohort"] >= 0) & in_window].copy()
    m["decision_ts"] = (int(T0.timestamp()) + m["cohort"] * COHORT_DAYS * DAY).astype("int64")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    end_ts = (m["end_date"] - epoch) // pd.Timedelta(seconds=1)
    closed_ts = (m["closed_time"] - epoch) // pd.Timedelta(seconds=1)
    m["horizon_days"] = (end_ts - m["decision_ts"]) / DAY
    m = m[closed_ts.isna() | (closed_ts > m["decision_ts"])].copy()  # still open at T_k
    closed_ts = closed_ts.reindex(m.index)
    m["settles_after_next"] = (closed_ts >= m["decision_ts"] + COHORT_DAYS * DAY).fillna(False)
    return m


def entry_features(
    pm: PolymarketCollector, yes_token_id: str, decision_ts: int, lookback_h: float = 144.0,
) -> dict[str, float]:
    """Pre-decision price information for one market, using data at or before ``decision_ts`` only.

    The CLOB history is a regular hourly grid that repeats the last price when nothing trades, so
    activity has to be read from *changes*. Returns ``price`` (last price at or before the decision
    date, NaN if none), ``n_changes`` (number of hourly price changes in the ``lookback_h`` hours
    before the decision) and ``price_age_h`` (hours since the last change; ``lookback_h`` if the
    price did not move at all). One API call (the window is within the CLOB history limit).
    """
    hist = pm.price_history(yes_token_id, decision_ts - int(lookback_h * 3600), decision_ts, fidelity=60)
    hist = hist[hist["ts"] <= decision_ts]
    if hist.empty:
        return {"price": float("nan"), "price_age_h": float("nan"), "n_changes": 0.0}
    moved = hist["price"].diff().fillna(0.0).to_numpy() != 0
    last_change = int(hist["ts"].to_numpy()[moved][-1]) if moved.any() else None
    age = (decision_ts - last_change) / 3600.0 if last_change is not None else float(lookback_h)
    return {"price": float(hist["price"].iloc[-1]), "price_age_h": age, "n_changes": float(moved.sum())}


def sample_markets(markets: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Uniform random sample of ``n`` markets (all of them if fewer), independent of outcomes."""
    if len(markets) <= n:
        return markets
    return markets.sample(n=n, random_state=seed).sort_index()


def build_universe(
    pm: PolymarketCollector,
    start: str = "2024-07-01",
    end: str = "2026-08-31",
    sample_size: int = 12_000,
    seed: int = 0,
    price_lo: float = 0.03,
    price_hi: float = 0.97,
    min_changes: int = 10,
    max_price_age_h: float = 24.0,
    progress_every: int = 500,
) -> pd.DataFrame:
    """List all resolved binary markets, sample, assign cohorts and attach ex-ante entry features.

    Selection uses only information available at the decision date: the scheduled end date, that
    the market is open, that it has traded actively (at least ``min_changes`` hourly price changes in
    the previous 6 days and a last change at most ``max_price_age_h`` hours old) and that the price
    is not extreme. There is
    no filter on lifetime volume and none on the outcome.
    """
    listed = pm.list_resolved_markets(start, end, volume_min=None, max_markets=10**7, window_days=7)
    m = sample_markets(assign_cohorts(listed), sample_size, seed)
    feats = []
    for i, row in enumerate(m.itertuples(index=False), 1):
        try:
            feats.append(entry_features(pm, row.yes_token_id, int(row.decision_ts)))
        except Exception as exc:  # noqa: BLE001 - one bad market must not abort a long pull
            logger.warning("price pull failed for %s: %s", row.market_id, exc)
            feats.append({"price": float("nan"), "price_age_h": float("nan"), "n_changes": 0.0})
        if i % progress_every == 0:
            logger.warning("entry features: %d / %d", i, len(m))
    m = pd.concat([m.reset_index(drop=True), pd.DataFrame(feats)], axis=1).dropna(subset=["price"])
    m = m[
        (m["price"] >= price_lo) & (m["price"] <= price_hi) & (m["n_changes"] >= min_changes)
        & (m["price_age_h"] <= max_price_age_h)
    ].copy()
    m["event_key"] = m["event_slug"].fillna(m["market_id"])
    keep = ["market_id", "event_key", "cohort", "decision_ts", "horizon_days", "price", "price_age_h",
            "n_changes", "outcome", "settles_after_next"]
    return m[keep].reset_index(drop=True)
