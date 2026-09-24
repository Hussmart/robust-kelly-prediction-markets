"""Collect the raw time series needed for every mapped Polymarket <-> Kalshi pair."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.collectors.kalshi import KalshiCollector
from src.collectors.polymarket import PolymarketCollector

logger = logging.getLogger(__name__)

MAPPING_PATH = Path(__file__).resolve().parents[2] / "data" / "mappings" / "fomc_pairs.csv"


@dataclass
class PairData:
    """Raw inputs for one pair over ``[start_ts, end_ts]`` (Unix seconds)."""

    pair: pd.Series
    start_ts: int
    end_ts: int
    poly_prices: pd.DataFrame
    poly_trades: pd.DataFrame
    kalshi_candles: pd.DataFrame
    kalshi_trades: pd.DataFrame


def load_mapping(path: Path = MAPPING_PATH) -> pd.DataFrame:
    """Read the curated pair table with ``event_time`` parsed as UTC."""
    df = pd.read_csv(path, dtype={"poly_market_id": str})
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    return df


def collect_pair(
    pair: pd.Series,
    pm: PolymarketCollector,
    ks: KalshiCollector,
    window_days: int = 21,
) -> PairData:
    """Fetch (or read from cache) both platforms' series for the ``window_days`` before the event.

    The window ends at ``pair.event_time`` (Kalshi's trading close, minutes before the FOMC
    announcement), so no downstream feature can observe the decision.
    """
    end_ts = int(pair.event_time.timestamp())
    start_ts = end_ts - window_days * 86400
    poly = pm.get_event_markets(pair.poly_event_slug).set_index("market_id").loc[pair.poly_market_id]
    kalshi_close = pair.event_time
    return PairData(
        pair=pair,
        start_ts=start_ts,
        end_ts=end_ts,
        poly_prices=pm.price_history(poly.yes_token_id, start_ts, end_ts),
        poly_trades=pm.trades(poly.condition_id, start_ts, end_ts),
        kalshi_candles=ks.candlesticks(pair.kalshi_ticker, start_ts, end_ts, close_time=kalshi_close),
        kalshi_trades=ks.trades(pair.kalshi_ticker, start_ts, end_ts, close_time=kalshi_close),
    )
