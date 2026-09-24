"""Kalshi collector: public REST API v2 (markets, candlesticks, trades), live + historical tiers.

Kalshi splits its public data into two tiers. Markets settled before a rolling cutoff
(``GET /historical/cutoff``) are served only from ``/historical/...`` endpoints. Newer
markets are served from the regular endpoints. The collector tries the tier implied by
the market's settlement time and falls back to the other tier on HTTP 404 or an empty
result, so callers never need to know which tier holds a market.

The two tiers also use different field names for the same quantity (``close`` vs
``close_dollars``, ``volume`` vs ``volume_fp``). ``_normalise_candles`` maps both to one
schema.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests

from src.collectors.cache import HttpClient, ParquetCache

logger = logging.getLogger(__name__)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
CANDLE_CHUNK_SECONDS = 20 * 24 * 3600
TRADES_PAGE = 1000

MARKET_COLUMNS = [
    "platform", "market_id", "event_ticker", "series_ticker", "question", "yes_sub_title",
    "open_time", "close_time", "outcome", "volume",
]
CANDLE_COLUMNS = ["ts", "price", "yes_bid", "yes_ask", "volume", "open_interest"]
TRADE_COLUMNS = ["ts", "yes_price", "size", "notional", "yes_flow"]


def _num(d: dict[str, Any] | None, *keys: str) -> float:
    """Return the first present key of ``d`` as float (NaN if none present / null)."""
    if not d:
        return np.nan
    for k in keys:
        if d.get(k) not in (None, ""):
            return float(d[k])
    return np.nan


def normalise_market(raw: dict[str, Any]) -> dict[str, Any]:
    """Map a Kalshi market payload (either tier) to the common market schema."""
    result = raw.get("result")
    outcome = 1.0 if result == "yes" else 0.0 if result == "no" else np.nan
    event = raw.get("event_ticker", "")
    return {
        "platform": "kalshi",
        "market_id": raw["ticker"],
        "event_ticker": event,
        "series_ticker": event.split("-")[0] if event else None,
        "question": raw.get("title"),
        "yes_sub_title": raw.get("yes_sub_title"),
        "open_time": pd.Timestamp(raw["open_time"]) if raw.get("open_time") else None,
        "close_time": pd.Timestamp(raw["close_time"]) if raw.get("close_time") else None,
        "outcome": outcome,
        "volume": _num(raw, "volume_fp", "volume"),
    }


def _normalise_candles(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten candlesticks from either tier to ``ts, price, yes_bid, yes_ask, volume, open_interest``.

    ``price`` is the last traded price in the period (NaN if nothing traded). Bid/ask are
    the closing top-of-book quotes, which exist even in periods without trades.
    """
    out = [{
        "ts": int(c["end_period_ts"]),
        "price": _num(c.get("price"), "close_dollars", "close"),
        "yes_bid": _num(c.get("yes_bid"), "close_dollars", "close"),
        "yes_ask": _num(c.get("yes_ask"), "close_dollars", "close"),
        "volume": _num(c, "volume_fp", "volume"),
        "open_interest": _num(c, "open_interest_fp", "open_interest"),
    } for c in rows]
    df = pd.DataFrame(out, columns=CANDLE_COLUMNS)
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def _epoch_seconds(values: pd.Series) -> pd.Series:
    """ISO-8601 strings -> integer Unix seconds, independent of pandas' datetime resolution."""
    dt = pd.to_datetime(values, utc=True, format="ISO8601")
    return ((dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).astype("int64")


def _normalise_trades(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert raw trades to YES-denominated trades with signed taker flow.

    ``yes_flow`` is ``+notional`` when the taker bought YES and ``-notional`` when the
    taker bought NO, the same sign convention as the Polymarket collector.
    """
    if not rows:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    df = pd.DataFrame(rows).drop_duplicates("trade_id")
    yes_price = df["yes_price_dollars"].astype(float)
    no_price = df["no_price_dollars"].astype(float)
    size = df["count_fp"].astype(float)
    taker_yes = df["taker_side"] == "yes"
    notional = size * np.where(taker_yes, yes_price, no_price)
    out = pd.DataFrame({
        "ts": _epoch_seconds(df["created_time"]),
        "yes_price": yes_price,
        "size": size,
        "notional": notional,
        "yes_flow": np.where(taker_yes, 1.0, -1.0) * notional,
    })
    return out.sort_values("ts").reset_index(drop=True)[TRADE_COLUMNS]


class KalshiCollector:
    """Cached client for Kalshi market metadata, hourly candlesticks and trades."""

    def __init__(self, cache: ParquetCache | None = None, client: HttpClient | None = None) -> None:
        """Use the given cache/client or defaults (Kalshi's public tier allows ~10 req/s)."""
        self.cache = cache or ParquetCache()
        self.client = client or HttpClient(calls_per_second=8.0)
        self._cutoff: pd.Timestamp | None = None

    @property
    def historical_cutoff(self) -> pd.Timestamp:
        """Settlement time before which markets live only in the historical tier."""
        if self._cutoff is None:
            data = self.client.get_json(f"{BASE_URL}/historical/cutoff")
            self._cutoff = pd.Timestamp(data["market_settled_ts"])
        return self._cutoff

    def _tiered(
        self, historical_first: bool, live: Callable[[], Any], hist: Callable[[], Any],
        is_empty: Callable[[Any], bool] = lambda r: not r,
    ) -> Any:
        """Call the preferred tier; on HTTP 404 or an empty result, fall back to the other.

        If the preferred tier returned an empty result and the other tier 404s, the empty
        result is returned (e.g. a window with no candles). A 404 from both tiers is raised.
        """
        order = (hist, live) if historical_first else (live, hist)
        result = None
        try:
            result = order[0]()
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 404:
                raise
        if result is not None and not is_empty(result):
            return result
        try:
            return order[1]()
        except requests.HTTPError as exc:
            if result is None or exc.response is None or exc.response.status_code != 404:
                raise
            return result

    def _is_historical(self, close_time: pd.Timestamp | None) -> bool:
        """Guess the tier from the close time (the 404 fallback corrects wrong guesses)."""
        return close_time is not None and close_time < self.historical_cutoff

    # ------------------------------------------------------------------ metadata
    def get_market(self, ticker: str) -> dict[str, Any]:
        """Return normalised metadata for one market ticker (cached)."""
        def fetch() -> pd.DataFrame:
            raw = self._tiered(
                True,
                lambda: self.client.get_json(f"{BASE_URL}/markets/{ticker}"),
                lambda: self.client.get_json(f"{BASE_URL}/historical/markets/{ticker}"),
            )
            return pd.DataFrame([normalise_market(raw["market"])])
        return self.cache.get_or_fetch("kalshi/markets", ticker, fetch).iloc[0].to_dict()

    def _paginate(self, url: str, params: dict[str, Any], key: str) -> list[dict[str, Any]]:
        """Follow Kalshi's cursor pagination, concatenating ``key`` lists."""
        rows: list[dict[str, Any]] = []
        cursor = None
        while True:
            p = dict(params, cursor=cursor) if cursor else dict(params)
            data = self.client.get_json(url, p)
            rows.extend(data.get(key, []))
            cursor = data.get("cursor")
            if not cursor or not data.get(key):
                return rows

    def list_settled_markets(self, series_ticker: str) -> pd.DataFrame:
        """All settled markets of a series across both tiers (e.g. ``KXFEDDECISION``)."""
        def fetch() -> pd.DataFrame:
            hist = self._paginate(f"{BASE_URL}/historical/markets",
                                  {"series_ticker": series_ticker, "limit": 1000}, "markets")
            live = self._paginate(f"{BASE_URL}/markets",
                                  {"series_ticker": series_ticker, "status": "settled", "limit": 1000},
                                  "markets")
            df = pd.DataFrame([normalise_market(m) for m in hist + live], columns=MARKET_COLUMNS)
            return df.drop_duplicates("market_id").dropna(subset=["outcome"]).reset_index(drop=True)
        return self.cache.get_or_fetch("kalshi/series", series_ticker, fetch)

    # ------------------------------------------------------------------ candles
    def candlesticks(
        self, ticker: str, start_ts: int, end_ts: int, period_minutes: int = 60,
        close_time: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Candlesticks on ``[start_ts, end_ts]`` with ``period_minutes`` in {1, 60, 1440}."""
        key = f"{ticker}_{start_ts}_{end_ts}_{period_minutes}"
        series = ticker.split("-")[0]

        def fetch() -> pd.DataFrame:
            rows: list[dict[str, Any]] = []
            lo = start_ts
            while lo < end_ts:
                hi = min(lo + CANDLE_CHUNK_SECONDS, end_ts)
                params = {"start_ts": lo, "end_ts": hi, "period_interval": period_minutes}
                data = self._tiered(
                    self._is_historical(close_time),
                    lambda: self.client.get_json(
                        f"{BASE_URL}/series/{series}/markets/{ticker}/candlesticks", params),
                    lambda: self.client.get_json(
                        f"{BASE_URL}/historical/markets/{ticker}/candlesticks", params),
                    is_empty=lambda d: not d.get("candlesticks"),
                )
                rows.extend(data.get("candlesticks", []))
                lo = hi
            return _normalise_candles(rows)

        return self.cache.get_or_fetch("kalshi/candles", key, fetch)

    # ------------------------------------------------------------------ trades
    def trades(
        self, ticker: str, start_ts: int, end_ts: int, close_time: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """YES-denominated taker trades on ``[start_ts, end_ts]``."""
        key = f"{ticker}_{start_ts}_{end_ts}"
        params = {"ticker": ticker, "min_ts": start_ts, "max_ts": end_ts, "limit": TRADES_PAGE}

        def fetch() -> pd.DataFrame:
            rows = self._tiered(
                self._is_historical(close_time),
                lambda: self._paginate(f"{BASE_URL}/markets/trades", params, "trades"),
                lambda: self._paginate(f"{BASE_URL}/historical/trades", params, "trades"),
            )
            return _normalise_trades(rows)

        return self.cache.get_or_fetch("kalshi/trades", key, fetch)
