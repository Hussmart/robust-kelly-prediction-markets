"""Polymarket collector: Gamma API (market metadata) + CLOB / data API (prices, trades).

Endpoints used (all public, no authentication):

* ``gamma-api.polymarket.com/markets``: market metadata, resolution, token ids.
* ``gamma-api.polymarket.com/events``: groups of markets (e.g. one FOMC meeting).
* ``clob.polymarket.com/prices-history``: YES-token price series. The endpoint rejects
  long ``[startTs, endTs]`` windows, so series are pulled in chunks of
  ``PRICE_CHUNK_SECONDS``.
* ``data-api.polymarket.com/trades``: executed trades. Pagination stops at an offset of
  10,000. Trades come newest-first, so when a window hits the cap we move the window's
  upper bound down to the oldest timestamp seen and continue (time-cursor pagination,
  ``_fetch_trades_window``). Boundary duplicates are removed during normalisation.

Privacy: the trades endpoint returns wallet addresses and user names. Those columns are
dropped at ingestion time and never written to disk (see ``_normalise_trades``).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.collectors.cache import HttpClient, ParquetCache

logger = logging.getLogger(__name__)

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
DATA_URL = "https://data-api.polymarket.com"

PRICE_CHUNK_SECONDS = 6 * 24 * 3600
TRADES_PAGE = 500
GAMMA_PAGE = 100
GAMMA_MAX_OFFSET = 2000
TRADES_MAX_OFFSET = 10_000

MARKET_COLUMNS = [
    "platform", "market_id", "condition_id", "question", "slug", "event_slug",
    "yes_token_id", "end_date", "closed_time", "outcome", "volume",
]
TRADE_COLUMNS = ["ts", "yes_price", "size", "notional", "yes_flow"]


def _parse_json_list(value: Any) -> list[Any]:
    """Gamma returns some list fields as JSON-encoded strings; decode either form."""
    if value is None:
        return []
    if isinstance(value, str):
        return json.loads(value)
    return list(value)


def _to_ts(value: Any) -> pd.Timestamp | None:
    """Parse a timestamp string into a tz-aware UTC ``Timestamp`` (``None`` if missing)."""
    if value in (None, ""):
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def normalise_market(raw: dict[str, Any], event_slug: str | None = None) -> dict[str, Any]:
    """Map a Gamma market payload to the project's common market schema.

    ``outcome`` is 1 if YES won, 0 if NO won, and NaN if the market is not (yet) cleanly
    resolved to a binary outcome.
    """
    outcomes = [str(o).lower() for o in _parse_json_list(raw.get("outcomes"))]
    prices = [float(p) for p in _parse_json_list(raw.get("outcomePrices"))]
    tokens = _parse_json_list(raw.get("clobTokenIds"))
    outcome = np.nan
    yes_token = None
    if outcomes == ["yes", "no"] and len(tokens) == 2:
        yes_token = str(tokens[0])
        resolved = raw.get("closed") and raw.get("umaResolutionStatus", "resolved") == "resolved"
        if resolved and len(prices) == 2 and sorted(prices) == [0.0, 1.0]:
            outcome = float(prices[0] == 1.0)
    events = raw.get("events") or []
    return {
        "platform": "polymarket",
        "market_id": str(raw["id"]),
        "condition_id": raw.get("conditionId"),
        "question": raw.get("question"),
        "slug": raw.get("slug"),
        "event_slug": event_slug or (events[0].get("slug") if events else None),
        "yes_token_id": yes_token,
        "end_date": _to_ts(raw.get("endDate")),
        "closed_time": _to_ts(raw.get("closedTime")),
        "outcome": outcome,
        "volume": float(raw.get("volume") or raw.get("volumeNum") or 0.0),
    }


def _normalise_trades(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert raw data-API trades to YES-denominated trades, dropping all user fields.

    Every trade is expressed from the YES side:

    * ``yes_price``: price of YES implied by the trade (``1 - p`` for a NO-token trade).
    * ``yes_flow``: signed taker notional, ``+`` when the taker added YES exposure
      (bought YES or sold NO) and ``-`` otherwise. Summing it over a window gives the
      order-flow imbalance used as a feature.
    """
    if not rows:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    df = pd.DataFrame(rows)
    key = ["transactionHash", "asset", "side", "size", "price", "timestamp"]
    df = df.drop_duplicates(subset=key + (["proxyWallet"] if "proxyWallet" in df else []))
    is_yes = df["outcomeIndex"].astype(int) == 0
    price = df["price"].astype(float)
    size = df["size"].astype(float)
    buy = df["side"].str.upper() == "BUY"
    sign = np.where(is_yes == buy, 1.0, -1.0)
    out = pd.DataFrame({
        "ts": df["timestamp"].astype("int64"),
        "yes_price": np.where(is_yes, price, 1.0 - price),
        "size": size,
        "notional": size * price,
    })
    out["yes_flow"] = sign * out["notional"]
    return out.sort_values("ts").reset_index(drop=True)[TRADE_COLUMNS]


class PolymarketCollector:
    """Cached client for Polymarket market metadata, price histories and trades."""

    def __init__(self, cache: ParquetCache | None = None, client: HttpClient | None = None) -> None:
        """Use the given cache/client or sensible defaults (5 req/s)."""
        self.cache = cache or ParquetCache()
        self.client = client or HttpClient(calls_per_second=5.0)

    # ------------------------------------------------------------------ metadata
    def get_market(self, market_id: str) -> dict[str, Any]:
        """Return normalised metadata for a single market id (cached)."""
        def fetch() -> pd.DataFrame:
            raw = self.client.get_json(f"{GAMMA_URL}/markets/{market_id}")
            return pd.DataFrame([normalise_market(raw)])
        row: dict[str, Any] = self.cache.get_or_fetch("polymarket/markets", str(market_id), fetch).iloc[0].to_dict()
        return row

    def get_event_markets(self, event_slug: str) -> pd.DataFrame:
        """Return all markets of an event (e.g. ``fed-decision-in-december``), cached."""
        def fetch() -> pd.DataFrame:
            events = self.client.get_json(f"{GAMMA_URL}/events", {"slug": event_slug})
            if not events:
                return pd.DataFrame(columns=MARKET_COLUMNS)
            return pd.DataFrame(
                [normalise_market(m, event_slug) for m in events[0].get("markets", [])],
                columns=MARKET_COLUMNS,
            )
        return self.cache.get_or_fetch("polymarket/events", event_slug, fetch)

    def list_resolved_markets(
        self,
        end_date_min: str,
        end_date_max: str,
        volume_min: float | None = 50_000.0,
        tag_slug: str | None = None,
        max_markets: int = 2_000,
        window_days: int | None = None,
    ) -> pd.DataFrame:
        """List resolved binary YES/NO markets ending in ``[end_date_min, end_date_max]``.

        Only markets with a clean 0/1 resolution are kept. ``volume_min`` filters on *lifetime*
        volume, which is future information relative to any decision taken before the market
        closes (a cheap market that later resolves YES trades at high prices and so accumulates
        more dollar volume). Pass ``volume_min=None`` for studies that must not select on it.
        ``window_days`` switches from monthly to fixed-length listing windows (needed to enumerate
        an unfiltered population, because the API rejects offsets beyond ~2,000 per window).
        """
        vol_key = "none" if volume_min is None else f"{volume_min:.0f}"
        key = f"{end_date_min}_{end_date_max}_{vol_key}_{tag_slug}_{max_markets}_{window_days}"

        def fetch() -> pd.DataFrame:
            # The API rejects offsets beyond ~2,000 (HTTP 422), so walk month by month.
            freq = "MS" if window_days is None else f"{window_days}D"
            edges = pd.date_range(end_date_min, end_date_max, freq=freq, tz="UTC").tolist()
            edges = [pd.Timestamp(end_date_min, tz="UTC")] + [e for e in edges if e > pd.Timestamp(end_date_min, tz="UTC")]
            edges.append(pd.Timestamp(end_date_max, tz="UTC") + pd.Timedelta(days=1))
            rows: list[dict[str, Any]] = []
            for lo, hi in zip(edges[:-1], edges[1:], strict=True):
                offset = 0
                while offset < GAMMA_MAX_OFFSET:
                    params: dict[str, Any] = {
                        "closed": "true", "limit": GAMMA_PAGE, "offset": offset,
                        "end_date_min": lo.strftime("%Y-%m-%d"), "end_date_max": hi.strftime("%Y-%m-%d"),
                        "order": "volumeNum", "ascending": "false",
                    }
                    if volume_min is not None:
                        params["volume_num_min"] = volume_min
                    if tag_slug:
                        params["tag_slug"] = tag_slug
                    try:
                        page = self.client.get_json(f"{GAMMA_URL}/markets", params)
                    except requests.HTTPError as exc:
                        if exc.response is not None and exc.response.status_code == 422:
                            break  # offset cap reached for this window
                        raise
                    if not page:  # the API caps the page size, so only an empty page ends a window
                        break
                    rows.extend(normalise_market(m) for m in page)
                    offset += len(page)
            df = pd.DataFrame(rows, columns=MARKET_COLUMNS)
            df = df.dropna(subset=["outcome", "yes_token_id"]).drop_duplicates("market_id")
            return df.sort_values("volume", ascending=False).head(max_markets).reset_index(drop=True)

        return self.cache.get_or_fetch("polymarket/resolved_lists", key, fetch)

    # ------------------------------------------------------------------ prices
    def price_history(self, yes_token_id: str, start_ts: int, end_ts: int, fidelity: int = 60) -> pd.DataFrame:
        """YES price series on ``[start_ts, end_ts]`` at ``fidelity`` minutes, as ``ts, price``."""
        key = f"{yes_token_id}_{start_ts}_{end_ts}_{fidelity}"

        def fetch() -> pd.DataFrame:
            parts = []
            lo = start_ts
            while lo < end_ts:
                hi = min(lo + PRICE_CHUNK_SECONDS, end_ts)
                data = self.client.get_json(
                    f"{CLOB_URL}/prices-history",
                    {"market": yes_token_id, "startTs": lo, "endTs": hi, "fidelity": fidelity},
                )
                parts.extend(data.get("history", []))
                lo = hi
            df = pd.DataFrame(parts, columns=["t", "p"]).rename(columns={"t": "ts", "p": "price"})
            df = df.astype({"ts": "int64", "price": "float64"})
            return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

        return self.cache.get_or_fetch("polymarket/prices", key, fetch)

    # ------------------------------------------------------------------ trades
    def _fetch_trades_window(self, condition_id: str, start: int, end: int) -> list[dict[str, Any]]:
        """Fetch all taker trades in ``[start, end]``, paging past the offset cap by time."""
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = self.client.get_json(
                f"{DATA_URL}/trades",
                {"market": condition_id, "start": start, "end": end,
                 "limit": TRADES_PAGE, "offset": offset, "takerOnly": "true"},
            )
            rows.extend(page)
            if len(page) < TRADES_PAGE:
                return rows
            offset += TRADES_PAGE
            if offset >= TRADES_MAX_OFFSET:
                oldest = min(int(r["timestamp"]) for r in rows)
                if oldest >= end:  # >10k trades within one second: cannot page further
                    logger.warning("trade window ending %s truncated at offset cap", end)
                    return rows
                end, offset = oldest, 0

    def trades(self, condition_id: str, start_ts: int, end_ts: int) -> pd.DataFrame:
        """YES-denominated taker trades on ``[start_ts, end_ts]`` (user fields removed)."""
        key = f"{condition_id}_{start_ts}_{end_ts}"
        return self.cache.get_or_fetch(
            "polymarket/trades", key,
            lambda: _normalise_trades(self._fetch_trades_window(condition_id, start_ts, end_ts)),
        )
