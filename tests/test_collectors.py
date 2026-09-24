"""Offline tests for the collectors: schema normalisation, sign conventions, tier fallback, cache."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import requests

from src.collectors import kalshi, polymarket
from src.collectors.cache import ParquetCache, _safe_key


# ---------------------------------------------------------------- cache
def test_cache_roundtrip_and_fetch_called_once(tmp_path):
    cache = ParquetCache(tmp_path)
    calls = []

    def fetch() -> pd.DataFrame:
        calls.append(1)
        return pd.DataFrame({"a": [1, 2]})

    first = cache.get_or_fetch("ns", "key", fetch)
    second = cache.get_or_fetch("ns", "key", fetch)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)


def test_safe_key_is_short_and_unique():
    a, b = "x" * 200 + "a", "x" * 200 + "b"
    assert len(_safe_key(a)) <= 80 and _safe_key(a) != _safe_key(b)
    assert "/" not in _safe_key("a/b:c")


# ---------------------------------------------------------------- polymarket
def _pm_market(**overrides: Any) -> dict[str, Any]:
    raw = {
        "id": "1", "conditionId": "0xabc", "question": "Q?", "slug": "q",
        "outcomes": '["Yes", "No"]', "outcomePrices": '["1", "0"]',
        "clobTokenIds": '["111", "222"]', "closed": True, "umaResolutionStatus": "resolved",
        "endDate": "2025-01-01T00:00:00Z", "closedTime": "2025-01-01 12:00:00+00", "volume": "10",
    }
    raw.update(overrides)
    return raw


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, 1.0),
        ({"outcomePrices": '["0", "1"]'}, 0.0),
        ({"outcomePrices": '["0.5", "0.5"]'}, np.nan),        # 50/50 or unresolved
        ({"closed": False}, np.nan),
        ({"outcomes": '["Trump", "Harris"]'}, np.nan),          # not a YES/NO market
    ],
)
def test_polymarket_outcome_parsing(overrides, expected):
    m = polymarket.normalise_market(_pm_market(**overrides))
    if np.isnan(expected):
        assert np.isnan(m["outcome"])
    else:
        assert m["outcome"] == expected
        assert m["yes_token_id"] == "111"
    assert m["closed_time"].tzinfo is not None


def test_polymarket_trades_yes_convention_and_pii_dropped():
    base = {"transactionHash": "h", "asset": "a", "timestamp": 100, "size": 10.0,
            "proxyWallet": "0xdead", "name": "alice", "pseudonym": "p", "bio": "b"}
    rows = [
        dict(base, outcomeIndex=0, side="BUY", price=0.6, transactionHash="1"),   # +YES
        dict(base, outcomeIndex=0, side="SELL", price=0.6, transactionHash="2"),  # -YES
        dict(base, outcomeIndex=1, side="BUY", price=0.4, transactionHash="3"),   # buy NO = -YES
        dict(base, outcomeIndex=1, side="SELL", price=0.4, transactionHash="4"),  # sell NO = +YES
        dict(base, outcomeIndex=1, side="SELL", price=0.4, transactionHash="4"),  # duplicate
    ]
    df = polymarket._normalise_trades(rows)
    assert list(df.columns) == polymarket.TRADE_COLUMNS
    assert len(df) == 4
    np.testing.assert_allclose(df["yes_price"], 0.6)
    np.testing.assert_allclose(np.sign(df["yes_flow"]), [1, -1, -1, 1])
    assert not {"proxyWallet", "name", "pseudonym", "bio"} & set(df.columns)


def test_polymarket_empty_trades_have_schema():
    assert list(polymarket._normalise_trades([]).columns) == polymarket.TRADE_COLUMNS


# ---------------------------------------------------------------- kalshi
def test_kalshi_candles_both_tiers_same_schema():
    historical = {"end_period_ts": 10, "price": {"close": "0.84"}, "yes_bid": {"close": "0.82"},
                  "yes_ask": {"close": "0.84"}, "volume": "5", "open_interest": "7"}
    live = {"end_period_ts": 10, "price": {"close_dollars": "0.84"},
            "yes_bid": {"close_dollars": "0.82"}, "yes_ask": {"close_dollars": "0.84"},
            "volume_fp": "5", "open_interest_fp": "7"}
    pd.testing.assert_frame_equal(kalshi._normalise_candles([historical]),
                                  kalshi._normalise_candles([live]))


def test_kalshi_candle_without_trades_has_nan_price_but_quotes():
    c = {"end_period_ts": 10, "price": {}, "yes_bid": {"close": "0.5"}, "yes_ask": {"close": "0.6"},
         "volume": "0", "open_interest": "1"}
    row = kalshi._normalise_candles([c]).iloc[0]
    assert np.isnan(row["price"]) and row["yes_bid"] == 0.5 and row["volume"] == 0


def test_kalshi_trades_sign_convention():
    rows = [
        {"trade_id": "1", "created_time": "2025-01-01T00:00:00Z", "yes_price_dollars": "0.7",
         "no_price_dollars": "0.3", "count_fp": "10", "taker_side": "yes"},
        {"trade_id": "2", "created_time": "2025-01-01T00:00:01Z", "yes_price_dollars": "0.7",
         "no_price_dollars": "0.3", "count_fp": "10", "taker_side": "no"},
    ]
    df = kalshi._normalise_trades(rows)
    np.testing.assert_allclose(df["yes_flow"], [7.0, -3.0])
    assert df["ts"].iloc[0] == pd.Timestamp("2025-01-01", tz="UTC").timestamp()


def test_kalshi_market_outcome():
    raw = {"ticker": "KXFED-25DEC-H0", "event_ticker": "KXFED-25DEC", "result": "no",
           "close_time": "2025-12-10T18:59:00Z", "open_time": "2025-08-01T00:00:00Z", "volume": "3"}
    m = kalshi.normalise_market(raw)
    assert m["outcome"] == 0.0 and m["series_ticker"] == "KXFED"
    assert np.isnan(kalshi.normalise_market(dict(raw, result=""))["outcome"])


def _http_error(status: int) -> requests.HTTPError:
    resp = requests.Response()
    resp.status_code = status
    return requests.HTTPError(response=resp)


def test_kalshi_tier_fallback_on_404_and_empty(tmp_path):
    k = kalshi.KalshiCollector(cache=ParquetCache(tmp_path))

    def missing() -> Any:
        raise _http_error(404)

    assert k._tiered(False, live=missing, hist=lambda: ["h"]) == ["h"]
    assert k._tiered(False, live=lambda: [], hist=lambda: ["h"]) == ["h"]
    assert k._tiered(True, live=lambda: ["l"], hist=lambda: ["h"]) == ["h"]

    def server_error() -> Any:
        raise _http_error(500)

    with pytest.raises(requests.HTTPError):
        k._tiered(False, live=server_error, hist=lambda: ["h"])


# ---------------------------------------------------------------- live smoke test
@pytest.mark.network
def test_live_dec_2025_fomc_markets(tmp_path):
    """Both platforms agree the Dec-2025 FOMC cut 25 bps (hits the public APIs)."""
    pm = polymarket.PolymarketCollector(cache=ParquetCache(tmp_path))
    ks = kalshi.KalshiCollector(cache=ParquetCache(tmp_path))
    assert pm.get_market("570361")["outcome"] == 1.0
    assert ks.get_market("KXFEDDECISION-25DEC-C25")["outcome"] == 1.0


def test_kalshi_tier_empty_then_404_returns_empty(tmp_path):
    k = kalshi.KalshiCollector(cache=ParquetCache(tmp_path))

    def missing() -> Any:
        raise _http_error(404)

    assert k._tiered(True, live=missing, hist=lambda: []) == []
    with pytest.raises(requests.HTTPError):
        k._tiered(True, live=missing, hist=missing)


def test_polymarket_identical_fills_from_different_wallets_are_kept():
    base = {"transactionHash": "h", "asset": "a", "timestamp": 100, "size": 10.0, "outcomeIndex": 0,
            "side": "BUY", "price": 0.6}
    rows = [dict(base, proxyWallet="w1"), dict(base, proxyWallet="w2"), dict(base, proxyWallet="w2")]
    assert len(polymarket._normalise_trades(rows)) == 2          # exact duplicate removed, distinct wallets kept
