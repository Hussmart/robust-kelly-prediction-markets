"""Warm the price cache for the universe in a different order (parallel to build_universe).

Usage: ``python scripts/warm_universe_cache.py reverse|middle``. Each worker pulls the entry
features of the same sampled markets that ``build_universe`` needs, so the main build finds them
cached and finishes sooner. Purely a speed-up; it changes no result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.collectors.polymarket import PolymarketCollector  # noqa: E402
from src.collectors.universe import assign_cohorts, entry_features, sample_markets  # noqa: E402

if __name__ == "__main__":
    pm = PolymarketCollector()
    listed = pm.list_resolved_markets("2024-07-01", "2026-08-31", volume_min=None, max_markets=10**7, window_days=7)
    m = sample_markets(assign_cohorts(listed), 12_000, 0)
    order = sys.argv[1] if len(sys.argv) > 1 else "reverse"
    rows = m.iloc[::-1] if order == "reverse" else pd.concat([m.iloc[len(m) // 2:], m.iloc[: len(m) // 2]])
    for i, row in enumerate(rows.itertuples(index=False), 1):
        try:
            entry_features(pm, row.yes_token_id, int(row.decision_ts))
        except Exception as exc:  # noqa: BLE001
            print("skip", row.market_id, exc, flush=True)
        if i % 500 == 0:
            print(i, flush=True)
