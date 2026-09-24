"""Download (and cache) raw series for every pair in data/mappings/fomc_pairs.csv."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collectors.kalshi import KalshiCollector  # noqa: E402
from src.collectors.pairs import collect_pair, load_mapping  # noqa: E402
from src.collectors.polymarket import PolymarketCollector  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    pm, ks = PolymarketCollector(), KalshiCollector()
    mapping = load_mapping()
    for i, pair in enumerate(mapping.itertuples(index=False), 1):
        t0 = time.time()
        d = collect_pair(pair, pm, ks)
        print(f"[{i:>2}/{len(mapping)}] {pair.pair_id:<26} poly px={len(d.poly_prices):>4} "
              f"tr={len(d.poly_trades):>6} | kalshi candles={len(d.kalshi_candles):>4} "
              f"tr={len(d.kalshi_trades):>6} ({time.time() - t0:.0f}s)", flush=True)
