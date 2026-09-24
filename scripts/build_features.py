"""Build the snapshot feature panel for all pairs -> data/processed/features.parquet.

A copy is written to data/sample/features.parquet (a few hundred KB) so that every
downstream stage can be reproduced without touching the APIs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.collectors.kalshi import KalshiCollector  # noqa: E402
from src.collectors.pairs import collect_pair, load_mapping  # noqa: E402
from src.collectors.polymarket import PolymarketCollector  # noqa: E402
from src.features.feature_engineering import add_persistence_label, build_pair_panel  # noqa: E402

if __name__ == "__main__":
    pm, ks = PolymarketCollector(), KalshiCollector()
    mapping = load_mapping()
    panels = []
    for pair in mapping.itertuples(index=False):
        d = collect_pair(pair, pm, ks)
        panel = build_pair_panel(
            pd.Series(pair._asdict()), d.start_ts, d.end_ts,
            d.poly_prices, d.poly_trades, d.kalshi_candles, d.kalshi_trades,
        )
        panels.append(panel)
    features = add_persistence_label(pd.concat(panels, ignore_index=True))
    features = features.merge(mapping[["pair_id", "event_time"]], on="pair_id")
    for out in (ROOT / "data" / "processed" / "features.parquet", ROOT / "data" / "sample" / "features.parquet"):
        out.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(out, index=False)
    kept = features.pair_id.nunique()
    print(f"{len(features)} snapshots, {kept}/{len(mapping)} pairs with overlapping prices")
    print(features.describe().T[["count", "mean", "50%", "max"]].round(4).to_string())
