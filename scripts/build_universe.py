"""Build the broad Polymarket universe -> data/processed/universe.parquet (+ data/sample copy)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.collectors.polymarket import PolymarketCollector  # noqa: E402
from src.collectors.universe import build_universe  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    uni = build_universe(PolymarketCollector())
    for out in (ROOT / "data" / "processed" / "universe.parquet", ROOT / "data" / "sample" / "universe.parquet"):
        out.parent.mkdir(parents=True, exist_ok=True)
        uni.to_parquet(out, index=False)
    print(f"{len(uni)} markets, {uni.cohort.nunique()} cohorts, {uni.event_key.nunique()} events, "
          f"yes-rate {uni.outcome.mean():.3f}, horizon days {uni.horizon_days.min():.1f}-{uni.horizon_days.max():.1f}")
