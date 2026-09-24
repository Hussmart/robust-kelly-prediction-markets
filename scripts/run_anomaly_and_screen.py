"""Stages 4 and 6: anomaly scores + consensus flags, and the LazyPredict classifier screen.

Writes ``data/sample/anomalies.parquet`` (all snapshots with IF score and consensus flags),
``results/anomaly_summary.csv`` and ``results/lazypredict_screen.csv``.

Scoring is walk-forward at the meeting level: the Isolation Forest and the consensus
thresholds used to score meeting ``m`` are fitted only on meetings that resolved before
``m`` started its 21-day window, so a flag never relies on the future.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anomaly.graph_consensus import ConsensusRule  # noqa: E402
from src.anomaly.isolation_forest import fit_score  # noqa: E402
from src.baseline.lazypredict_screen import run_screen  # noqa: E402

MIN_TRAIN_MEETINGS = 4


def walk_forward_scores(features: pd.DataFrame, z_min: float = 2.0) -> pd.DataFrame:
    """Score every snapshot with models fitted on strictly earlier meetings."""
    order = features.groupby("meeting").event_time.first().sort_values()
    start = features.groupby("meeting").ts.min()
    parts = []
    for m, t_event in order.items():
        train_meetings = [k for k, t in order.items() if t < pd.Timestamp(start[m], unit="s", tz="UTC")]
        cur = features[features.meeting == m].copy()
        if len(train_meetings) < MIN_TRAIN_MEETINGS:
            cur["if_score"] = np.nan
            cur["price_signal"] = cur["liquidity_signal"] = cur["consensus"] = np.nan
        else:
            train = features[features.meeting.isin(train_meetings)]
            cur["if_score"] = fit_score(train, cur)
            ref = fit_score(train, train)
            cur["if_flag"] = cur["if_score"] >= np.quantile(ref, 0.95)
            flags = ConsensusRule.fit(train, z_min=z_min).flag(cur)
            cur[["price_signal", "liquidity_signal", "consensus"]] = flags
        parts.append(cur)
    out = pd.concat(parts, ignore_index=True)
    out["if_flag"] = out["if_flag"].astype("boolean") if "if_flag" in out else pd.NA
    return out


def main() -> None:
    """Run both stages and print the headline numbers."""
    features = pd.read_parquet(ROOT / "data" / "sample" / "features.parquet")
    scored = walk_forward_scores(features)
    scored.to_parquet(ROOT / "data" / "sample" / "anomalies.parquet", index=False)
    (ROOT / "results").mkdir(exist_ok=True)

    ok = scored.dropna(subset=["if_score"])
    big = ok[ok.abs_div >= 0.02]
    summary = pd.DataFrame({
        "snapshots_scored": [len(ok)],
        "if_flagged": [int(ok.if_flag.sum())],
        "price_signal": [int(ok.price_signal.astype(bool).sum())],
        "liquidity_signal": [int(ok.liquidity_signal.astype(bool).sum())],
        "consensus": [int(ok.consensus.astype(bool).sum())],
        "abs_div>=2c": [len(big)],
    })
    summary.to_csv(ROOT / "results" / "anomaly_summary.csv", index=False)
    print(summary.T.to_string(header=False))

    lab = ok.dropna(subset=["persistent"])
    print("\nPersistence rate of >=2c divergences (label mean) by flag:")
    for name in ("if_flag", "price_signal", "liquidity_signal", "consensus"):
        g = lab.groupby(lab[name].astype(bool)).persistent.agg(["mean", "count"]).round(3)
        print(f"  {name}:", g.to_dict("index"))

    warnings.filterwarnings("ignore")
    screen = run_screen(features)
    screen.to_csv(ROOT / "results" / "lazypredict_screen.csv")
    print("\nLazyPredict screen (chronological test meetings):")
    print(screen.head(12).round(3).to_string())


if __name__ == "__main__":
    main()
