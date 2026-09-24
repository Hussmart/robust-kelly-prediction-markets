"""Build the hand-curated Polymarket <-> Kalshi mapping table for FOMC decision markets.

How the table is made (and why it is *not* fully automatic):

1. The meeting-level pairing below (Polymarket event slug <-> Kalshi event ticker) was
   written by hand after reading both platforms' event titles and dates.
2. Inside a meeting, outcome buckets are matched by explicit rules, and only when the two
   contracts pay out on exactly the same set of Fed decisions:

       Polymarket "No change"            <-> Kalshi H0   (hold)
       Polymarket "decrease 25 bps"      <-> Kalshi C25
       Polymarket "decrease 50+ bps"     <-> Kalshi C26  ("Cut >25bps"; moves are 25bp multiples)
       Polymarket "increase 25 bps"      <-> Kalshi H25  (only where Polymarket splits hikes)
       Polymarket "increase 50+ bps"     <-> Kalshi H26

   Buckets that are unions on one side (Polymarket "increase 25+ bps" = Kalshi H25 + H26;
   Kalshi C26 = Polymarket "50 bps" + "75+ bps" for Nov-2024..Jan-2025) are NOT paired.
3. Every generated row is checked by reading both question texts, and the script refuses
   to write the table if the two sides of any pair resolved differently.

Output: ``data/mappings/fomc_pairs.csv``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.collectors.kalshi import KalshiCollector  # noqa: E402
from src.collectors.polymarket import PolymarketCollector  # noqa: E402

# Meeting-level mapping, curated by hand: (Polymarket event slug, Kalshi event ticker).
MEETINGS: list[tuple[str, str]] = [
    ("fed-interest-rates-may-2024", "FEDDECISION-24MAY"),
    ("fed-interest-rates-june-2024", "FEDDECISION-24JUN"),
    ("fed-interest-rates-july-2024", "FEDDECISION-24JUL"),
    ("fed-interest-rates-september-2024", "FEDDECISION-24SEP"),
    ("fed-interest-rates-november-2024", "FEDDECISION-24NOV"),
    ("fed-interest-rates-december-2024", "KXFEDDECISION-24DEC"),
    ("fed-interest-rates-january-2025", "KXFEDDECISION-25JAN"),
    ("fed-decision-in-march", "KXFEDDECISION-25MAR"),
    ("fed-decision-in-may-2025", "KXFEDDECISION-25MAY"),
    ("fed-decision-in-june", "KXFEDDECISION-25JUN"),
    ("fed-decision-in-july", "KXFEDDECISION-25JUL"),
    ("fed-decision-in-september", "KXFEDDECISION-25SEP"),
    ("fed-decision-in-october", "KXFEDDECISION-25OCT"),
    ("fed-decision-in-december", "KXFEDDECISION-25DEC"),
    ("fed-decision-in-january", "KXFEDDECISION-26JAN"),
    ("fed-decision-in-march-885", "KXFEDDECISION-26MAR"),
    ("fed-decision-in-april", "KXFEDDECISION-26APR"),
    ("fed-decision-in-june-825", "KXFEDDECISION-26JUN"),
    ("fed-decision-in-july-181", "KXFEDDECISION-26JUL"),
    ("fed-decision-in-september-762", "KXFEDDECISION-26SEP"),
]

# Polymarket question pattern -> Kalshi ticker suffix. Order matters: first match wins.
BUCKET_RULES: list[tuple[str, str, str]] = [
    (r"no change", "H0", "hold"),
    (r"decrease.*\b25 bps", "C25", "cut_25"),
    (r"decrease.*\b50\+ bps", "C26", "cut_50plus"),
    (r"increase.*\b25 bps", "H25", "hike_25"),     # plain "25 bps", not "25+ bps"
    (r"increase.*\b50\+ bps", "H26", "hike_50plus"),
]


def classify(question: str) -> tuple[str, str] | None:
    """Return ``(kalshi_suffix, bucket)`` for a Polymarket FOMC question, or None if unpaired."""
    q = question.lower()
    for pattern, suffix, bucket in BUCKET_RULES:
        if re.search(pattern, q) and "25+ bps" not in q:
            return suffix, bucket
    return None


def build() -> pd.DataFrame:
    """Generate the pair table and validate that both sides resolved identically."""
    pm, ks = PolymarketCollector(), KalshiCollector()
    kalshi_all = pd.concat(
        [ks.list_settled_markets("FEDDECISION"), ks.list_settled_markets("KXFEDDECISION")]
    ).set_index("market_id")
    rows = []
    for slug, event in MEETINGS:
        for m in pm.get_event_markets(slug).itertuples():
            match = classify(m.question)
            if match is None:
                continue
            suffix, bucket = match
            ticker = f"{event}-{suffix}"
            if ticker not in kalshi_all.index:
                continue
            k = kalshi_all.loc[ticker]
            rows.append({
                "pair_id": f"{event}-{suffix}",
                "category": "fomc",
                "meeting": event.split("-")[1],
                "bucket": bucket,
                "poly_market_id": m.market_id,
                "poly_event_slug": slug,
                "poly_question": m.question,
                "kalshi_ticker": ticker,
                "kalshi_question": k["question"],
                "kalshi_yes_sub_title": k["yes_sub_title"],
                # Kalshi trading closes minutes before the 2pm-ET announcement; we use it as
                # the common cut-off so that no feature can see the decision itself.
                "event_time": k["close_time"],
                "poly_outcome": m.outcome,
                "kalshi_outcome": k["outcome"],
            })
    df = pd.DataFrame(rows)
    bad = df[df.poly_outcome != df.kalshi_outcome]
    if len(bad):
        raise ValueError(f"resolution mismatch, mapping is wrong:\n{bad}")
    df["outcome"] = df["poly_outcome"].astype(int)
    return df.drop(columns=["poly_outcome", "kalshi_outcome"]).sort_values(["event_time", "bucket"])


if __name__ == "__main__":
    table = build()
    out = ROOT / "data" / "mappings" / "fomc_pairs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"wrote {len(table)} pairs over {table.meeting.nunique()} meetings -> {out}")
    print(table.groupby("bucket").size().to_string())
