"""Bipartite equivalence graph with a joint price AND liquidity anomaly consensus rule.

Graph ``G = (P ∪ K, E)``: ``P`` are Polymarket contracts, ``K`` Kalshi contracts and an
edge ``(i, j)`` means ``i ~ j`` (equivalent payoff, see ``docs/methodology.md`` §1). Each
edge carries a time series of divergences. A snapshot on an edge is *flagged* only if two
signals fire at once:

* **price signal**   ``S_price = 1{ z(|logit divergence|) >= z_min }`` where ``z`` is a
  robust z-score (median / 1.4826 MAD) against the training distribution of divergences;
* **liquidity signal** ``S_liq = 1{ both venues are liquid, fresh and tightly quoted }``.

The second signal is a *validity* condition. A big gap between two thin or stale books is
what a false positive looks like, because nobody could trade it, while the same gap between two
active books is a candidate real mispricing. Requiring both signals is the "joint
anomaly, not a single signal" rule from my earlier TrueWind project. It is a consensus
of two independent detectors: the price detector says the venues disagree, the
liquidity detector says the disagreement is tradable.

Liquidity thresholds are quantiles of the training data (never hand-picked absolute
levels), so the rule adapts to how thin these markets are overall.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826


def build_equivalence_graph(mapping: pd.DataFrame) -> nx.Graph:
    """Bipartite graph with one edge per equivalent pair.

    Nodes are ``("poly", market_id)`` / ``("kalshi", ticker)`` with attribute ``bipartite``
    0 / 1. Edges carry ``pair_id`` and ``meeting``. Contracts of the same meeting are
    mutually exclusive, so each meeting forms one connected group of edges only through
    shared *event* membership, not through shared nodes. The graph therefore is a perfect
    matching (max degree 1), which ``validate_graph`` asserts.
    """
    g = nx.Graph()
    for r in mapping.itertuples(index=False):
        u, v = ("poly", str(r.poly_market_id)), ("kalshi", r.kalshi_ticker)
        g.add_node(u, bipartite=0)
        g.add_node(v, bipartite=1)
        g.add_edge(u, v, pair_id=r.pair_id, meeting=r.meeting, bucket=r.bucket)
    return g


def validate_graph(g: nx.Graph) -> None:
    """Assert the graph is bipartite and a matching (each contract has exactly one twin)."""
    if not nx.is_bipartite(g):
        raise ValueError("equivalence graph must be bipartite")
    too_many = [n for n, d in g.degree() if d != 1]
    if too_many:
        raise ValueError(f"contracts with != 1 equivalent counterpart: {too_many[:5]}")


def robust_z(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Robust z-score of ``x`` against ``reference``: ``(x - median) / (1.4826 * MAD)``.

    Falls back to the reference standard deviation when the MAD is zero (many exact
    zeros), and to 1 if that is zero too, so the result is always finite.
    """
    ref = np.asarray(reference, dtype=float)
    ref = ref[~np.isnan(ref)]
    med = np.median(ref)
    scale = MAD_TO_SIGMA * np.median(np.abs(ref - med))
    if scale == 0:
        scale = ref.std() or 1.0
    return (np.asarray(x, dtype=float) - med) / scale


@dataclass(frozen=True)
class ConsensusRule:
    """Thresholds of the joint rule, calibrated on training snapshots only.

    Attributes:
        z_min: minimum robust z of ``|logit divergence|`` for the price signal.
        min_volume: per-venue trailing-24h notional required for the liquidity signal
            (a quantile of the training volumes).
        max_stale_h: maximum hours since the last trade on each venue.
        max_spread: maximum Kalshi quoted spread.
        ref_center / ref_scale: median and scaled MAD of ``|logit div|`` in training.
    """

    z_min: float
    min_volume_poly: float
    min_volume_kalshi: float
    max_stale_h: float
    max_spread: float
    ref_center: float
    ref_scale: float

    @classmethod
    def fit(
        cls, train: pd.DataFrame, z_min: float = 3.0, volume_q: float = 0.25,
        stale_h: float = 12.0, spread_q: float = 0.75,
    ) -> "ConsensusRule":
        """Estimate every threshold from the training snapshots."""
        ref = train["logit_div"].abs().to_numpy()
        center = float(np.median(ref))
        scale = float(MAD_TO_SIGMA * np.median(np.abs(ref - center))) or float(ref.std() or 1.0)
        return cls(
            z_min=z_min,
            min_volume_poly=float(train.loc[train.volume_poly > 0, "volume_poly"].quantile(volume_q)),
            min_volume_kalshi=float(train.loc[train.volume_kalshi > 0, "volume_kalshi"].quantile(volume_q)),
            max_stale_h=stale_h,
            max_spread=float(train["spread_kalshi"].quantile(spread_q)),
            ref_center=center,
            ref_scale=scale,
        )

    def price_signal(self, df: pd.DataFrame) -> pd.Series:
        """Boolean series: robust z of ``|logit div|`` reaches ``z_min``."""
        z = (df["logit_div"].abs() - self.ref_center) / self.ref_scale
        return z >= self.z_min

    def liquidity_signal(self, df: pd.DataFrame) -> pd.Series:
        """Boolean series: both venues liquid, fresh, and (if quoted) tight."""
        spread_ok = df["spread_kalshi"].isna() | (df["spread_kalshi"] <= self.max_spread)
        return (
            (df["volume_poly"] >= self.min_volume_poly)
            & (df["volume_kalshi"] >= self.min_volume_kalshi)
            & (df["stale_poly_h"] <= self.max_stale_h)
            & (df["stale_kalshi_h"] <= self.max_stale_h)
            & spread_ok
        )

    def flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return ``price_signal``, ``liquidity_signal`` and their conjunction ``consensus``."""
        price, liq = self.price_signal(df), self.liquidity_signal(df)
        return pd.DataFrame({"price_signal": price, "liquidity_signal": liq,
                             "consensus": price & liq}, index=df.index)
