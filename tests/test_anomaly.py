"""Tests for the anomaly modules: equivalence graph, robust z, consensus rule, Isolation Forest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.anomaly.graph_consensus import ConsensusRule, build_equivalence_graph, robust_z, validate_graph
from src.anomaly.isolation_forest import IF_FEATURES, derive_if_features, fit_score
from tests.test_backtest import _synthetic_features


def test_equivalence_graph_is_a_bipartite_matching():
    mapping = pd.DataFrame({"poly_market_id": ["1", "2"], "kalshi_ticker": ["K1", "K2"],
                            "pair_id": ["p1", "p2"], "meeting": ["a", "a"], "bucket": ["hold", "cut_25"]})
    g = build_equivalence_graph(mapping)
    validate_graph(g)
    assert g.number_of_nodes() == 4 and g.number_of_edges() == 2
    dup = pd.concat([mapping, mapping.iloc[[0]].assign(kalshi_ticker="K3", pair_id="p3")])
    with pytest.raises(ValueError):
        validate_graph(build_equivalence_graph(dup))


def test_robust_z_is_robust_to_outliers():
    ref = np.concatenate([np.random.default_rng(0).normal(0, 1, 1000), [1e6]])
    z = robust_z(np.array([0.0, 3.0]), ref)
    assert z[0] == pytest.approx(0.0, abs=0.1) and z[1] == pytest.approx(3.0, abs=0.3)
    assert np.isfinite(robust_z(np.array([1.0]), np.zeros(10))).all()


def test_consensus_requires_both_signals():
    train = _synthetic_features()
    rule = ConsensusRule.fit(train, z_min=2.0)
    row = train.iloc[[0]].copy()
    big = row.assign(logit_div=5.0)
    small = row.assign(logit_div=0.0)
    thin = big.assign(volume_poly=0.0)
    assert rule.flag(big)["consensus"].iloc[0]
    assert not rule.flag(small)["consensus"].iloc[0]          # no price signal
    assert not rule.flag(thin)["consensus"].iloc[0]           # price signal but illiquid
    assert rule.flag(thin)["price_signal"].iloc[0] and not rule.flag(thin)["liquidity_signal"].iloc[0]


def test_isolation_forest_scores_planted_outlier_highest():
    df = _synthetic_features()
    df["ofi_gap"] = 0.0
    out = df.iloc[[0]].copy()
    out["logit_div"], out["abs_div"], out["spread_kalshi"] = 4.0, 0.6, 0.4
    both = pd.concat([df, out], ignore_index=True)
    scores = fit_score(df, both)
    assert scores.argmax() == len(both) - 1
    assert set(IF_FEATURES) <= set(derive_if_features(both).columns)
