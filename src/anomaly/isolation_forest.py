"""Isolation Forest anomaly scoring on engineered cross-platform features.

The idea follows ``borisbanushev/anomaliesinoptions`` (unsupervised anomaly scores as a
first-pass mispricing screen). The implementation is our own, and so is the choice of
features: cross-*venue* divergence of the same event instead of option-pricing
features.

Isolation Forest (Liu, Ting & Zhou, 2008) isolates a point by random axis-aligned splits.
Anomalies are isolated in fewer splits. With ``h(x)`` the path length of ``x`` in one
tree and ``c(n) = 2 H(n-1) - 2(n-1)/n`` the average path length of an unsuccessful
BST search over ``n`` points (``H`` the harmonic number), the score is

    s(x, n) = 2 ** ( -E[h(x)] / c(n) )  in (0, 1],

close to 1 for anomalies and well below 0.5 for normal points. We report
``anomaly_score = -score_samples(x)``, which is sklearn's version of ``s`` (higher =
more anomalous).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

# Features describe *how* the two venues disagree and *how liquid* they are, never the
# level of the price itself (which carries no information about mispricing).
IF_FEATURES: list[str] = [
    "abs_logit_div",
    "abs_div",
    "spread_kalshi",
    "roll_spread_poly",
    "log_volume_poly",
    "log_volume_kalshi",
    "ofi_poly",
    "ofi_kalshi",
    "ofi_gap",
    "vol_poly",
    "vol_kalshi",
    "log_hours_to_close",
    "log_stale_poly",
    "log_stale_kalshi",
]


def derive_if_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add the transformed columns used by the Isolation Forest to a copy of ``features``."""
    out = features.copy()
    out["abs_logit_div"] = out["logit_div"].abs()
    out["log_volume_poly"] = np.log1p(out["volume_poly"])
    out["log_volume_kalshi"] = np.log1p(out["volume_kalshi"])
    out["ofi_gap"] = (out["ofi_poly"] - out["ofi_kalshi"]).abs()
    out["log_hours_to_close"] = np.log1p(out["hours_to_close"])
    # Never-traded venues get the maximal staleness of the window (21 days).
    out["log_stale_poly"] = np.log1p(out["stale_poly_h"].fillna(21 * 24))
    out["log_stale_kalshi"] = np.log1p(out["stale_kalshi_h"].fillna(21 * 24))
    return out


def make_isolation_forest(n_estimators: int = 300, random_state: int = 0) -> Pipeline:
    """Median imputation -> robust scaling -> Isolation Forest.

    Robust scaling (median / IQR) keeps the heavy-tailed volume features from dominating
    split selection. Isolation Forest is scale-invariant per split, but imputation and
    scaling make the pipeline reusable and reproducible.
    """
    return make_pipeline(
        FunctionTransformer(lambda x: np.asarray(x, dtype=float)),
        SimpleImputer(strategy="median"),
        RobustScaler(),
        IsolationForest(n_estimators=n_estimators, random_state=random_state),
    )


def fit_score(train: pd.DataFrame, score: pd.DataFrame | None = None, **kwargs: int) -> np.ndarray:
    """Fit on ``train`` and return anomaly scores (higher = more anomalous) for ``score``.

    In the backtest ``train`` holds only snapshots from meetings resolved before the
    decision time, so the notion of "normal" never uses the future.
    """
    model = make_isolation_forest(**kwargs)
    model.fit(derive_if_features(train)[IF_FEATURES])
    target = train if score is None else score
    return -model.score_samples(derive_if_features(target)[IF_FEATURES])


def flag_top_quantile(scores: np.ndarray, reference: np.ndarray, q: float = 0.95) -> np.ndarray:
    """Flag scores above the ``q`` quantile of ``reference`` (the training-set scores)."""
    return scores >= np.quantile(reference, q)
