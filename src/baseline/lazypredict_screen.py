"""LazyPredict screen of off-the-shelf classifiers for the real-vs-noise divergence label.

Question: given a snapshot where the two venues disagree by ``|d| >= 2c``, can features
known *at that time* predict whether the divergence is *persistent* (still there, same
sign, at least half the size, 24 h later) rather than transient noise? LazyPredict fits
~30 scikit-learn classifiers with default settings, which gives a fast first look at which
model family is worth tuning (https://github.com/shankarpandala/lazypredict; used here as
a tool, none of its code is copied).

Validation is **grouped by meeting and ordered in time** (train on earlier meetings, test
on later ones). Random splits would leak: snapshots of one meeting are strongly
autocorrelated and the label looks 24 h ahead, so neighbours share information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.anomaly.isolation_forest import derive_if_features

SCREEN_FEATURES: list[str] = [
    "abs_logit_div", "abs_div", "spread_kalshi", "roll_spread_poly", "log_volume_poly",
    "log_volume_kalshi", "ofi_poly", "ofi_kalshi", "ofi_gap", "vol_poly", "vol_kalshi",
    "log_hours_to_close", "log_stale_poly", "log_stale_kalshi",
]


def chronological_split(df: pd.DataFrame, test_frac: float = 0.3, embargo_h: float = 24.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the earliest meetings, test on the latest, with a time embargo between them.

    Meetings are ordered by ``event_time``; the last ``test_frac`` of meetings form the
    test set. Training snapshots whose 24 h look-ahead label window reaches into the test
    period are dropped (``embargo_h``).
    """
    order = df.groupby("meeting").event_time.first().sort_values().index.tolist()
    n_test = max(1, int(round(len(order) * test_frac)))
    test_meetings = set(order[-n_test:])
    test = df[df.meeting.isin(test_meetings)]
    train = df[~df.meeting.isin(test_meetings)]
    cutoff = test.event_time.min() - pd.Timedelta(hours=embargo_h)
    return train[train.event_time <= cutoff], test


def prepare_screen_data(features: pd.DataFrame) -> pd.DataFrame:
    """Keep labelled snapshots (``persistent`` not NaN) and add the model features."""
    labelled = features.dropna(subset=["persistent"])
    return derive_if_features(labelled)


def run_screen(features: pd.DataFrame, verbose: int = 0, random_state: int = 0) -> pd.DataFrame:
    """Fit every LazyPredict classifier on the training meetings and score on the test meetings.

    Returns LazyPredict's leaderboard (sorted by balanced accuracy on the test meetings)
    with an added ``majority_baseline`` row of the trivial always-predict-majority
    classifier, so a reader can see whether any model beats guessing.
    """
    from lazypredict.Supervised import LazyClassifier
    from sklearn.impute import SimpleImputer

    data = prepare_screen_data(features)
    train, test = chronological_split(data)
    imp = SimpleImputer(strategy="median").fit(train[SCREEN_FEATURES])
    x_tr, x_te = imp.transform(train[SCREEN_FEATURES]), imp.transform(test[SCREEN_FEATURES])
    y_tr, y_te = train["persistent"].astype(int).to_numpy(), test["persistent"].astype(int).to_numpy()
    clf = LazyClassifier(verbose=verbose, ignore_warnings=True, custom_metric=None, random_state=random_state)
    models, _ = clf.fit(pd.DataFrame(x_tr, columns=SCREEN_FEATURES), pd.DataFrame(x_te, columns=SCREEN_FEATURES), y_tr, y_te)
    majority = int(np.round(y_tr.mean()))
    base_acc = float(np.mean(y_te == majority))
    models.loc["MajorityBaseline"] = {
        "Accuracy": base_acc, "Balanced Accuracy": 0.5, "ROC AUC": 0.5, "F1 Score": np.nan, "Time Taken": 0.0,
    }
    return models.sort_values("Balanced Accuracy", ascending=False)
