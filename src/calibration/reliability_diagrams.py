"""Brier score, expected calibration error, reliability diagrams, bootstrap bands.

Definitions (``q_i`` forecast, ``y_i`` outcome, ``n`` samples):

* Brier score      ``BS = (1/n) sum (q_i - y_i)^2``. It decomposes as
  ``reliability - resolution + uncertainty`` (Murphy, 1973).
* ECE (equal-mass) ``ECE = sum_b (n_b / n) * | mean_b(y) - mean_b(q) |`` over ``B`` bins
  holding equal numbers of samples. Equal-mass bins avoid the near-empty bins that
  equal-width bins produce when most prices sit near 0 or 1.
* Log-loss         ``-(1/n) sum [ y log q + (1-y) log(1-q) ]`` with ``q`` clipped.

Uncertainty of a calibrated probability comes from a **cluster bootstrap**. Snapshots of
the same market are strongly correlated, so resampling snapshots would understate the
variance. We resample whole *markets* (clusters) with replacement, refit the calibrator,
and read the percentile interval of its prediction at the point of interest.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np


class Calibrator(Protocol):
    """Anything with ``fit(p, y)`` returning self and ``predict(p)``."""

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator": ...
    def predict(self, p: np.ndarray) -> np.ndarray: ...


def brier_score(q: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between forecast probabilities and binary outcomes."""
    q, y = np.asarray(q, float), np.asarray(y, float)
    return float(np.mean((q - y) ** 2))


def log_loss(q: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    """Binary cross-entropy with forecasts clipped to ``[eps, 1 - eps]``."""
    q, y = np.clip(np.asarray(q, float), eps, 1 - eps), np.asarray(y, float)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def reliability_table(q: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict[str, np.ndarray]:
    """Equal-mass reliability bins: mean forecast, observed frequency and count per bin."""
    q, y = np.asarray(q, float), np.asarray(y, float)
    order = np.argsort(q, kind="mergesort")
    bins = np.array_split(order, min(n_bins, len(q)))
    return {
        "mean_q": np.array([q[b].mean() for b in bins]),
        "freq": np.array([y[b].mean() for b in bins]),
        "count": np.array([len(b) for b in bins]),
    }


def expected_calibration_error(q: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Equal-mass ECE: count-weighted mean absolute gap between forecast and frequency."""
    t = reliability_table(q, y, n_bins)
    return float(np.sum(t["count"] * np.abs(t["freq"] - t["mean_q"])) / t["count"].sum())


def cluster_bootstrap_predict(
    make_calibrator: Callable[[], Calibrator],
    p: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    p_eval: np.ndarray,
    n_boot: int = 500,
    alpha: float = 0.10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap ``(median, lower, upper)`` of the calibrated probability at ``p_eval``.

    Whole ``groups`` (markets) are resampled with replacement. Resamples that contain
    a single outcome class are skipped. The returned interval is the
    ``[alpha/2, 1 - alpha/2]`` percentile interval (default 90%).
    """
    p, y, groups, p_eval = map(np.asarray, (p, y, groups, p_eval))
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in ids}
    draws: list[np.ndarray] = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        draws.append(make_calibrator().fit(p[idx], y[idx]).predict(p_eval))
    if not draws:
        raise ValueError("no valid bootstrap resample (need both outcome classes)")
    arr = np.vstack(draws)
    return (np.median(arr, axis=0), np.quantile(arr, alpha / 2, axis=0), np.quantile(arr, 1 - alpha / 2, axis=0))


def plot_reliability(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], path: str, title: str = "Reliability diagram",
    counts: np.ndarray | None = None,
) -> None:
    """Save a reliability diagram. ``curves`` maps a label to ``(mean_q, freq)`` arrays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for label, (mq, fr) in curves.items():
        ax.plot(mq, fr, "o-", ms=4, label=label)
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
