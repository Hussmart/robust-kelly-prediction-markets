"""Performance metrics: cumulative log-growth, max drawdown, Sharpe-like ratio.

All metrics are computed from a vector of per-round *log-returns* ``r_t = log(W_t / W_{t-1})``
where ``W_t`` is wealth after round ``t``. Log-returns add up, so cumulative log-growth is
``sum r_t``, and wealth is ``exp(cumsum(r))``.
"""

from __future__ import annotations

import numpy as np


def wealth_curve(log_returns: np.ndarray, w0: float = 1.0) -> np.ndarray:
    """Wealth path ``W_0, W_1, ..., W_T`` (length ``T + 1``) from per-round log-returns."""
    return w0 * np.exp(np.concatenate([[0.0], np.cumsum(np.asarray(log_returns, dtype=float))]))


def cumulative_log_growth(log_returns: np.ndarray) -> float:
    """Total log-growth ``sum_t r_t`` = ``log(W_T / W_0)``."""
    return float(np.sum(log_returns))


def max_drawdown(log_returns: np.ndarray) -> float:
    """Largest peak-to-trough fall of the wealth curve, as a fraction in ``[0, 1]``.

    ``MDD = max_t (1 - W_t / max_{s <= t} W_s)``; 0 for a curve that never falls.
    """
    w = wealth_curve(log_returns)
    peak = np.maximum.accumulate(w)
    return float(np.max(1.0 - w / peak))


def sharpe_like(log_returns: np.ndarray, periods_per_year: float = 8.0) -> float:
    """Annualised mean/std of per-round log-returns (no risk-free rate).

    ``periods_per_year`` is the number of rounds per year (the FOMC meets 8 times a year).
    Returns NaN with fewer than two rounds or zero variance. With ~15 rounds this is a
    very noisy statistic and it is reported as such.
    """
    r = np.asarray(log_returns, dtype=float)
    if len(r) < 2:
        return float("nan")
    sd = np.std(r, ddof=1)
    if sd <= 1e-12 * max(1.0, abs(np.mean(r))):
        return float("nan")
    return float(np.mean(r) / sd * np.sqrt(periods_per_year))


def summarize(log_returns: np.ndarray, periods_per_year: float = 8.0) -> dict[str, float]:
    """All headline metrics for one strategy, plus the fraction of rounds with positive log-return (rounds with no bet count as not positive) and worst round."""
    r = np.asarray(log_returns, dtype=float)
    return {
        "rounds": float(len(r)),
        "cum_log_growth": cumulative_log_growth(r),
        "final_wealth": float(np.exp(np.sum(r))),
        "mean_log_return": float(np.mean(r)) if len(r) else float("nan"),
        "max_drawdown": max_drawdown(r) if len(r) else float("nan"),
        "sharpe_like": sharpe_like(r, periods_per_year),
        "frac_rounds_positive": float(np.mean(r > 0)) if len(r) else float("nan"),
        "worst_round": float(np.min(r)) if len(r) else float("nan"),
    }


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 5000, alpha: float = 0.10, seed: int = 0
) -> dict[str, float]:
    """Bootstrap the difference in cumulative log-growth ``sum(a) - sum(b)`` over rounds.

    Rounds are resampled *jointly* (the same rounds for both strategies), which respects the
    pairing: both strategies face the same markets in every round. Returns the observed
    difference, a ``1 - alpha`` percentile interval and the bootstrap probability that the
    difference is positive. With few rounds the interval is wide, which is the point.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("a and b must be equal-length non-empty 1-D arrays")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = (a[idx] - b[idx]).sum(axis=1)
    return {
        "observed": float(np.sum(a - b)),
        "lo": float(np.quantile(diffs, alpha / 2)),
        "hi": float(np.quantile(diffs, 1 - alpha / 2)),
        "prob_positive": float(np.mean(diffs > 0)),
    }


def paired_bootstrap_stat(
    a: np.ndarray, b: np.ndarray, stat, n_boot: int = 3000, alpha: float = 0.10, seed: int = 0
) -> dict[str, float]:
    """Bootstrap ``stat(a) - stat(b)`` for any statistic of a return series.

    Rounds are resampled jointly (same indices for both strategies) and the statistic is
    recomputed on each resample, e.g. ``max_drawdown`` or ``sharpe_like``. Resampling rounds i.i.d.
    ignores serial dependence, which is acceptable for cohorts that do not overlap in time but
    is an approximation for drawdowns.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("a and b must be equal-length non-empty 1-D arrays")
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), size=len(a))
        va, vb = stat(a[idx]), stat(b[idx])
        if np.isfinite(va) and np.isfinite(vb):
            diffs.append(va - vb)
    d = np.asarray(diffs)
    return {
        "observed": float(stat(a) - stat(b)),
        "lo": float(np.quantile(d, alpha / 2)),
        "hi": float(np.quantile(d, 1 - alpha / 2)),
        "prob_positive": float(np.mean(d > 0)),
    }
