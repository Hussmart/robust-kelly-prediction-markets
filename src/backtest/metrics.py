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
    """All headline metrics for one strategy, plus hit rate and worst round."""
    r = np.asarray(log_returns, dtype=float)
    return {
        "rounds": float(len(r)),
        "cum_log_growth": cumulative_log_growth(r),
        "final_wealth": float(np.exp(np.sum(r))),
        "mean_log_return": float(np.mean(r)) if len(r) else float("nan"),
        "max_drawdown": max_drawdown(r) if len(r) else float("nan"),
        "sharpe_like": sharpe_like(r, periods_per_year),
        "win_rate": float(np.mean(r > 0)) if len(r) else float("nan"),
        "worst_round": float(np.min(r)) if len(r) else float("nan"),
    }
