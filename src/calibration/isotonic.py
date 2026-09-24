"""Isotonic regression calibration via the pool-adjacent-violators algorithm (PAVA).

Model: ``P(Y=1 | p) = f(p)`` with ``f`` non-decreasing and otherwise free. The
least-squares monotone fit

    min_f  sum_i w_i (y_i - f(p_i))^2   s.t.  f(p_i) <= f(p_j) whenever p_i <= p_j

is solved exactly by PAVA (Barlow et al., 1972): scan left to right keeping blocks of
constant fitted value, and whenever a new block's mean is below the previous block's
mean, merge them into their weighted mean. The result is a step function; prediction
interpolates linearly between block centres and clips outside the training range.

Isotonic regression is more flexible than Platt scaling (no parametric form) but it needs
more data and its step function is noisy in sparse regions, which the bootstrap bands
in ``reliability_diagrams.py`` show.
"""

from __future__ import annotations

import numpy as np


def pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators: non-decreasing least-squares fit of ``y``.

    ``y`` must already be ordered by the explanatory variable. Returns the fitted value
    for every element.
    """
    n = len(y)
    value = np.empty(n)
    weight = np.empty(n)
    size = np.empty(n, dtype=int)
    top = -1
    for i in range(n):
        top += 1
        value[top], weight[top], size[top] = y[i], w[i], 1
        while top > 0 and value[top - 1] > value[top]:
            tot = weight[top - 1] + weight[top]
            value[top - 1] = (weight[top - 1] * value[top - 1] + weight[top] * value[top]) / tot
            weight[top - 1] = tot
            size[top - 1] += size[top]
            top -= 1
    return np.repeat(value[: top + 1], size[: top + 1])


class IsotonicCalibrator:
    """Monotone non-parametric calibrator ``q = f(p)`` fitted by PAVA."""

    def __init__(self, floor: float = 0.0, ceil: float = 1.0) -> None:
        """``floor`` / ``ceil`` clip the output (e.g. 0.005 / 0.995 to avoid exact 0 or 1)."""
        self.floor, self.ceil = floor, ceil
        self.x_: np.ndarray = np.array([0.0, 1.0])
        self.f_: np.ndarray = np.array([0.0, 1.0])

    def fit(self, p: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "IsotonicCalibrator":
        """Fit the monotone map. Tied prices are averaged before PAVA."""
        p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
        w = np.ones_like(p) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        order = np.argsort(p, kind="mergesort")
        p, y, w = p[order], y[order], w[order]
        uniq, start = np.unique(p, return_index=True)
        wsum = np.add.reduceat(w, start)
        ymean = np.add.reduceat(w * y, start) / wsum
        self.x_, self.f_ = uniq, pava(ymean, wsum)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        """Piecewise-linear interpolation of the fitted steps, clipped to the training range."""
        return np.clip(np.interp(np.asarray(p, dtype=float), self.x_, self.f_), self.floor, self.ceil)
