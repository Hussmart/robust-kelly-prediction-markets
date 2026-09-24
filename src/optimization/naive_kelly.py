"""Closed-form-style (non-robust) Kelly allocation for independent binary contracts.

Model (see ``docs/methodology.md`` §5). Bet ``i`` costs ``c_i`` per contract and pays 1 if
it wins, so a stake of ``f_i`` (fraction of wealth) returns ``1 + f_i b_i`` on a win, with
net odds ``b_i = (1 - c_i) / c_i``, and ``1 - f_i`` on a loss. With independent bets
settled sequentially with reinvestment, the expected log-growth is *separable*:

    g(f; p) = sum_i [ p_i log(1 + f_i b_i) + (1 - p_i) log(1 - f_i) ].

Each term is concave, so ``max g`` subject to ``sum_i f_i <= F`` and ``0 <= f_i <= f_max``
is a convex problem. The KKT conditions give a one-dimensional problem in the budget
multiplier ``mu >= 0``: each ``f_i(mu)`` solves

    p_i b_i / (1 + f_i b_i) - (1 - p_i) / (1 - f_i) = mu,

whose left side is strictly decreasing in ``f_i``. So ``f_i(mu)`` is found by a bracketed
root search, and ``mu`` by bisection on ``sum_i f_i(mu) = F``. With ``mu = 0`` and a slack
budget, ``f_i = (p_i b_i - (1 - p_i)) / b_i``, the classical Kelly fraction.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def net_odds(c: np.ndarray) -> np.ndarray:
    """Net odds ``b = (1 - c) / c`` of a contract bought at price ``c``."""
    c = np.asarray(c, dtype=float)
    return (1.0 - c) / c


def growth(f: np.ndarray, p: np.ndarray, c: np.ndarray) -> float:
    """Expected log-growth ``g(f; p)`` of allocation ``f`` when win probabilities are ``p``."""
    f, p, b = (np.asarray(x, dtype=float) for x in (f, p, net_odds(c)))
    return float(np.sum(p * np.log1p(f * b) + (1.0 - p) * np.log1p(-f)))


def _marginal(f: float, p: float, b: float) -> float:
    """Derivative of one bet's expected log-growth with respect to its stake."""
    return p * b / (1.0 + f * b) - (1.0 - p) / (1.0 - f)


def _stake_at_multiplier(p: float, b: float, mu: float, f_max: float) -> float:
    """Stake solving ``marginal(f) = mu`` on ``[0, f_max]`` (0 if even ``f = 0`` is below ``mu``)."""
    if _marginal(0.0, p, b) <= mu:
        return 0.0
    if _marginal(f_max, p, b) >= mu:
        return f_max
    return brentq(lambda f: _marginal(f, p, b) - mu, 0.0, f_max, xtol=1e-14)


def kelly_fractions(p: np.ndarray, c: np.ndarray, budget: float = 1.0, f_max: float = 0.5) -> np.ndarray:
    """Optimal non-robust stakes for win probabilities ``p`` and prices ``c``.

    Args:
        p: (calibrated) win probability of each bet.
        c: purchase price of each contract (so ``p > c`` is a positive-edge bet).
        budget: maximum total fraction of wealth staked, ``F``.
        f_max: per-bet cap (must be < 1).

    Returns:
        Array of stakes ``f_i in [0, f_max]`` with ``sum f_i <= budget``.
    """
    p, c = np.asarray(p, dtype=float), np.asarray(c, dtype=float)
    if not 0 < f_max < 1:
        raise ValueError("f_max must lie in (0, 1)")
    b = net_odds(c)

    def stakes(mu: float) -> np.ndarray:
        return np.array([_stake_at_multiplier(pi, bi, mu, f_max) for pi, bi in zip(p, b)])

    f = stakes(0.0)
    if f.sum() <= budget:
        return f
    lo, hi = 0.0, float(max(_marginal(0.0, pi, bi) for pi, bi in zip(p, b)))
    for _ in range(200):  # bisection on the budget multiplier
        mid = 0.5 * (lo + hi)
        if stakes(mid).sum() > budget:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    return stakes(hi)
