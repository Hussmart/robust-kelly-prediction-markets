"""Pyomo MILP for budgeted-uncertainty (Bertsimas-Sim) robust Kelly, solved with HiGHS.

Derivation (full version in ``docs/methodology.md`` §5). Start from the separable
log-growth of ``naive_kelly``. Fix ``f``. It is linear in the win probabilities:

    g(f; p) = sum_i B_i(f_i) + sum_i p_i D_i(f_i),
    B_i(f) = log(1 - f),   D_i(f) = log(1 + f b_i) - log(1 - f) >= 0.

Uncertainty set (Bertsimas & Sim, 2004). The estimate ``p^_i`` may be wrong by up to
``d_i`` (typically ``p^_i`` minus the lower bootstrap bound). Only *downward* deviations
hurt (``D_i >= 0``), and at most ``Gamma`` of the ``N`` estimates deviate at once:

    U(Gamma) = { p : p_i = p^_i - d_i z_i,  0 <= z_i <= 1,  sum_i z_i <= Gamma }.

Worst case for fixed ``f``:

    min_{p in U} g = sum_i [B_i + p^_i D_i] - max_z { sum_i d_i D_i z_i : sum z <= Gamma, 0 <= z <= 1 }.

The inner maximisation is an LP whose dual is (``lambda`` for the budget row, ``nu_i`` for
the ``z_i <= 1`` rows):

    min  Gamma * lambda + sum_i nu_i   s.t.  lambda + nu_i >= d_i D_i(f_i),  lambda, nu >= 0,

so, by strong duality, the robust problem is the single maximisation

    max_{f, lambda, nu}  sum_i [B_i(f_i) + p^_i D_i(f_i)] - Gamma * lambda - sum_i nu_i
    s.t. lambda + nu_i >= d_i D_i(f_i),   sum_i f_i <= F,   0 <= f_i <= f_max.

*Why a MILP.* ``D_i`` is neither convex nor concave (``log(1 + f b)`` is concave while
``-log(1 - f)`` is convex), so the constraint ``lambda + nu_i >= d_i D_i(f_i)`` is not a
convex constraint. We replace ``B_i``, ``D_i`` by piecewise-linear interpolants on the
grid ``0 = f_0 < ... < f_K = f_max`` using the incremental (delta) formulation,

    f_i = sum_k delta_ik (f_k - f_{k-1}),   delta_{i,k+1} <= y_ik <= delta_ik,  y_ik in {0,1}, delta in [0,1],

whose binaries force segments to fill in order, which the non-convex constraint needs
(without them the relaxation would fill the segments in whatever order minimises
``D_i``). The result is a MILP. Special cases checked in ``tests/test_robust_kelly.py``:

* ``Gamma = 0``: ``lambda`` and ``nu`` vanish, the MILP is the (piecewise-linear) naive Kelly.
* ``Gamma >= N``: every estimate sits at its worst case ``p^_i - d_i``, i.e. Kelly at the
  lower bounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyomo.environ as pyo

from src.optimization.naive_kelly import net_odds

P_FLOOR = 1e-3


@dataclass
class RobustKellyResult:
    """Solution of the robust Kelly MILP.

    Attributes:
        f: optimal stakes.
        objective: optimal worst-case growth of the piecewise-linear model.
        worst_case_growth: exact (not piecewise-linear) worst-case growth of ``f``.
        gamma: uncertainty budget used.
        status: solver termination condition.
        optimal: True if HiGHS proved optimality (within ``mip_gap``); False if it stopped at
            the time limit with a feasible incumbent.
    """

    f: np.ndarray
    objective: float
    worst_case_growth: float
    gamma: float
    status: str
    optimal: bool = True


def _grid(f_max: float, n_segments: int) -> np.ndarray:
    """Quadratically spaced breakpoints (dense near 0, where Kelly stakes live)."""
    return f_max * (np.arange(n_segments + 1) / n_segments) ** 2


def clip_deviation(p_hat: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Limit each deviation so that the worst-case probability stays above ``P_FLOOR``."""
    p_hat, d = np.asarray(p_hat, dtype=float), np.asarray(d, dtype=float)
    return np.minimum(np.maximum(d, 0.0), np.maximum(p_hat - P_FLOOR, 0.0))


def worst_case_growth(f: np.ndarray, p_hat: np.ndarray, c: np.ndarray, d: np.ndarray, gamma: float) -> float:
    """Exact worst-case expected log-growth of ``f`` over ``U(gamma)``.

    The adversary's LP has a greedy solution: spend the budget on the estimates with the
    largest ``d_i D_i(f_i)``, whole units first and the fractional remainder last.
    """
    f, p_hat = np.asarray(f, dtype=float), np.asarray(p_hat, dtype=float)
    d = clip_deviation(p_hat, d)
    b = net_odds(c)
    B = np.log1p(-f)
    D = np.log1p(f * b) - np.log1p(-f)
    nominal = float(np.sum(B + p_hat * D))
    loss = np.sort(d * D)[::-1]
    g = min(float(gamma), float(len(loss)))
    whole = int(np.floor(g + 1e-12))
    penalty = float(loss[:whole].sum())
    if whole < len(loss) and g > whole:
        penalty += (g - whole) * float(loss[whole])
    return nominal - penalty


def solve_robust_kelly(
    p_hat: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    gamma: float,
    budget: float = 1.0,
    f_max: float = 0.5,
    n_segments: int = 40,
    time_limit: float = 60.0,
    mip_gap: float = 1e-6,
) -> RobustKellyResult:
    """Solve the Bertsimas-Sim robust Kelly MILP with Pyomo + HiGHS.

    Args:
        p_hat: point estimates of the win probability of each bet.
        c: purchase prices (win pays 1).
        d: maximum downward deviation of each estimate (e.g. ``p_hat - lower_bound``).
        gamma: uncertainty budget in ``[0, N]`` (max number of estimates at worst case).
        budget: maximum total fraction of wealth staked.
        f_max: per-bet cap, ``< 1``.
        n_segments: number of piecewise-linear segments per bet.
        time_limit: solver time limit in seconds.
        mip_gap: relative MIP gap.
    """
    p_hat, c = np.asarray(p_hat, dtype=float), np.asarray(c, dtype=float)
    n = len(p_hat)
    if not (len(c) == len(d) == n):
        raise ValueError("p_hat, c and d must have the same length")
    if not 0 < f_max < 1:
        raise ValueError("f_max must lie in (0, 1)")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    d = clip_deviation(p_hat, d)
    b = net_odds(c)

    grid = _grid(f_max, n_segments)
    df = np.diff(grid)                                             # segment widths
    B_seg = np.tile(np.diff(np.log1p(-grid)), (n, 1))              # B_i increments
    D_full = np.log1p(np.outer(b, grid)) - np.log1p(-grid)[None, :]
    D_seg = np.diff(D_full, axis=1)                                # D_i increments, N x K
    obj_seg = B_seg + p_hat[:, None] * D_seg                       # nominal growth increments

    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, n - 1)
    m.K = pyo.RangeSet(0, n_segments - 1)
    m.Kb = pyo.RangeSet(0, n_segments - 2)                         # ordering binaries
    m.delta = pyo.Var(m.I, m.K, bounds=(0, 1))
    m.y = pyo.Var(m.I, m.Kb, domain=pyo.Binary)
    m.lam = pyo.Var(domain=pyo.NonNegativeReals)
    m.nu = pyo.Var(m.I, domain=pyo.NonNegativeReals)

    m.order_hi = pyo.Constraint(m.I, m.Kb, rule=lambda m, i, k: m.delta[i, k + 1] <= m.y[i, k])
    m.order_lo = pyo.Constraint(m.I, m.Kb, rule=lambda m, i, k: m.y[i, k] <= m.delta[i, k])
    m.budget = pyo.Constraint(expr=sum(float(df[k]) * m.delta[i, k] for i in m.I for k in m.K) <= budget)
    m.worst_case_row = pyo.Constraint(
        m.I,
        rule=lambda m, i: m.lam + m.nu[i] >= float(d[i]) * sum(float(D_seg[i, k]) * m.delta[i, k] for k in m.K),
    )
    m.obj = pyo.Objective(
        expr=sum(float(obj_seg[i, k]) * m.delta[i, k] for i in m.I for k in m.K)
        - gamma * m.lam - sum(m.nu[i] for i in m.I),
        sense=pyo.maximize,
    )

    solver = pyo.SolverFactory("appsi_highs")
    solver.config.time_limit = time_limit
    solver.config.mip_gap = mip_gap
    res = solver.solve(m, load_solutions=False)
    condition = res.solver.termination_condition
    if condition not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.maxTimeLimit):
        raise RuntimeError(f"robust Kelly MILP did not solve: {condition}")
    m.solutions.load_from(res)
    f = np.array([sum(df[k] * pyo.value(m.delta[i, k]) for k in range(n_segments)) for i in range(n)])
    f = np.clip(f, 0.0, f_max)
    return RobustKellyResult(
        f=f,
        objective=float(pyo.value(m.obj)),
        worst_case_growth=worst_case_growth(f, p_hat, c, d, gamma),
        gamma=float(gamma),
        status=str(condition),
        optimal=condition == pyo.TerminationCondition.optimal,
    )
