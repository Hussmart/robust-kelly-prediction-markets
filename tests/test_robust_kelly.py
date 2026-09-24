"""Correctness tests for the Bertsimas-Sim robust Kelly MILP.

The checks that matter most (they verify the *duality-based reformulation*, not just that a
solver returns something):

* Gamma = 0  ->  ordinary Kelly.
* Gamma = N  ->  Kelly evaluated at the worst-case probabilities p_hat - d.
* For 0 < Gamma < N the MILP optimum equals an independent solution of the max-min problem
  (vertex enumeration of the uncertainty set + SLSQP on the epigraph form).
* The optimal worst-case value is non-increasing in Gamma, and the robust stakes never do
  worse in the worst case than naive Kelly stakes.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.optimize import minimize

from src.optimization.naive_kelly import growth, kelly_fractions, net_odds
from src.optimization.robust_kelly import clip_deviation, solve_robust_kelly, worst_case_growth

P_HAT = np.array([0.62, 0.45, 0.30, 0.75])
PRICE = np.array([0.50, 0.38, 0.22, 0.70])
DEV = np.array([0.10, 0.06, 0.12, 0.04])
F_MAX = 0.5


# ---------------------------------------------------------------- naive Kelly
def test_classical_kelly_fraction_single_bet():
    p, c = 0.6, 0.5                        # b = 1 -> f* = p - (1-p)/b = 0.2
    f = kelly_fractions(np.array([p]), np.array([c]), budget=1.0)
    assert f[0] == pytest.approx(0.2, abs=1e-9)


def test_kelly_zero_for_negative_edge():
    assert kelly_fractions(np.array([0.4]), np.array([0.5]))[0] == 0.0


def test_kelly_respects_budget_and_cap():
    p, c = np.array([0.9, 0.9, 0.9]), np.array([0.5, 0.5, 0.5])
    f = kelly_fractions(p, c, budget=0.3, f_max=0.5)
    assert f.sum() == pytest.approx(0.3, abs=1e-9)
    assert kelly_fractions(np.array([0.99]), np.array([0.5]), f_max=0.4)[0] == pytest.approx(0.4)


def test_kelly_budget_solution_is_optimal_vs_random_feasible_points():
    rng = np.random.default_rng(0)
    f_star = kelly_fractions(P_HAT, PRICE, budget=0.4, f_max=F_MAX)
    g_star = growth(f_star, P_HAT, PRICE)
    for _ in range(2000):
        f = rng.dirichlet(np.ones(len(P_HAT) + 1))[:-1] * 0.4
        assert growth(f, P_HAT, PRICE) <= g_star + 1e-9


# ---------------------------------------------------------------- reductions
def test_gamma_zero_reduces_to_naive_kelly():
    rob = solve_robust_kelly(P_HAT, PRICE, DEV, gamma=0.0, f_max=F_MAX)
    naive = kelly_fractions(P_HAT, PRICE, budget=1.0, f_max=F_MAX)
    assert rob.status.endswith("optimal")
    np.testing.assert_allclose(rob.f, naive, atol=0.02)
    assert growth(rob.f, P_HAT, PRICE) == pytest.approx(growth(naive, P_HAT, PRICE), abs=1e-4)


def test_gamma_n_reduces_to_worst_case_kelly():
    d = clip_deviation(P_HAT, DEV)
    rob = solve_robust_kelly(P_HAT, PRICE, DEV, gamma=len(P_HAT), f_max=F_MAX)
    worst = kelly_fractions(P_HAT - d, PRICE, budget=1.0, f_max=F_MAX)
    np.testing.assert_allclose(rob.f, worst, atol=0.02)
    assert growth(rob.f, P_HAT - d, PRICE) == pytest.approx(growth(worst, P_HAT - d, PRICE), abs=1e-4)


def test_gamma_above_n_equals_gamma_n():
    a = solve_robust_kelly(P_HAT, PRICE, DEV, gamma=len(P_HAT), f_max=F_MAX)
    b = solve_robust_kelly(P_HAT, PRICE, DEV, gamma=len(P_HAT) + 3, f_max=F_MAX)
    np.testing.assert_allclose(a.f, b.f, atol=1e-6)


def test_zero_deviation_makes_gamma_irrelevant():
    zero = np.zeros(len(P_HAT))
    a = solve_robust_kelly(P_HAT, PRICE, zero, gamma=0, f_max=F_MAX)
    b = solve_robust_kelly(P_HAT, PRICE, zero, gamma=3, f_max=F_MAX)
    np.testing.assert_allclose(a.f, b.f, atol=1e-6)


# ---------------------------------------------------------------- independent check of the duality
def _maxmin_reference(p_hat, c, d, gamma, budget=1.0, f_max=F_MAX):
    """Solve max_f min_{p in U(gamma)} g(f; p) by vertex enumeration + SLSQP (integer gamma)."""
    n = len(p_hat)
    d = clip_deviation(p_hat, d)
    vertices = [np.array([1.0 if i in S else 0.0 for i in range(n)])
                for S in itertools.combinations(range(n), int(gamma))]
    cons = [{"type": "ineq", "fun": lambda x, z=z: growth(x[:n], p_hat - d * z, c) - x[n]} for z in vertices]
    cons.append({"type": "ineq", "fun": lambda x: budget - x[:n].sum()})
    x0 = np.concatenate([np.full(n, 0.01), [-1.0]])
    res = minimize(lambda x: -x[n], x0, constraints=cons, bounds=[(0, f_max)] * n + [(None, None)],
                   method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    return res.x[:n], -res.fun


@pytest.mark.parametrize("gamma", [1, 2, 3])
def test_milp_matches_independent_maxmin_solution(gamma):
    p, c, d = P_HAT[:3], PRICE[:3], DEV[:3]
    rob = solve_robust_kelly(p, c, d, gamma=gamma, f_max=F_MAX, n_segments=60)
    _, ref_value = _maxmin_reference(p, c, d, gamma)
    assert rob.worst_case_growth == pytest.approx(ref_value, abs=2e-4)
    assert rob.objective == pytest.approx(ref_value, abs=2e-3)   # piecewise-linear model value


def test_worst_case_growth_matches_vertex_enumeration():
    rng = np.random.default_rng(4)
    f = rng.uniform(0, 0.2, size=4)
    d = clip_deviation(P_HAT, DEV)
    for gamma in (0, 1, 2, 4):
        verts = [P_HAT - d * np.array([1.0 if i in S else 0.0 for i in range(4)])
                 for S in itertools.combinations(range(4), gamma)]
        expected = min(growth(f, v, PRICE) for v in verts)
        assert worst_case_growth(f, P_HAT, PRICE, DEV, gamma) == pytest.approx(expected, abs=1e-12)


def test_fractional_gamma_interpolates_between_integers():
    f = np.array([0.1, 0.05, 0.02, 0.2])
    lo, hi = (worst_case_growth(f, P_HAT, PRICE, DEV, g) for g in (1, 2))
    mid = worst_case_growth(f, P_HAT, PRICE, DEV, 1.5)
    assert mid == pytest.approx((lo + hi) / 2, abs=1e-12)


# ---------------------------------------------------------------- structural properties
def test_optimal_worst_case_value_nonincreasing_in_gamma():
    values = [solve_robust_kelly(P_HAT, PRICE, DEV, gamma=g, f_max=F_MAX).objective for g in range(5)]
    assert all(a >= b - 1e-9 for a, b in zip(values, values[1:]))


def test_robust_stakes_beat_naive_stakes_in_the_worst_case():
    naive = kelly_fractions(P_HAT, PRICE, f_max=F_MAX)
    for gamma in (1, 2, 4):
        rob = solve_robust_kelly(P_HAT, PRICE, DEV, gamma=gamma, f_max=F_MAX)
        assert (worst_case_growth(rob.f, P_HAT, PRICE, DEV, gamma)
                >= worst_case_growth(naive, P_HAT, PRICE, DEV, gamma) - 1e-4)


def test_more_uncertainty_shrinks_total_stake():
    totals = [solve_robust_kelly(P_HAT, PRICE, DEV, gamma=g, f_max=F_MAX).f.sum() for g in (0, 4)]
    assert totals[1] < totals[0]


def test_robust_solution_respects_constraints():
    rob = solve_robust_kelly(P_HAT, PRICE, DEV, gamma=2, budget=0.25, f_max=0.15)
    assert rob.f.sum() <= 0.25 + 1e-8 and rob.f.max() <= 0.15 + 1e-8 and rob.f.min() >= 0.0


def test_bet_with_no_edge_gets_zero_stake():
    rob = solve_robust_kelly(np.array([0.4, 0.7]), np.array([0.5, 0.5]), np.array([0.05, 0.05]), gamma=1)
    assert rob.f[0] == 0.0 and rob.f[1] > 0.0


def test_input_validation():
    with pytest.raises(ValueError):
        solve_robust_kelly(P_HAT, PRICE[:2], DEV, gamma=1)
    with pytest.raises(ValueError):
        solve_robust_kelly(P_HAT, PRICE, DEV, gamma=-1)
    with pytest.raises(ValueError):
        solve_robust_kelly(P_HAT, PRICE, DEV, gamma=1, f_max=1.0)


def test_net_odds():
    np.testing.assert_allclose(net_odds(np.array([0.5, 0.25])), [1.0, 3.0])


def test_result_reports_optimality_and_time_limit_is_not_silent():
    rob = solve_robust_kelly(P_HAT, PRICE, DEV, gamma=1, f_max=F_MAX)
    assert rob.optimal and rob.status.endswith("optimal")
