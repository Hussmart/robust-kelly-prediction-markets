"""Simulation study: what does budgeted robustness buy, and what does it cost?

The real backtest has ~1 bet per meeting, which cannot separate allocation methods. Here
the ground truth is known, so the *exact* expected log-growth of every allocation can be
computed (no outcome noise) and the price of robustness measured directly.

Data generating process (per instance, ``N`` independent bets):

* true win probability   ``p_i ~ U(0.25, 0.85)``
* market price           ``c_i = p_i - e_i`` with edge ``e_i ~ U(0.02, 0.12)``
* estimate               ``p^_i = p_i + noise_i`` where ``noise_i ~ N(0, 0.02)`` for every bet, and
  ``k_bad`` randomly chosen bets are additionally *over-estimated* by exactly their
  uncertainty radius ``d_i ~ U(0.06, 0.14)`` (the adversary of the Bertsimas-Sim set,
  realised at its vertices).

The allocator knows ``p^``, ``c`` and ``d`` but not which estimates are bad. Compared:

* naive Kelly (trusts ``p^``), robust Kelly for ``Gamma`` in ``GAMMAS``, equal weight.

Metric: exact expected log-growth ``g(f; p_true)`` of each allocation, averaged over
instances (plus its 5th percentile across instances and the share of instances with
negative growth). Outputs ``results/simulation_*.csv`` and ``figures/simulation_gamma.png``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optimization.naive_kelly import growth, kelly_fractions  # noqa: E402
from src.optimization.robust_kelly import solve_robust_kelly  # noqa: E402

N_BETS = 6
GAMMAS = (0, 1, 2, 3, 6)
K_BAD = (0, 1, 2, 4)
N_INSTANCES = 150
BUDGET, F_MAX = 0.6, 0.3


def make_instance(rng: np.random.Generator, k_bad: int) -> dict[str, np.ndarray]:
    """One random instance; ``k_bad`` estimates are over-estimated by their full radius."""
    p = rng.uniform(0.25, 0.85, N_BETS)
    c = np.clip(p - rng.uniform(0.02, 0.12, N_BETS), 0.05, 0.95)
    d = rng.uniform(0.06, 0.14, N_BETS)
    err = rng.normal(0, 0.02, N_BETS)
    bad = rng.choice(N_BETS, size=k_bad, replace=False) if k_bad else np.array([], dtype=int)
    err[bad] += d[bad]
    return {"p": p, "c": c, "d": d, "p_hat": np.clip(p + err, 0.02, 0.98)}


def evaluate(inst: dict[str, np.ndarray]) -> dict[str, float]:
    """Exact true expected log-growth of every strategy on one instance."""
    p, c, d, p_hat = inst["p"], inst["c"], inst["d"], inst["p_hat"]
    out = {"naive_kelly": growth(kelly_fractions(p_hat, c, BUDGET, F_MAX), p, c)}
    for g in GAMMAS:
        f = solve_robust_kelly(p_hat, c, d, g, BUDGET, F_MAX, n_segments=30).f
        out[f"robust_G{g}"] = growth(f, p, c)
    out["equal_weight"] = growth(np.full(N_BETS, min(BUDGET / N_BETS, F_MAX)), p, c)
    out["oracle_kelly"] = growth(kelly_fractions(p, c, BUDGET, F_MAX), p, c)
    return out


def main() -> None:
    """Run all scenarios and write tables and the figure."""
    rng = np.random.default_rng(2024)
    rows = []
    t0 = time.time()
    for k in K_BAD:
        for i in range(N_INSTANCES):
            res = evaluate(make_instance(rng, k))
            rows.extend({"k_bad": k, "instance": i, "strategy": s, "growth": v} for s, v in res.items())
        print(f"k_bad={k} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "simulation_raw.csv", index=False)

    summ = df.groupby(["k_bad", "strategy"]).growth.agg(
        mean="mean", sem=lambda x: x.std() / np.sqrt(len(x)), p05=lambda x: x.quantile(0.05),
        frac_negative=lambda x: (x < 0).mean()).reset_index()
    summ.to_csv(ROOT / "results" / "simulation_summary.csv", index=False)
    piv = summ.pivot(index="strategy", columns="k_bad", values="mean")
    print("\nMean true expected log-growth per round:")
    print(piv.round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = list(GAMMAS)
    for k in K_BAD:
        m = [piv.loc[f"robust_G{g}", k] for g in GAMMAS]
        axes[0].plot(x, m, "o-", label=f"{k} bad estimates")
    axes[0].set(xlabel=r"allocator uncertainty budget $\Gamma$", ylabel="mean true log-growth per round",
                title="Robust Kelly vs. how many estimates are actually bad")
    axes[0].legend(fontsize=8)
    for k in K_BAD:
        for strat, colour in [("naive_kelly", "C3"), ("equal_weight", "C2")]:
            axes[1].plot([k], [piv.loc[strat, k]], "s", color=colour, label=strat if k == 0 else None)
        best = max(GAMMAS, key=lambda g: piv.loc[f"robust_G{g}", k])
        axes[1].plot([k], [piv.loc[f"robust_G{best}", k]], "o", color="C0", label="best robust" if k == 0 else None)
    axes[1].set(xlabel="number of over-estimated probabilities (of 6)", ylabel="mean true log-growth per round",
                title="Strategies compared")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "simulation_gamma.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
