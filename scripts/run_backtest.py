"""Stage 8: walk-forward backtest of naive Kelly / robust Kelly / equal weight / random.

The primary configuration is ``BacktestConfig()`` (24 h horizon, 1c execution cost,
Platt calibration with a unit-variance prior and Laplace-approximation intervals, pair
weighting, budget 0.6, per-bet cap 0.3). Disclosure: the Laplace intervals and the pair
weighting were introduced *after* a first run showed that cluster-bootstrap intervals
collapse to +-0.2c when every favourite in the training sample won; that first
configuration is kept below as the ``bootstrap_no_weights`` sensitivity. Nothing else was
tuned on the results.

Outputs (``results/`` and ``figures/``): summary tables and the plots used in the README.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.backtest_engine import BacktestConfig, BacktestResult, build_rounds, run_backtest  # noqa: E402
from src.backtest.metrics import max_drawdown, sharpe_like, wealth_curve  # noqa: E402

GAMMAS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)


def run(features: pd.DataFrame, cfg: BacktestConfig, gammas=GAMMAS, n_random: int = 500) -> BacktestResult:
    """Build walk-forward rounds and evaluate all strategies."""
    return run_backtest(build_rounds(features, cfg), cfg, gammas=gammas, n_random=n_random)


def plot_wealth(res: BacktestResult, path: Path, title: str) -> None:
    """Wealth curves of the headline strategies."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for col, style in [("naive_kelly", "-"), ("robust_G1", "-"), ("robust_G2", "-"), ("equal_weight", "--"), ("random", ":")]:
        if col in res.returns:
            ax.plot(wealth_curve(res.returns[col].to_numpy()), style, marker="o", ms=3, label=col)
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set(xlabel="round (FOMC meeting, chronological)", ylabel="wealth (start = 1)", title=title, yscale="log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gamma(res: BacktestResult, path: Path) -> None:
    """Cumulative log-growth, drawdown and Sharpe-like as a function of Gamma."""
    cols = [c for c in res.returns if c.startswith("robust_G")]
    g = [float(c[8:]) for c in cols]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (name, fn) in zip(axes, [("cumulative log-growth", lambda r: r.sum()), ("max drawdown", max_drawdown),
                                     ("Sharpe-like (annualised)", sharpe_like)]):
        ax.plot(g, [fn(res.returns[c].to_numpy()) for c in cols], "o-", label="robust Kelly")
        ax.axhline(fn(res.returns["naive_kelly"].to_numpy()), color="C3", ls="--", label="naive Kelly")
        ax.axhline(fn(res.returns["equal_weight"].to_numpy()), color="C2", ls=":", label="equal weight")
        ax.set(xlabel=r"uncertainty budget $\Gamma$", title=name)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the primary backtest plus sensitivity analyses."""
    features = pd.read_parquet(ROOT / "data" / "sample" / "features.parquet")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    base = BacktestConfig()

    primary = run(features, base)
    n_bets = [len(r.p_hat) for r in primary.rounds]
    print(f"primary: {len(primary.rounds)} rounds, bets/round mean={np.mean(n_bets):.2f} max={max(n_bets)}")
    summary = primary.summary()
    summary.to_csv(out / "backtest_primary.csv")
    primary.returns.to_csv(out / "backtest_primary_round_returns.csv")
    print(summary.round(4).to_string())
    plot_wealth(primary, ROOT / "figures" / "backtest_wealth.png", "Wealth, primary configuration")
    plot_gamma(primary, ROOT / "figures" / "backtest_gamma_sensitivity.png")

    # Sensitivity analyses: each changes ONE setting of the primary configuration.
    variants: dict[str, BacktestConfig] = {
        "bootstrap_intervals": replace(base, interval="bootstrap"),
        "bootstrap_no_weights": replace(base, interval="bootstrap", pair_weighting=False),
        "weak_prior_platt": replace(base, platt_ridge=1e-3),
        "consensus_filter": replace(base, use_consensus=True, z_min=2.0),
        "isotonic": replace(base, calibrator="isotonic"),
        "cost_0.5c": replace(base, cost=0.005),
        "cost_2c": replace(base, cost=0.02),
        "horizon_48h": replace(base, horizon_h=48.0),
        "horizon_72h": replace(base, horizon_h=72.0),
        "budget_0.3": replace(base, budget=0.3, f_max=0.15),
    }
    rows = []
    for name, cfg in variants.items():
        res = run(features, cfg, gammas=(0.0, 1.0, 2.0, 5.0), n_random=200)
        s = res.summary()
        for strat in s.index:
            rows.append({"variant": name, "strategy": strat, "rounds": int(s.loc[strat, "rounds"]),
                         "bets_per_round": float(np.mean([len(r.p_hat) for r in res.rounds])) if res.rounds else np.nan,
                         **s.loc[strat, ["cum_log_growth", "max_drawdown", "sharpe_like", "worst_round"]].to_dict()})
    sens = pd.DataFrame(rows)
    sens.to_csv(out / "backtest_sensitivity.csv", index=False)
    print("\nSensitivity (cumulative log-growth):")
    print(sens.pivot(index="strategy", columns="variant", values="cum_log_growth").round(3).to_string())


if __name__ == "__main__":
    main()
