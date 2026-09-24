"""Broad-universe study: out-of-time calibration and the allocation backtest with N >> 1.

Reads ``data/sample/universe.parquet`` (built by ``scripts/build_universe.py``).

1. Calibration, expanding window: for each 14-day cohort, calibrators are fitted on earlier
   cohorts only. Brier / ECE / log-loss of raw price vs Platt vs isotonic, with cluster
   (event) bootstrap intervals for the Brier improvement.
2. Backtest: naive Kelly vs robust Kelly (several Gamma) vs equal weight vs random, on the
   same rounds; paired bootstrap intervals for strategy differences.
3. Sensitivity: one setting changed at a time.

Outputs: ``results/universe_*.csv`` and ``figures/universe_*.png``.
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

from src.backtest.backtest_engine import BacktestConfig, run_backtest  # noqa: E402
from src.backtest.metrics import (  # noqa: E402
    max_drawdown, paired_bootstrap_diff, paired_bootstrap_stat, sharpe_like, wealth_curve,
)
from src.backtest.universe_backtest import (  # noqa: E402
    UniverseConfig, build_universe_rounds, expanding_window_predictions,
)
from src.calibration.reliability_diagrams import (  # noqa: E402
    brier_score, expected_calibration_error, log_loss, reliability_table,
)

RES, FIG = ROOT / "results", ROOT / "figures"
GAMMAS = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0)


def calibration_study(universe: pd.DataFrame, cfg: UniverseConfig) -> pd.DataFrame:
    """Out-of-time calibration metrics, a reliability figure and event-bootstrap Brier gains."""
    preds = expanding_window_predictions(universe, cfg)
    y = preds["outcome"].to_numpy()
    rows = [{"method": m, "n": len(y), "events": preds.event_key.nunique(),
             "brier": brier_score(preds[m].to_numpy(), y), "ece": expected_calibration_error(preds[m].to_numpy(), y, 10),
             "log_loss": log_loss(preds[m].to_numpy(), y)} for m in ("raw", "platt", "isotonic")]
    # Cluster bootstrap over events for the Brier improvement of each calibrator over the raw price.
    rng = np.random.default_rng(0)
    groups = {k: g.index.to_numpy() for k, g in preds.groupby("event_key")}
    keys = list(groups)
    sq = {m: (preds[m].to_numpy() - y) ** 2 for m in ("raw", "platt", "isotonic")}
    gains: dict[str, list[float]] = {"platt": [], "isotonic": []}
    for _ in range(1000):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, size=len(keys), replace=True)])
        for m in gains:
            gains[m].append(sq["raw"][idx].mean() - sq[m][idx].mean())
    table = pd.DataFrame(rows)
    for m in gains:
        lo, hi = np.quantile(gains[m], [0.05, 0.95])
        table.loc[table.method == m, ["brier_gain_lo", "brier_gain_hi"]] = [lo, hi]
    table.to_csv(RES / "universe_calibration.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for m in ("raw", "platt", "isotonic"):
        t = reliability_table(preds[m].to_numpy(), y, 12)
        ax.plot(t["mean_q"], t["freq"], "o-", ms=4, label=m)
    ax.set(xlabel="forecast probability", ylabel="observed frequency", xlim=(0, 1), ylim=(0, 1),
           title="Out-of-time reliability, broad universe")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "universe_reliability.png", dpi=150)
    plt.close(fig)
    return table


def backtest(universe: pd.DataFrame, cfg: UniverseConfig, gammas=GAMMAS, n_random: int = 300):
    """Build rounds and evaluate all strategies (allocator settings taken from ``cfg``)."""
    rounds = build_universe_rounds(universe, cfg)
    bt = BacktestConfig(budget=cfg.budget, f_max=cfg.f_max, seed=cfg.seed)
    return run_backtest(rounds, bt, gammas=gammas, n_random=n_random)


def bet_diagnostics(rounds, universe: pd.DataFrame) -> pd.DataFrame:
    """Bet-level table: side, price bin, predicted vs realised win rate and ROI, event-bootstrap CI."""
    ev = universe.set_index("market_id")["event_key"]
    rows = [(r.meeting, m, side, ph, c, bool(w)) for r in rounds
            for m, side, ph, c, w in zip(r.pair_ids, r.sides, r.p_hat, r.price, r.wins, strict=True)]
    b = pd.DataFrame(rows, columns=["round", "market_id", "side", "p_hat", "price", "win"])
    b["roi"] = np.where(b.win, 1 / b.price - 1, -1.0)
    b["event"] = b.market_id.map(ev)
    b["price_bin"] = pd.cut(b.price, [0, 0.1, 0.2, 0.4, 0.7, 1.0])
    table = (b.groupby(["side", "price_bin"], observed=True)
             .agg(n=("win", "size"), price=("price", "mean"), p_hat=("p_hat", "mean"), win_rate=("win", "mean"), roi=("roi", "mean"))
             .reset_index())
    table.to_csv(RES / "universe_bet_diagnostics.csv", index=False)
    g = b.groupby("event").roi.agg(["sum", "count"])
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(g), len(g))
        boots.append(g["sum"].to_numpy()[idx].sum() / g["count"].to_numpy()[idx].sum())
    lo, hi = np.quantile(boots, [0.05, 0.95])
    overall = pd.DataFrame({"bets": [len(b)], "events": [len(g)], "share_yes": [float((b.side == "YES").mean())],
                            "mean_price": [b.price.mean()], "mean_p_hat": [b.p_hat.mean()], "win_rate": [b.win.mean()],
                            "mean_roi": [b.roi.mean()], "roi_lo": [lo], "roi_hi": [hi]})
    overall.to_csv(RES / "universe_bets_overall.csv", index=False)
    print("\nBet diagnostics:", overall.round(3).to_dict("records")[0])
    return table


def main() -> None:
    """Run the calibration study, the primary backtest with inference, and sensitivity runs."""
    universe = pd.read_parquet(ROOT / "data" / "sample" / "universe.parquet")
    RES.mkdir(exist_ok=True)
    cfg = UniverseConfig()
    print(f"universe: {len(universe)} markets, {universe.cohort.nunique()} cohorts, "
          f"{universe.event_key.nunique()} events, yes-rate {universe.outcome.mean():.3f}")

    cal = calibration_study(universe, cfg)
    print("\nOut-of-time calibration:\n", cal.round(4).to_string(index=False))

    res = backtest(universe, cfg)
    n_bets = np.array([len(r.p_hat) for r in res.rounds])
    bets = pd.DataFrame({"round": [r.meeting for r in res.rounds], "n_bets": n_bets,
                         "mean_edge": [float(np.mean(r.p_hat - r.price)) if len(r.p_hat) else np.nan for r in res.rounds],
                         "mean_d": [float(np.mean(r.d)) if len(r.d) else np.nan for r in res.rounds],
                         "bet_win_rate": [float(np.mean(r.wins)) if len(r.wins) else np.nan for r in res.rounds]})
    bets.to_csv(RES / "universe_rounds.csv", index=False)
    print(f"\nprimary: {len(res.rounds)} rounds, bets/round mean {n_bets.mean():.1f} (max {n_bets.max()}), "
          f"rounds without bets {int((n_bets == 0).sum())}")
    bet_diagnostics(res.rounds, universe)
    summary = res.summary()
    summary.to_csv(RES / "universe_backtest.csv")
    res.returns.to_csv(RES / "universe_round_returns.csv")
    print(summary[["cum_log_growth", "final_wealth", "max_drawdown", "sharpe_like", "frac_rounds_positive", "worst_round"]].round(4).to_string())

    # Paired bootstrap over rounds.
    r = res.returns
    pairs = [("naive_kelly", "equal_weight"), ("naive_kelly", "random")] + [
        (f"robust_G{g:g}", "naive_kelly") for g in GAMMAS[1:]] + [(f"robust_G{g:g}", "equal_weight") for g in GAMMAS[1:]]
    rows = [{"a": a, "b": b, **paired_bootstrap_diff(r[a].to_numpy(), r[b].to_numpy())} for a, b in pairs]
    inf = pd.DataFrame(rows)
    inf.to_csv(RES / "universe_paired_bootstrap.csv", index=False)
    risk_rows = []
    for a_, b_ in [("robust_G3", "naive_kelly"), ("robust_G5", "naive_kelly"), ("robust_G3", "equal_weight"), ("robust_G5", "equal_weight")]:
        for name, fn in (("max_drawdown", max_drawdown), ("sharpe_like", sharpe_like)):
            risk_rows.append({"a": a_, "b": b_, "statistic": name, **paired_bootstrap_stat(r[a_].to_numpy(), r[b_].to_numpy(), fn)})
    pd.DataFrame(risk_rows).to_csv(RES / "universe_paired_bootstrap_risk.csv", index=False)
    print("\nPaired bootstrap of risk statistics (a - b):\n", pd.DataFrame(risk_rows).round(3).to_string(index=False))
    print("\nPaired bootstrap of cumulative log-growth differences (a - b):\n", inf.round(3).to_string(index=False))

    # Figures.
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for col, style in [("naive_kelly", "-"), ("robust_G2", "-"), ("robust_G5", "-"), ("equal_weight", "--"), ("random", ":")]:
        ax.plot(wealth_curve(r[col].to_numpy()), style, label=col)
    ax.set(xlabel="round (14-day cohort)", ylabel="wealth (start = 1)", title="Broad universe: wealth", yscale="log")
    ax.axhline(1, color="k", lw=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "universe_wealth.png", dpi=150)
    plt.close(fig)

    cols = [c for c in r.columns if c.startswith("robust_G")]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    for ax, (name, fn) in zip(axes, [("cumulative log-growth", lambda x: x.sum()), ("max drawdown", max_drawdown),
                                     ("Sharpe-like", sharpe_like)], strict=True):
        ax.plot([float(c[8:]) for c in cols], [fn(r[c].to_numpy()) for c in cols], "o-", label="robust Kelly")
        for base, col in (("naive_kelly", "C3"), ("equal_weight", "C2")):
            ax.axhline(fn(r[base].to_numpy()), color=col, ls="--", label=base)
        ax.set(xlabel=r"uncertainty budget $\Gamma$", title=name)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "universe_gamma.png", dpi=150)
    plt.close(fig)

    # Sensitivity: one change at a time.
    variants = {
        "cost_0.5c": replace(cfg, cost=0.005), "cost_2c": replace(cfg, cost=0.02),
        "cost_3c": replace(cfg, cost=0.03), "cost_5c": replace(cfg, cost=0.05),
        "min_edge_2c": replace(cfg, min_edge=0.02), "max_bets_6": replace(cfg, max_bets=6),
        "max_bets_20": replace(cfg, max_bets=20), "isotonic": replace(cfg, calibrator="isotonic"),
        "bootstrap_intervals": replace(cfg, interval="bootstrap"), "several_per_event": replace(cfg, one_per_event=False),
        "weak_prior": replace(cfg, platt_ridge=1e-3), "budget_0.2": replace(cfg, budget=0.2, f_max=0.05),
        "burn_in_4": replace(cfg, min_train_cohorts=4), "burn_in_12": replace(cfg, min_train_cohorts=12),
    }
    subsets = {"active_40changes": universe[universe.n_changes >= 40], "fresh_3h": universe[universe.price_age_h <= 3.0]}
    jobs = [(n, universe, c) for n, c in variants.items()] + [(n, sub, cfg) for n, sub in subsets.items()]
    rows = []
    paired: list[dict] = []
    for name, data, vcfg in jobs:
        vres = backtest(data, vcfg, gammas=(0.0, 2.0, 5.0), n_random=100)
        for robust in ("robust_G2", "robust_G5"):
            paired.append({"variant": name, "a": robust, "b": "naive_kelly",
                           **paired_bootstrap_diff(vres.returns[robust].to_numpy(), vres.returns["naive_kelly"].to_numpy(), n_boot=2000)})
        s = vres.summary()
        nb = np.mean([len(x.p_hat) for x in vres.rounds])
        for strat in s.index:
            rows.append({"variant": name, "strategy": strat, "rounds": int(s.loc[strat, "rounds"]), "bets_per_round": nb,
                         **s.loc[strat, ["cum_log_growth", "max_drawdown", "sharpe_like", "worst_round"]].to_dict()})
        print(f"sensitivity {name} done", flush=True)
    sens = pd.DataFrame(rows)
    sens.to_csv(RES / "universe_sensitivity.csv", index=False)
    pd.DataFrame(paired).to_csv(RES / "universe_sensitivity_paired.csv", index=False)
    print("\nSensitivity (cumulative log-growth):")
    print(sens.pivot(index="strategy", columns="variant", values="cum_log_growth").round(3).to_string())


if __name__ == "__main__":
    main()
