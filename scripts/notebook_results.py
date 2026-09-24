"""Cells of the final results notebook (built by ``make_notebooks.py results``)."""

from scripts.make_notebooks import SETUP

RESULTS = [
    ("md", "# 02 - Results\n\nEverything here is read from `results/` and `figures/`, which the scripts in `scripts/` "
           "regenerate. No number is typed in by hand.\n\n"
           "Two real-data experiments and one simulation:\n"
           "1. **Broad universe**: 2,720 resolved Polymarket binary markets, 57 fortnightly cohorts, ~11 bets per round. "
           "Selection uses only information available at each decision date.\n"
           "2. **FOMC pairs**: 63 equivalent Polymarket/Kalshi contracts on 20 meetings (cross-venue detection, "
           "1-3 bets per round).\n"
           "3. **Simulation** with known ground truth, to isolate what robust Kelly does."),
    ("code", SETUP + "from IPython.display import Image, display\nR = ROOT / 'results'\nF = ROOT / 'figures'"),
    ("md", "## 1. Broad universe: is a market price already a good probability?"),
    ("code", "pd.read_csv(R / 'universe_calibration.csv').round(4)"),
    ("md", "`brier_gain_*` is the 90% event-cluster bootstrap interval of `Brier(raw) - Brier(method)`. "
           "A positive value would mean the calibrator helps. Both intervals lie below zero: recalibrating "
           "hurts out of time."),
    ("code", "display(Image(F / 'universe_reliability.png', width=430))"),
    ("md", "## 2. Broad universe: allocation backtest"),
    ("code", "pd.read_csv(R / 'universe_bets_overall.csv').round(3)"),
    ("code", "pd.read_csv(R / 'universe_backtest.csv', index_col=0)[['cum_log_growth', 'final_wealth', 'max_drawdown', "
             "'sharpe_like', 'frac_rounds_positive', 'worst_round']].round(3)"),
    ("code", "display(Image(F / 'universe_wealth.png')); display(Image(F / 'universe_gamma.png'))"),
    ("md", "### Are the differences real? Paired bootstrap over rounds (a - b)"),
    ("code", "pd.read_csv(R / 'universe_paired_bootstrap.csv').round(3)"),
    ("code", "pd.read_csv(R / 'universe_paired_bootstrap_risk.csv').round(3)"),
    ("md", "### Sensitivity (cumulative log-growth)"),
    ("code", "s = pd.read_csv(R / 'universe_sensitivity.csv')\n"
             "s.pivot(index='strategy', columns='variant', values='cum_log_growth').round(2)"),
    ("md", "Robust Kelly minus naive Kelly per variant, with a 90% paired-bootstrap interval:"),
    ("code", "pd.read_csv(R / 'universe_sensitivity_paired.csv').round(3)"),
    ("md", "## 3. Simulation: the price of robustness (known ground truth)"),
    ("code", "sim = pd.read_csv(R / 'simulation_summary.csv')\n"
             "sim.pivot(index='strategy', columns='k_bad', values='mean').round(4)"),
    ("code", "display(Image(F / 'simulation_gamma.png'))"),
    ("md", "## 4. FOMC cross-venue study"),
    ("code", "pd.read_csv(R / 'decision_time_divergence.csv').round(4)"),
    ("code", "pd.read_csv(R / 'anomaly_persistence.csv').round(3)"),
    ("code", "cal = pd.read_csv(R / 'calibration_metrics.csv')\n"
             "cal[cal.venue == 'pooled'].round(4)"),
    ("code", "display(Image(F / 'calibration_bootstrap_band.png'))"),
    ("code", "pd.read_csv(R / 'backtest_primary.csv', index_col=0)[['cum_log_growth', 'final_wealth', 'max_drawdown', "
             "'frac_rounds_positive']].round(3)"),
    ("md", "The FOMC backtest earns its return by buying near-certain favourites (all won), which is carry, "
           "not evidence of an edge; see the README."),
]
