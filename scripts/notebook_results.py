"""Cells of the final results notebook (built by ``make_notebooks.py results``)."""

from scripts.make_notebooks import SETUP

RESULTS = [
    ("md", "# 02 - Results\n\nEverything here is read from `results/` and `figures/`, which the scripts in `scripts/` "
           "regenerate. No number is typed in by hand."),
    ("code", SETUP + "from IPython.display import Image, display\nR = ROOT / 'results'\nF = ROOT / 'figures'"),
    ("md", "## 1. Calibration (leave-one-meeting-out, 20 FOMC meetings)"),
    ("code", "cal = pd.read_csv(R / 'calibration_metrics.csv')\n"
             "cal.pivot(index='method', columns='venue', values='brier').round(4).join(\n"
             "    cal.pivot(index='method', columns='venue', values='ece').round(4), lsuffix='_brier', rsuffix='_ece')"),
    ("code", "display(Image(F / 'calibration_reliability.png', width=420)); display(Image(F / 'calibration_bootstrap_band.png'))"),
    ("md", "Only 16 of 246 near-decision snapshots have a price in (0.15, 0.85), so the slope of the calibration "
           "curve is identified by very few points; the isotonic fit is essentially a step function."),
    ("md", "## 2. Real-vs-noise screen (LazyPredict)"),
    ("code", "pd.read_csv(R / 'lazypredict_screen.csv').head(8).round(3)"),
    ("md", "## 3. Backtest (14 FOMC rounds, primary configuration)"),
    ("code", "pd.read_csv(R / 'backtest_primary.csv', index_col=0).round(4)"),
    ("code", "display(Image(F / 'backtest_wealth.png')); display(Image(F / 'backtest_gamma_sensitivity.png'))"),
    ("md", "### Sensitivity (cumulative log-growth; each column changes one setting)"),
    ("code", "s = pd.read_csv(R / 'backtest_sensitivity.csv')\n"
             "s.pivot(index='strategy', columns='variant', values='cum_log_growth').round(3)"),
    ("md", "## 4. Simulation: the price of robustness (known ground truth)"),
    ("code", "sim = pd.read_csv(R / 'simulation_summary.csv')\n"
             "sim.pivot(index='strategy', columns='k_bad', values='mean').round(4)"),
    ("code", "display(Image(F / 'simulation_gamma.png'))"),
]
