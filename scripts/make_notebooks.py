"""Build and execute the project notebooks.

Usage: ``python scripts/make_notebooks.py anomalies`` or ``... results``.
Notebooks only read the committed files in ``data/sample`` and ``results``, so they run
offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

SETUP = """import sys, pathlib
ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == 'notebooks' else pathlib.Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option('display.width', 200)
"""

ANOMALIES = [
    ("md", "# 01 - Cross-venue divergences and flagged anomalies\n\n"
           "Exploratory look at the paired Polymarket/Kalshi FOMC contracts, then the two detectors: "
           "Isolation Forest scores and the joint price-AND-liquidity consensus rule. "
           "All scores are walk-forward (fitted on earlier meetings only)."),
    ("code", SETUP + "an = pd.read_parquet(ROOT / 'data' / 'sample' / 'anomalies.parquet')\n"
                     "print(an.shape, an.pair_id.nunique(), 'pairs', an.meeting.nunique(), 'meetings')\n"
                     "an[['abs_div', 'logit_div', 'spread_kalshi', 'volume_poly', 'volume_kalshi']].describe().round(4)"),
    ("md", "## How large are the divergences?\nMost gaps are about one cent, comparable to the spread and fee."),
    ("code", "fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))\n"
             "ax[0].hist(an.abs_div * 100, bins=40); ax[0].axvline(2, color='r', ls='--', label='2c')\n"
             "ax[0].set(xlabel='|Polymarket - Kalshi| (cents)', title='Divergence distribution'); ax[0].legend()\n"
             "b = an.assign(bin=pd.cut(an.hours_to_close, [0, 12, 24, 48, 96, 192, 430]))\n"
             "b.groupby('bin', observed=True).abs_div.mean().mul(100).plot.bar(ax=ax[1])\n"
             "ax[1].set(xlabel='hours to announcement', ylabel='mean |div| (cents)', title='Divergence vs. time to close')\n"
             "plt.tight_layout(); plt.show()"),
    ("md", "## Detector counts\n`if_flag` = top 5% Isolation-Forest score of the training data; "
           "`consensus` = price signal AND liquidity signal."),
    ("code", "sc = an.dropna(subset=['if_score'])\n"
             "pd.Series({'scored snapshots': len(sc), 'IF flagged': int(sc.if_flag.sum()),\n"
             "           'price signal': int(sc.price_signal.astype(bool).sum()),\n"
             "           'liquidity signal': int(sc.liquidity_signal.astype(bool).sum()),\n"
             "           'consensus (both)': int(sc.consensus.astype(bool).sum()),\n"
             "           '|div| >= 2c': int((sc.abs_div >= 0.02).sum())})"),
    ("md", "## Are flagged divergences more persistent?\nLabel = the >=2c gap is still there, same sign and >= half size, "
           "24h later. The counts per group matter as much as the rates: the samples are small."),
    ("code", "lab = sc.dropna(subset=['persistent'])\n"
             "rows = []\n"
             "for name in ['if_flag', 'price_signal', 'liquidity_signal', 'consensus']:\n"
             "    for val, g in lab.groupby(lab[name].astype(bool)):\n"
             "        rows.append({'detector': name, 'flag': val, 'n': len(g), 'persistent_rate': round(g.persistent.mean(), 3)})\n"
             "pd.DataFrame(rows)"),
    ("code", "top = sc[sc.consensus.astype(bool)].sort_values('abs_div', ascending=False).head(12)\n"
             "top[['pair_id', 'hours_to_close', 'p_poly', 'p_kalshi', 'div', 'volume_poly', 'volume_kalshi', 'if_score', 'persistent']].round(3)"),
    ("md", "## Example: a flagged pair over time"),
    ("code", "pid = top.pair_id.iloc[0]\n"
             "g = an[an.pair_id == pid].sort_values('hours_to_close', ascending=False)\n"
             "fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))\n"
             "ax[0].plot(g.hours_to_close, g.p_poly, label='Polymarket'); ax[0].plot(g.hours_to_close, g.p_kalshi, label='Kalshi')\n"
             "ax[0].invert_xaxis(); ax[0].set(title=pid, xlabel='hours to close', ylabel='YES price'); ax[0].legend()\n"
             "ax[1].plot(g.hours_to_close, g['div'] * 100); ax[1].axhline(0, color='k', lw=.5); ax[1].invert_xaxis()\n"
             "ax[1].set(xlabel='hours to close', ylabel='divergence (cents)'); plt.tight_layout(); plt.show()"),
]


def build(cells: list[tuple[str, str]], path: Path) -> None:
    """Assemble ``cells``, execute the notebook in place and save it to ``path``."""
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in cells]
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
    path.parent.mkdir(exist_ok=True)
    nbf.write(nb, path)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "anomalies":
        build(ANOMALIES, NB_DIR / "01_anomalies.ipynb")
    else:
        from scripts.notebook_results import RESULTS
        build(RESULTS, NB_DIR / "02_results.ipynb")
