"""Generate the results blocks of README.md from ``results/*.csv``.

Every number quoted in the README's results sections comes from here, so it cannot drift from the
committed results. Blocks live between ``<!-- BEGIN:name -->`` and ``<!-- END:name -->`` markers.

Usage: ``python scripts/make_readme_tables.py`` rewrites the blocks; ``--check`` exits non-zero if the
README is out of date (used by a test and by CI).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
README = ROOT / "README.md"


def _csv(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(R / name, **kw)


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _f(x: float, nd: int = 3) -> str:
    return "n/a" if pd.isna(x) else f"{x:.{nd}f}"


def _ci(lo: float, hi: float, nd: int = 3) -> str:
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def block_headline() -> str:
    """The headline findings as bullets."""
    dec = _csv("decision_time_divergence.csv").iloc[0]
    cal = _csv("universe_calibration.csv").set_index("method")
    bets = _csv("universe_bets_overall.csv").iloc[0]
    rounds = _csv("universe_rounds.csv")
    per = _csv("universe_paired_bootstrap_risk.csv")
    sens = _csv("universe_sensitivity_paired.csv")
    sim = _csv("simulation_summary.csv").pivot(index="strategy", columns="k_bad", values="mean")
    n_markets = int(cal.loc["raw", "n"])
    excl = int((~((sens.lo <= 0) & (0 <= sens.hi))).sum())
    dd = per[(per.a == "robust_G3") & (per.b == "equal_weight") & (per.statistic == "max_drawdown")].iloc[0]
    loss = 1 - sim.loc["naive_kelly", 2] / sim.loc["naive_kelly", 0]
    return "\n".join([
        f"* **The two venues agree.** On the 20 FOMC meetings the mean absolute cross-venue gap 24 h before the "
        f"announcement is **{dec.mean_abs_div * 100:.2f}¢** (median {dec.median_abs_div * 100:.2f}¢); only "
        f"**{int(dec.pairs_ge_2c)} of {int(dec.pairs)}** pairs differ by 2¢ or more.",
        f"* **A market price is already a good probability.** On {n_markets:,} out-of-time predictions from a broad "
        f"universe, recalibrating with Platt scaling or isotonic regression *increases* the Brier score "
        f"({cal.loc['raw', 'brier']:.4f} raw, {cal.loc['platt', 'brier']:.4f} Platt, {cal.loc['isotonic', 'brier']:.4f} "
        f"isotonic). The 90% event-bootstrap interval of the Platt gain is "
        f"{_ci(cal.loc['platt', 'brier_gain_lo'], cal.loc['platt', 'brier_gain_hi'], 4)} (positive would mean it helps).",
        f"* **No exploitable edge.** The {int(bets.bets)} bets the calibrated model chose over "
        f"{len(rounds)} fortnightly rounds returned **{bets.mean_roi * 100:+.1f}%** per bet, 90% interval "
        f"[{bets.roi_lo * 100:+.1f}%, {bets.roi_hi * 100:+.1f}%], consistent with zero.",
        f"* **Robust Kelly does not beat naive Kelly on real data**, because there is no edge to protect. Of "
        f"{len(sens)} robust-vs-naive sensitivity comparisons, {excl} has a 90% interval excluding zero, about what "
        f"chance alone produces. Robust Kelly with Γ = 3 does have a much smaller maximum drawdown than equal weight "
        f"({dd.observed:+.2f}, 90% interval {_ci(dd.lo, dd.hi, 2)}).",
        f"* **The mechanism is verified where the truth is known.** In the simulation, naive Kelly loses "
        f"{loss * 100:.0f}% of its growth once two of six probability estimates are wrong, and robust Kelly with a matched "
        f"Γ recovers most of it.",
    ])


def block_universe_calibration() -> str:
    """Out-of-time calibration on the broad universe."""
    cal = _csv("universe_calibration.csv")
    rows = [[r.method, f"{r.brier:.4f}", f"{r.ece:.4f}", f"{r.log_loss:.4f}",
             "-" if pd.isna(r.brier_gain_lo) else _ci(r.brier_gain_lo, r.brier_gain_hi, 4)] for r in cal.itertuples()]
    return _md_table(["method", "Brier", "ECE", "log-loss", "90% CI of Brier gain over raw"], rows)


def block_universe_backtest() -> str:
    """Primary universe backtest table."""
    s = _csv("universe_backtest.csv", index_col=0)
    names = {"naive_kelly": "naive Kelly", "robust_G0": "robust Γ = 0", "robust_G1": "robust Γ = 1", "robust_G2": "robust Γ = 2",
             "robust_G3": "robust Γ = 3", "robust_G5": "robust Γ = 5", "robust_G8": "robust Γ = 8", "equal_weight": "equal weight",
             "random": "random subset"}
    rows = [[names[k], _f(v.cum_log_growth), _f(v.final_wealth, 2), _f(v.max_drawdown, 2), _f(v.sharpe_like, 2), _f(v.worst_round, 2)]
            for k, v in s.iterrows()]
    return _md_table(["strategy", "cum. log-growth", "final wealth", "max drawdown", "Sharpe-like", "worst round"], rows)


def block_universe_inference() -> str:
    """Paired-bootstrap differences in cumulative log-growth."""
    inf = _csv("universe_paired_bootstrap.csv")
    rows = [[f"`{r.a}` − `{r.b}`", _f(r.observed), _ci(r.lo, r.hi, 2), _f(r.prob_positive, 2)] for r in inf.itertuples()]
    return _md_table(["comparison", "observed", "90% paired-bootstrap CI", "P(difference > 0)"], rows)


def block_universe_sensitivity() -> str:
    """Selected sensitivity rows: naive vs robust in the most efficient subsets."""
    s = _csv("universe_sensitivity.csv")
    p = _csv("universe_sensitivity_paired.csv")
    variants = ["active_40changes", "fresh_3h", "cost_3c", "cost_5c", "isotonic"]
    rows = []
    for v in variants:
        g = s[s.variant == v].set_index("strategy").cum_log_growth
        d = p[(p.variant == v) & (p.a == "robust_G5")].iloc[0]
        rows.append([f"`{v}`", _f(g["naive_kelly"], 2), _f(g["robust_G5"], 2), _f(g["equal_weight"], 2),
                     f"{d.observed:+.2f} {_ci(d.lo, d.hi, 2)}"])
    return _md_table(["variant", "naive Kelly", "robust Γ = 5", "equal weight", "robust Γ = 5 − naive (90% CI)"], rows)


def block_simulation() -> str:
    """Simulation table."""
    piv = _csv("simulation_summary.csv").pivot(index="strategy", columns="k_bad", values="mean")
    order = [("oracle_kelly", "oracle Kelly (true p; upper bound)"), ("naive_kelly", "naive Kelly")] + [
        (f"robust_G{g}", f"robust Γ = {g}") for g in (1, 2, 3, 6)] + [("equal_weight", "equal weight")]
    rows = []
    for key, label in order:
        vals = piv.loc[key]
        best = {c: piv.loc[[k for k, _ in order if k != "oracle_kelly"], c].idxmax() for c in piv.columns}
        rows.append([label] + [(f"**{vals[c]:.4f}**" if best[c] == key else f"{vals[c]:.4f}") for c in piv.columns])
    return _md_table(["strategy"] + [f"k = {c}" for c in piv.columns], rows)


def block_fomc_findings() -> str:
    """FOMC anomaly-detection findings."""
    an = _csv("anomaly_summary.csv").iloc[0]
    pers = _csv("anomaly_persistence.csv")
    size = _csv("lazypredict_test_size.csv").iloc[0]
    scr = _csv("lazypredict_screen.csv", index_col=0)
    best = scr.drop(index="MajorityBaseline", errors="ignore")["Balanced Accuracy"].idxmax()

    def rate(det: str, flag: bool) -> str:
        r = pers[(pers.detector == det) & (pers.flag == flag)].iloc[0]
        return f"{r.persistent_rate * 100:.0f}% (n = {int(r.n)})"

    return "\n".join([
        f"* Over {int(an.snapshots_scored):,} walk-forward-scored snapshots: {int(an.if_flagged)} flagged by Isolation Forest, "
        f"{int(an.price_signal)} by the price signal, {int(an.liquidity_signal):,} pass the liquidity signal, "
        f"**{int(an.consensus)} by the joint consensus rule**.",
        f"* Persistence of ≥ 2¢ gaps 24 h later: liquid books {rate('liquidity_signal', True)} vs. illiquid "
        f"{rate('liquidity_signal', False)}, consistent with the rationale for the liquidity condition. But consensus-flagged "
        f"gaps persist {rate('consensus', True)} vs. {rate('consensus', False)} for the rest: **the joint rule did not "
        f"improve persistence in this sample**.",
        f"* LazyPredict screen (chronological test, {int(size.test_snapshots)} snapshots from {int(size.test_meetings)} "
        f"meetings, majority-class rate {size.test_persistent_rate * 100:.0f}%): best balanced accuracy "
        f"**{scr.loc[best, 'Balanced Accuracy']:.2f}** ({best}), i.e. near chance. No model is worth tuning on this sample.",
    ])


def block_fomc_calibration() -> str:
    """FOMC calibration table (pooled price)."""
    c = _csv("calibration_metrics.csv")
    c = c[c.venue == "pooled"].set_index("method")
    names = {"raw": "raw price", "platt": "Platt, prior sd 1", "platt_weak_prior": "Platt, weak prior", "isotonic": "isotonic"}
    rows = [[names[m], f"{c.loc[m, 'brier']:.4f}", f"{c.loc[m, 'ece']:.4f}", f"{c.loc[m, 'log_loss']:.4f}"]
            for m in ("raw", "platt", "platt_weak_prior", "isotonic")]
    return _md_table(["method", "Brier", "ECE", "log-loss"], rows)


def block_fomc_backtest() -> str:
    """FOMC backtest table."""
    s = _csv("backtest_primary.csv", index_col=0)
    names = {"naive_kelly": "naive Kelly", "robust_G0": "robust Γ = 0", "robust_G0.5": "robust Γ = 0.5", "robust_G1": "robust Γ = 1",
             "robust_G2": "robust Γ = 2", "robust_G3": "robust Γ = 3", "robust_G5": "robust Γ = 5", "equal_weight": "equal weight",
             "random": "random subset"}
    rows = [[names[k], _f(v.cum_log_growth), _f(v.final_wealth, 3), _f(v.max_drawdown, 2)] for k, v in s.iterrows()]
    return _md_table(["strategy", "cum. log-growth", "final wealth", "max drawdown"], rows)


BLOCKS = {
    "headline": block_headline, "universe_calibration": block_universe_calibration,
    "universe_backtest": block_universe_backtest, "universe_inference": block_universe_inference,
    "universe_sensitivity": block_universe_sensitivity, "simulation": block_simulation,
    "fomc_findings": block_fomc_findings, "fomc_calibration": block_fomc_calibration, "fomc_backtest": block_fomc_backtest,
}


def render(text: str) -> str:
    """Return ``text`` with every marked block regenerated."""
    for name, fn in BLOCKS.items():
        pattern = re.compile(rf"<!-- BEGIN:{name} -->.*?<!-- END:{name} -->", re.S)
        if not pattern.search(text):
            raise KeyError(f"marker for block {name!r} not found in README.md")
        block = "\n".join([f"<!-- BEGIN:{name} -->", fn(), f"<!-- END:{name} -->"])
        text = pattern.sub(lambda _m, block=block: block, text)
    return text


def main(check: bool = False) -> int:
    """Rewrite README blocks, or (with ``check``) report whether they are current."""
    old = README.read_text(encoding="utf-8")
    new = render(old)
    if check:
        return 0 if new == old else 1
    README.write_text(new, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
