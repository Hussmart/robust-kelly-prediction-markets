"""Stage 5: calibrate venue prices against realised outcomes and write metrics + figures.

Evaluation is **leave-one-meeting-out** (LOMO): for every meeting, the calibrator is fitted
on all *other* meetings and predicts the held-out one. Buckets of a meeting are mutually
exclusive and their snapshots are dependent, so holding out whole meetings is the
appropriate resampling unit. Metrics are computed on the pooled out-of-fold predictions.

Outputs:
* ``figures/calibration_reliability.png``   reliability diagram, raw vs Platt vs isotonic
* ``figures/calibration_bootstrap_band.png`` Platt / isotonic curves with cluster-bootstrap bands
* ``results/calibration_metrics.csv``        Brier / ECE / log-loss per venue and method
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration.isotonic import IsotonicCalibrator  # noqa: E402
from src.calibration.platt_scaling import PlattCalibrator  # noqa: E402
from src.calibration.reliability_diagrams import (  # noqa: E402
    brier_score, cluster_bootstrap_predict, expected_calibration_error, log_loss, reliability_table,
)

HORIZON_H, WINDOW_H = 24.0, 12.0


def calibration_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Long table ``(meeting, pair_id, venue, price, outcome)`` at snapshots near the horizon."""
    near = features[(features.hours_to_close - HORIZON_H).abs() <= WINDOW_H]
    parts = []
    for venue, col in (("polymarket", "p_poly"), ("kalshi", "p_kalshi"), ("pooled", "mid")):
        d = near.assign(mid=0.5 * (near.p_poly + near.p_kalshi))
        parts.append(pd.DataFrame({"meeting": d.meeting, "pair_id": d.pair_id, "venue": venue,
                                   "price": d[col], "outcome": d.outcome, "event_time": d.event_time}))
    return pd.concat(parts, ignore_index=True)


def lomo_predictions(df: pd.DataFrame, make) -> np.ndarray:
    """Out-of-fold calibrated probabilities, holding out one meeting at a time."""
    out = np.full(len(df), np.nan)
    for m in df.meeting.unique():
        te = (df.meeting == m).to_numpy()
        tr = ~te
        if df.outcome[tr].nunique() < 2:
            continue
        out[te] = make().fit(df.price[tr].to_numpy(), df.outcome[tr].to_numpy()).predict(df.price[te].to_numpy())
    return out


def main() -> None:
    """Run the calibration study and write tables and figures."""
    features = pd.read_parquet(ROOT / "data" / "sample" / "features.parquet")
    frame = calibration_frame(features)
    (ROOT / "results").mkdir(exist_ok=True)
    rows, curves = [], {}
    for venue, g in frame.groupby("venue"):
        g = g.reset_index(drop=True)
        preds = {
            "raw": g.price.to_numpy(),
            "platt_weak_prior": lomo_predictions(g, lambda: PlattCalibrator(ridge=1e-3)),
            "platt": lomo_predictions(g, lambda: PlattCalibrator(ridge=1.0)),
            "isotonic": lomo_predictions(g, lambda: IsotonicCalibrator(0.005, 0.995)),
        }
        ok = ~np.isnan(preds["platt"]) & ~np.isnan(preds["isotonic"]) & ~np.isnan(preds["platt_weak_prior"])
        y = g.outcome.to_numpy()[ok]
        for name, q in preds.items():
            q = q[ok]
            rows.append({"venue": venue, "method": name, "n": int(ok.sum()), "n_meetings": g.meeting.nunique(),
                         "brier": brier_score(q, y), "ece": expected_calibration_error(q, y, 8),
                         "log_loss": log_loss(q, y)})
            if venue == "pooled":
                t = reliability_table(q, y, 8)
                curves[name] = (t["mean_q"], t["freq"])
        if venue == "pooled":
            pooled = g[ok].reset_index(drop=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(ROOT / "results" / "calibration_metrics.csv", index=False)
    print(metrics.round(4).to_string(index=False))

    # Reliability diagram (pooled price, out-of-fold).
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for name, (mq, fr) in curves.items():
        ax.plot(mq, fr, "o-", ms=4, label=name)
    ax.set(xlabel="forecast probability", ylabel="observed frequency", xlim=(0, 1), ylim=(0, 1),
           title="Reliability (pooled price, leave-one-meeting-out)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "calibration_reliability.png", dpi=150)
    plt.close(fig)

    # Calibration curves with cluster-bootstrap bands, fitted on all data.
    grid = np.linspace(0.02, 0.98, 49)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), sharey=True)
    for ax, (name, make) in zip(axes, [("Platt (ridge 1)", lambda: PlattCalibrator(ridge=1.0)), ("Isotonic", lambda: IsotonicCalibrator(0.005, 0.995))]):
        med, lo, hi = cluster_bootstrap_predict(make, pooled.price.to_numpy(), pooled.outcome.to_numpy(),
                                                pooled.meeting.to_numpy(), grid, n_boot=500)
        ax.fill_between(grid, lo, hi, alpha=0.25, label="90% cluster-bootstrap band")
        ax.plot(grid, make().fit(pooled.price.to_numpy(), pooled.outcome.to_numpy()).predict(grid), label=f"{name} fit")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="identity")
        ax.set(xlabel="market price", title=f"{name} calibration curve", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("calibrated probability")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "calibration_bootstrap_band.png", dpi=150)
    plt.close(fig)

    for rho in (1e-3, 1.0):
        platt = PlattCalibrator(ridge=rho).fit(pooled.price.to_numpy(), pooled.outcome.to_numpy())
        print(f"Platt fit on pooled data, ridge={rho:g}: a = {platt.a_:.3f}, b = {platt.b_:.3f} (identity is a=1, b=0)")
    mid = pooled[(pooled.price > 0.15) & (pooled.price < 0.85)]
    print(f"snapshots with price in (0.15, 0.85): {len(mid)} of {len(pooled)} "
          f"({mid.pair_id.nunique()} pairs) -> the slope is identified by very few points")

if __name__ == "__main__":
    main()
