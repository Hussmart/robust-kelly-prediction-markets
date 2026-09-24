"""Smoke tests: the study scripts' helper functions run end to end on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import run_universe
from tests.test_universe import _universe


def test_universe_calibration_study_writes_table_and_figure(tmp_path, monkeypatch):
    monkeypatch.setattr(run_universe, "RES", tmp_path)
    monkeypatch.setattr(run_universe, "FIG", tmp_path)
    table = run_universe.calibration_study(_universe(n_cohorts=12, per_cohort=80), run_universe.UniverseConfig(min_train_cohorts=6))
    assert set(table.method) == {"raw", "platt", "isotonic"}
    assert (tmp_path / "universe_calibration.csv").exists() and (tmp_path / "universe_reliability.png").exists()
    lo, hi = table.loc[table.method == "platt", ["brier_gain_lo", "brier_gain_hi"]].iloc[0]
    assert lo <= hi and np.isfinite([lo, hi]).all()


def test_universe_backtest_helper_returns_all_strategies():
    res = run_universe.backtest(_universe(n_cohorts=10), run_universe.UniverseConfig(min_train_cohorts=6), gammas=(0.0, 2.0), n_random=5)
    assert {"naive_kelly", "robust_G0", "robust_G2", "equal_weight", "random"} <= set(res.returns.columns)
    assert isinstance(res.summary(), pd.DataFrame)
