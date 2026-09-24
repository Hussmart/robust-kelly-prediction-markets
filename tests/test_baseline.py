"""Tests for the LazyPredict screen helpers (the screen itself is exercised by the script)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.baseline.lazypredict_screen import chronological_split, prepare_screen_data


def _frame() -> pd.DataFrame:
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    rows = [dict(meeting=f"m{m}", event_time=t0 + pd.Timedelta(days=40 * m), persistent=float(m % 2), k=i)
            for m in range(10) for i in range(3)]
    return pd.DataFrame(rows)


def test_split_is_chronological_by_meeting_with_no_overlap():
    train, test = chronological_split(_frame(), test_frac=0.3)
    assert set(train.meeting).isdisjoint(test.meeting)
    assert train.event_time.max() < test.event_time.min()
    assert test.meeting.nunique() == 3


def test_embargo_drops_training_rows_close_to_test_period():
    f = _frame()
    f.loc[f.meeting == "m6", "event_time"] = f.loc[f.meeting == "m7", "event_time"].iloc[0] - pd.Timedelta(hours=5)
    train, _ = chronological_split(f, test_frac=0.3, embargo_h=24.0)
    assert "m6" not in set(train.meeting)


def test_prepare_drops_unlabelled_rows():
    f = pd.DataFrame({"persistent": [1.0, np.nan], "logit_div": [0.1, 0.2], "volume_poly": [1.0, 1.0],
                      "volume_kalshi": [1.0, 1.0], "ofi_poly": [0.0, 0.0], "ofi_kalshi": [0.0, 0.0],
                      "hours_to_close": [10.0, 10.0], "stale_poly_h": [1.0, 1.0], "stale_kalshi_h": [1.0, 1.0]})
    assert len(prepare_screen_data(f)) == 1
