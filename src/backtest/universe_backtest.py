"""Walk-forward backtest on the broad Polymarket universe (14-day non-overlapping cohorts).

Round ``k`` = cohort ``k`` of ``collectors.universe``: all resolved binary markets whose
entry price is observed at decision date ``T_k`` and which resolve 7-14 days later. Because
every cohort resolves before ``T_{k+1}``, capital never overlaps and wealth compounds
sequentially.

At ``T_k`` the engine

1. fits the calibrator on cohorts ``< k`` only (all of them resolved before ``T_k``),
   weighting each market by ``1 / (markets of its event)`` so that a many-outcome event
   counts once,
2. for every market compares buying YES at ``price + cost`` with buying NO at
   ``1 - price + cost`` and takes the side with the larger calibrated edge,
3. keeps positive-edge candidates, at most one per event (the bets of one event are
   dependent, often mutually exclusive), and at most ``max_bets`` of them (largest edge
   first), then
4. hands ``(p_hat, d, price)`` to the allocators and settles with the realised outcomes.

The result object and the strategy set are shared with ``backtest_engine`` so both
experiments use identical allocators and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.backtest_engine import Round, _make_calibrator
from src.calibration.reliability_diagrams import cluster_bootstrap_predict


@dataclass(frozen=True)
class UniverseConfig:
    """Parameters of the universe backtest (fixed before looking at results)."""

    min_train_cohorts: int = 8         # burn-in, in cohorts of 14 days
    cost: float = 0.01                 # execution cost added to the purchase price
    min_edge: float = 0.0              # minimum calibrated edge p_hat - price to be a candidate
    max_bets: int = 12                 # at most this many bets per round (largest edge first)
    one_per_event: bool = True         # at most one bet per event
    calibrator: str = "platt"          # "platt" | "isotonic"
    platt_ridge: float = 1.0           # Gaussian prior strength toward the identity map
    interval: str = "laplace"          # "laplace" | "bootstrap" (bootstrap resamples events)
    alpha: float = 0.10                # two-sided level of the uncertainty interval
    n_boot: int = 200
    budget: float = 0.35               # max total fraction of wealth staked per round
    f_max: float = 0.10                # per-bet cap
    seed: int = 0


def _event_weights(df: pd.DataFrame) -> np.ndarray:
    """``1 / (number of markets of the same event)`` so each event has total weight one."""
    return (1.0 / df.groupby("event_key")["event_key"].transform("size")).to_numpy()


def calibrated_predictions(
    train: pd.DataFrame, target: pd.DataFrame, cfg: UniverseConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit on ``train`` and return ``(p_yes, lower, upper)`` for the prices of ``target``."""
    w = _event_weights(train)
    cal = _make_calibrator(cfg.calibrator, cfg.platt_ridge)
    cal.fit(train["price"].to_numpy(), train["outcome"].to_numpy(), sample_weight=w)
    p = target["price"].to_numpy()
    q = cal.predict(p)
    if cfg.interval == "laplace" and cfg.calibrator == "platt":
        lo, hi = cal.predict_interval(p, cfg.alpha)
    else:
        _, lo, hi = cluster_bootstrap_predict(
            lambda: _make_calibrator(cfg.calibrator, cfg.platt_ridge), train["price"].to_numpy(),
            train["outcome"].to_numpy(), train["event_key"].to_numpy(), p,
            n_boot=cfg.n_boot, alpha=cfg.alpha, seed=cfg.seed,
        )
    return q, lo, hi


def build_universe_rounds(universe: pd.DataFrame, cfg: UniverseConfig | None = None) -> list[Round]:
    """Walk-forward construction of the candidate bets of every cohort after the burn-in."""
    cfg = cfg or UniverseConfig()
    rounds: list[Round] = []
    cohorts = sorted(universe["cohort"].unique())
    for k in cohorts:
        train = universe[universe["cohort"] < k]
        cur = universe[universe["cohort"] == k]
        if train["cohort"].nunique() < cfg.min_train_cohorts or train["outcome"].nunique() < 2 or cur.empty:
            continue
        p_yes, lo, hi = calibrated_predictions(train, cur, cfg)
        c_yes = cur["price"].to_numpy() + cfg.cost
        c_no = 1.0 - cur["price"].to_numpy() + cfg.cost
        take_yes = (p_yes - c_yes) >= ((1 - p_yes) - c_no)
        p_side = np.where(take_yes, p_yes, 1 - p_yes)
        d_side = np.where(take_yes, p_yes - lo, hi - p_yes)
        price = np.where(take_yes, c_yes, c_no)
        edge = p_side - price
        outcome = cur["outcome"].to_numpy()
        cand = pd.DataFrame({
            "market_id": cur["market_id"].to_numpy(), "event_key": cur["event_key"].to_numpy(),
            "side": np.where(take_yes, "YES", "NO"), "p_hat": p_side, "d": np.maximum(d_side, 0.0),
            "price": price, "edge": edge, "win": np.where(take_yes, outcome == 1, outcome == 0),
        })
        cand = cand[(cand.edge > cfg.min_edge) & (cand.price > 0.01) & (cand.price < 0.99)]
        cand = cand.sort_values("edge", ascending=False)
        if cfg.one_per_event:
            cand = cand.drop_duplicates("event_key")
        cand = cand.head(cfg.max_bets)
        rounds.append(Round(
            meeting=f"C{int(k):02d}", pair_ids=cand.market_id.tolist(), sides=cand.side.tolist(),
            p_hat=cand.p_hat.to_numpy(), d=cand.d.to_numpy(), price=cand.price.to_numpy(),
            wins=cand.win.to_numpy(), decision_ts=int(cur["decision_ts"].iloc[0]),
            n_train_meetings=int(train["cohort"].nunique()),
        ))
    return rounds


def expanding_window_predictions(universe: pd.DataFrame, cfg: UniverseConfig | None = None) -> pd.DataFrame:
    """Out-of-time calibrated probabilities for every market after the burn-in.

    Returns the universe rows of the evaluated cohorts with columns ``raw``, ``platt`` and
    ``isotonic`` (each a probability of YES), used for the reliability study.
    """
    cfg = cfg or UniverseConfig()
    parts = []
    for k in sorted(universe["cohort"].unique()):
        train = universe[universe["cohort"] < k]
        cur = universe[universe["cohort"] == k]
        if train["cohort"].nunique() < cfg.min_train_cohorts or train["outcome"].nunique() < 2 or cur.empty:
            continue
        out = cur.copy()
        out["raw"] = cur["price"].to_numpy()
        for name in ("platt", "isotonic"):
            cal = _make_calibrator(name, cfg.platt_ridge)
            cal.fit(train["price"].to_numpy(), train["outcome"].to_numpy(), sample_weight=_event_weights(train))
            out[name] = cal.predict(cur["price"].to_numpy())
        parts.append(out)
    return pd.concat(parts, ignore_index=True)
