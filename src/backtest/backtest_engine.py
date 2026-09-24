"""Round-based walk-forward backtest engine.

One *round* is one FOMC meeting. At the decision time ``T - horizon_h`` (default 24 h
before the announcement) the engine

1. fits the probability calibrator using **only meetings that resolved before this
   decision time** (walk-forward, no look-ahead),
2. turns every mapped pair into a candidate bet: buy the cheaper venue's YES, or the NO
   of the dearer venue, whichever has the larger calibrated edge, paying an execution
   cost ``cost`` on top of the price,
3. optionally keeps only candidates flagged by the joint price-and-liquidity consensus
   rule (``graph_consensus.ConsensusRule``, also fitted on training data only),
4. sizes the bets with each strategy (naive Kelly, robust Kelly for several ``Gamma``,
   equal weight, random), and
5. settles them against the *realised* outcome with the true payoff structure, including
   the fact that buckets of one meeting are mutually exclusive.

Round log-returns are compounded; ``metrics.py`` turns them into the headline numbers.

The optimiser's model treats the bets of a round as independent (separable log-growth,
see ``naive_kelly``), which is an approximation for mutually exclusive buckets. The
settlement below does *not* make that approximation, so the reported P&L is what
would actually have happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.anomaly.graph_consensus import ConsensusRule
from src.backtest.metrics import summarize
from src.calibration.isotonic import IsotonicCalibrator
from src.calibration.platt_scaling import PlattCalibrator
from src.calibration.reliability_diagrams import cluster_bootstrap_predict
from src.optimization.naive_kelly import kelly_fractions, net_odds
from src.optimization.robust_kelly import solve_robust_kelly


@dataclass(frozen=True)
class BacktestConfig:
    """Parameters of the walk-forward experiment (all fixed *before* looking at results)."""

    horizon_h: float = 24.0            # decision time before the announcement
    horizon_tol_h: float = 3.0         # snapshot must be within this of the horizon
    train_window_h: float = 12.0       # calibrator uses snapshots in horizon +- this window
    min_train_meetings: int = 6
    cost: float = 0.01                 # execution cost added to the purchase price
    min_edge: float = 0.0              # minimum calibrated edge p_hat - c to be a candidate
    use_consensus: bool = False        # additionally require the joint anomaly flag
    z_min: float = 3.0
    calibrator: str = "platt"          # "platt" | "isotonic"
    platt_ridge: float = 1.0           # prior strength toward the identity map (see docs §4)
    interval: str = "laplace"          # "laplace" (Bayesian Platt) | "bootstrap" (cluster bootstrap)
    pair_weighting: bool = True        # each pair counts once, however many snapshots it has
    n_boot: int = 200
    alpha: float = 0.10                # two-sided bootstrap level -> 90% interval
    budget: float = 0.6                # max total fraction of wealth staked per round
    f_max: float = 0.3                 # per-bet cap
    seed: int = 0


@dataclass
class Round:
    """Candidate bets of one meeting, ready for allocation and settlement."""

    meeting: str
    pair_ids: list[str]
    sides: list[str]
    p_hat: np.ndarray
    d: np.ndarray
    price: np.ndarray
    wins: np.ndarray
    decision_ts: int
    n_train_meetings: int = 0


@dataclass
class BacktestResult:
    """Per-round log-returns of every strategy plus the rounds they were computed on."""

    returns: pd.DataFrame
    rounds: list[Round] = field(default_factory=list)
    config: BacktestConfig = field(default_factory=BacktestConfig)

    def summary(self) -> pd.DataFrame:
        """One row of headline metrics per strategy."""
        return pd.DataFrame({k: summarize(self.returns[k].to_numpy()) for k in self.returns}).T


def _make_calibrator(name: str, platt_ridge: float = 1.0):
    if name == "platt":
        return PlattCalibrator(ridge=platt_ridge)
    if name == "isotonic":
        return IsotonicCalibrator(floor=0.005, ceil=0.995)
    raise ValueError(f"unknown calibrator {name!r}")


def decision_snapshots(features: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """One snapshot per pair: the one whose ``hours_to_close`` is closest to the horizon."""
    f = features.assign(gap=(features.hours_to_close - cfg.horizon_h).abs())
    f = f[f.gap <= cfg.horizon_tol_h]
    return f.loc[f.groupby("pair_id").gap.idxmin()].drop(columns="gap")


def build_rounds(features: pd.DataFrame, cfg: BacktestConfig = BacktestConfig()) -> list[Round]:
    """Walk-forward construction of the candidate bets of every meeting."""
    feats = features.copy()
    feats["mid"] = 0.5 * (feats.p_poly + feats.p_kalshi)
    meeting_time = feats.groupby("meeting").event_time.first().sort_values()
    decisions = decision_snapshots(feats, cfg)
    win = feats[(feats.hours_to_close - cfg.horizon_h).abs() <= cfg.train_window_h]
    rounds: list[Round] = []
    for meeting, t_event in meeting_time.items():
        dec_ts = int(t_event.timestamp() - cfg.horizon_h * 3600)
        prior = [m for m, t in meeting_time.items() if t < pd.Timestamp(dec_ts, unit="s", tz="UTC")]
        if len(prior) < cfg.min_train_meetings:
            continue
        train = win[win.meeting.isin(prior)]
        cur = decisions[decisions.meeting == meeting]
        if train.outcome.nunique() < 2 or cur.empty:
            continue
        w = (1.0 / train.groupby("pair_id").pair_id.transform("size")).to_numpy() if cfg.pair_weighting else None
        cal = _make_calibrator(cfg.calibrator, cfg.platt_ridge)
        cal.fit(train.mid.to_numpy(), train.outcome.to_numpy(), sample_weight=w)
        p_yes = cal.predict(cur.mid.to_numpy())
        if cfg.interval == "laplace" and cfg.calibrator == "platt":
            lo, hi = cal.predict_interval(cur.mid.to_numpy(), cfg.alpha)
        else:
            _, lo, hi = cluster_bootstrap_predict(
                lambda: _make_calibrator(cfg.calibrator, cfg.platt_ridge), train.mid.to_numpy(),
                train.outcome.to_numpy(), train.meeting.to_numpy(), cur.mid.to_numpy(),
                n_boot=cfg.n_boot, alpha=cfg.alpha, seed=cfg.seed,
            )
        # YES on the cheaper venue vs NO on the dearer venue.
        c_yes = np.minimum(cur.p_poly, cur.p_kalshi).to_numpy() + cfg.cost
        c_no = 1.0 - np.maximum(cur.p_poly, cur.p_kalshi).to_numpy() + cfg.cost
        edge_yes, edge_no = p_yes - c_yes, (1 - p_yes) - c_no
        take_yes = edge_yes >= edge_no
        p_side = np.where(take_yes, p_yes, 1 - p_yes)
        d_side = np.where(take_yes, p_yes - lo, hi - p_yes)     # adverse deviation of p_side
        price = np.where(take_yes, c_yes, c_no)
        edge = p_side - price
        keep = (edge > cfg.min_edge) & (price < 0.99) & (price > 0.01)
        if cfg.use_consensus:
            rule = ConsensusRule.fit(train, z_min=cfg.z_min)
            keep &= rule.flag(cur)["consensus"].to_numpy()
        if not keep.any():
            rounds.append(Round(meeting, [], [], *(np.array([]),) * 4, dec_ts, len(prior)))
            continue
        outcome = cur.outcome.to_numpy()
        rounds.append(Round(
            meeting=meeting,
            pair_ids=cur.pair_id.to_numpy()[keep].tolist(),
            sides=np.where(take_yes, "YES", "NO")[keep].tolist(),
            p_hat=p_side[keep], d=np.maximum(d_side[keep], 0.0), price=price[keep],
            wins=np.where(take_yes, outcome == 1, outcome == 0)[keep],
            decision_ts=dec_ts, n_train_meetings=len(prior),
        ))
    return rounds


def settle(round_: Round, f: np.ndarray) -> float:
    """Realised log-return of stakes ``f`` on ``round_`` (0 if nothing is staked).

    A winning bet returns ``f * b`` and a losing bet loses its stake ``f``.
    """
    if len(f) == 0:
        return 0.0
    b = net_odds(round_.price)
    gain = np.where(round_.wins, f * b, -f)
    return float(np.log1p(gain.sum()))


def equal_weight(n: int, budget: float, f_max: float) -> np.ndarray:
    """Split the budget equally across ``n`` bets (each capped at ``f_max``)."""
    return np.full(n, min(budget / n, f_max)) if n else np.array([])


def run_backtest(
    rounds: list[Round],
    cfg: BacktestConfig = BacktestConfig(),
    gammas: tuple[float, ...] = (0.0, 1.0, 2.0),
    n_random: int = 500,
) -> BacktestResult:
    """Evaluate all allocation strategies on ``rounds``.

    Strategies: ``naive_kelly``, ``robust_G{g}`` for every ``g`` in ``gammas`` (``g`` is
    clipped to the round's number of bets ``N``), ``equal_weight``, and ``random`` (mean
    log-return over ``n_random`` random subsets with random stakes).
    """
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, float]] = []
    for r in rounds:
        n = len(r.p_hat)
        row: dict[str, float] = {"meeting": r.meeting, "n_bets": n}
        row["naive_kelly"] = settle(r, kelly_fractions(r.p_hat, r.price, cfg.budget, cfg.f_max)) if n else 0.0
        for g in gammas:
            f = solve_robust_kelly(r.p_hat, r.price, r.d, min(g, n), cfg.budget, cfg.f_max).f if n else np.array([])
            row[f"robust_G{g:g}"] = settle(r, f)
        row["equal_weight"] = settle(r, equal_weight(n, cfg.budget, cfg.f_max))
        if n:
            draws = []
            for _ in range(n_random):
                k = rng.integers(1, n + 1)
                idx = rng.choice(n, size=k, replace=False)
                f = np.zeros(n)
                f[idx] = np.minimum(rng.dirichlet(np.ones(k)) * cfg.budget, cfg.f_max)
                draws.append(settle(r, f))
            row["random"] = float(np.mean(draws))
        else:
            row["random"] = 0.0
        rows.append(row)
    df = pd.DataFrame(rows).set_index("meeting")
    return BacktestResult(returns=df.drop(columns="n_bets"), rounds=rounds, config=cfg)
