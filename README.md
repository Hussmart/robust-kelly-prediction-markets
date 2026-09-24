# Calibrated Mispricing Detection & Robust Capital Allocation for Prediction Markets

> Status: work in progress. This README grows stage by stage; sections for
> methodology, results and limitations are added as the corresponding code lands.

## Motivation

Prediction markets such as **Polymarket** and **Kalshi** price binary contracts
that pay \$1 if a real-world event happens (a Fed rate decision, an election
result, a macro data release). A contract's price is usually read as the market's
probability of the event. Three problems get in the way of treating those prices
as decision-grade probabilities:

1. **The same event can carry different prices on different venues.** The two
   platforms have different user bases, fee schedules, collateral (USDC vs USD)
   and access rules, so identical events can trade at different prices. Some of
   these gaps are real mispricings. Others are artefacts: stale last-trade
   prices, thin books, or wide spreads. Telling the two apart is an
   **anomaly-detection** problem, and a price gap alone isn't a reliable signal.

2. **Prices aren't necessarily calibrated probabilities.** Favourite–longshot
   bias, risk premia and liquidity effects mean that an event priced at 0.80 may
   not happen 80% of the time. Before a price is used for decisions it has to be
   **recalibrated** against realised outcomes, and the leftover **uncertainty**
   of that estimate has to be quantified.

3. **Estimated probabilities are noisy, and Kelly sizing is sensitive to them.**
   The Kelly criterion maximises long-run log-growth given the *true*
   probability. Plugging in a noisy estimate often leads to over-betting.
   **Robust optimisation** protects the allocation against adversarial
   estimation error. With Bertsimas–Sim budgeted uncertainty, at most Γ of the N
   estimates are assumed to deviate to their worst case at the same time. This
   keeps the model tractable (it can be written as a MILP) and makes its
   conservativeness tunable.

This project combines the three steps into one pipeline:

```
cross-platform data  →  features  →  anomaly detection (Isolation Forest + graph consensus)
                                   →  probability calibration (Platt / isotonic + bootstrap bands)
                                   →  robust Kelly allocation (Bertsimas–Sim, Pyomo + HiGHS)
                                   →  walk-forward backtest vs. naive Kelly and equal-weight
```

It applies the methodology of my undergraduate thesis (calibrating NLI
confidence scores, then robust MILP-based allocation under budgeted
uncertainty) to a new domain where outcomes are observable and the value of
calibration and robustness can be measured directly.

## Repository layout

```
src/collectors/    Polymarket + Kalshi API clients, parquet cache
src/features/      feature engineering on paired market time series
src/anomaly/       Isolation Forest, bipartite graph-consensus check
src/calibration/   Platt scaling, isotonic regression, reliability diagrams
src/optimization/  naive Kelly, robust (Bertsimas–Sim) Kelly in Pyomo
src/backtest/      walk-forward backtest engine and metrics
src/baseline/      LazyPredict classifier screen
notebooks/         exploratory analysis and final results
tests/             pytest suite
docs/              mathematical methodology
```
