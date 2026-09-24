# Audit log: bugs and weaknesses found, and what was done

A self-review of the first complete version of the pipeline. Each item says what was wrong,
how it was found, its effect on results, and the regression test that now guards it.

## Bugs that changed results

| # | Where | Problem | Effect | Guard |
|---|---|---|---|---|
| 1 | `collectors/polymarket.py: list_resolved_markets` | Assumed a Gamma page size of 500 and stopped at the first short page. The API caps pages at 100, so the listing silently returned **100 markets instead of thousands**; later, beyond offset ~2,000 the API answers HTTP 422. | Any broad study would have run on a 100-market sample without any error. | Month-window listing that ends a window only on an empty page or HTTP 422 (`GAMMA_PAGE`, `GAMMA_MAX_OFFSET`). |
| 2 | `features/feature_engineering.py: kalshi_implied_price` | A quote with `bid == 0` was treated as invalid, so long-shot contracts (quote 0.00 / 0.02) fell back to stale last trades. | Long-shot prices were biased; after the fix the FOMC decision-time mean divergence moved from 0.80¢ to 0.65¢, pairs with a gap ≥ 2¢ from 3 to 4, and 5 more snapshots became usable. All FOMC results were regenerated. | `test_kalshi_zero_bid_quote_is_valid_but_empty_book_is_not` |
| 3a | `collectors/universe.py` (caught before it was ever committed) | Cohorts were first defined by each market's *actual* resolution time. Early-resolving markets are disproportionately YES, so the universe itself would have leaked the outcome. | Would have produced a spurious "favourite–longshot" edge. | Cohorts use the scheduled `end_date` only and require the market to be open at the decision date; `test_cohorts_use_only_the_scheduled_end_date_not_the_actual_resolution`. |
| 3b | `collectors/universe.py` (caught before it was ever committed) | The universe was first restricted to markets with lifetime volume ≥ \$100k. Lifetime volume is future information: a cheap market that goes on to resolve YES trades at high prices and so accumulates more dollar volume than one that drifts to zero. In that exploratory build cheap markets resolved YES 3–6¢ more often than priced in the higher-volume quintiles (−1¢ in the lowest), and the pipeline reported **+22% return per bet** (90% interval +7% to +38%) and 9× final wealth for naive Kelly. | A completely spurious edge. It disappeared (bets returned +1.5%, interval −5.8% to +8.8%) once the population was all resolved binary markets, sampled at random and filtered only on information available at the decision date. That exploratory build was not kept. | `list_resolved_markets(volume_min=None)`, ex-ante activity filter (`entry_features`), `test_entry_features_use_only_data_up_to_the_decision_date`, `test_activity_is_measured_by_price_changes_not_by_the_number_of_grid_points`. |
| 3c | `collectors/universe.py` | The first activity measure counted hourly grid points, but the CLOB history repeats the last price on every hour, so every market looked equally active (about 144 points). | Would have made the activity filter a no-op. | Activity is the number of price *changes* and the age of the last change. |
| 4 | `backtest_engine` (found in an earlier run, disclosed in the README) | Cluster-bootstrap intervals collapse to ±0.2¢ when every favourite in the training sample won, so the "uncertainty set" carried no uncertainty. | Robust Kelly appeared identical to naive Kelly. | Bayesian Platt with a Laplace interval (`PlattCalibrator.predict_interval`), Monte Carlo coverage test. |

## Latent bugs and robustness fixes (no change in reported numbers)

| # | Where | Problem | Fix |
|---|---|---|---|
| 5 | `optimization/robust_kelly.py` | The solver's termination status was never checked, so a time-limit or infeasible solve could return garbage stakes silently. | Non-optimal/non-time-limit termination raises; `RobustKellyResult.optimal` reports whether optimality was proven. |
| 6 | `collectors/kalshi.py: candlesticks` | Lambdas captured the loop variable `params` by reference (works only because they are called immediately). | Bound with default arguments. |
| 7 | `collectors/polymarket.py: _normalise_trades` | Trade de-duplication ignored the wallet, so two identical fills in one transaction by different accounts would merge into one. | Wallet is part of the de-duplication key before it is dropped. |
| 8 | `backtest/metrics.py` | `win_rate` counted rounds with no bet as losses. | Renamed `frac_rounds_positive` and documented. |
| 9 | `backtest_engine.py` | Dataclass instances as default arguments (`cfg=BacktestConfig()`). | `None` defaults. |
| 10 | `anomaly/graph_consensus.py` | Docstring described connectivity that the graph does not have. | Rewritten; `validate_graph` asserts the actual invariant. |
| 11 | typing/lint | 10 mypy errors, 6 ruff findings, 11 public callables without docstrings. | Fixed; ruff runs in CI. |

## Weaknesses of the first version, and how they were addressed

* **Numbers in the README were not traceable to committed files** (persistence rates, decision-time
  divergence, LazyPredict test size). They are now written to `results/` by the scripts.
* **The real-data backtest was degenerate**: about one bet per round, every bet a near-certain
  favourite, no losing round. It could not compare allocation methods. A second experiment was
  added on a broad universe of resolved markets (see `docs/methodology.md` §7) with about ten
  concurrent bets per round, real losses, and paired-bootstrap intervals for strategy
  differences.
* **No continuous integration, license or dependency lock**: added.

## Known limitations that remain

See the README's *Limitations* section. In short: small samples, Polymarket price series
semantics (last trade vs midpoint) are not documented by the API, a constant execution cost stands in
for fees and slippage, and the optimiser's separable objective assumes independent bets.
