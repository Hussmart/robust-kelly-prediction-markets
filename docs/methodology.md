# Methodology

This document gives the mathematical formulation behind every stage of the pipeline.
The notation here is the notation used in the code: every symbol is tied to the
function that implements it.

---

## 1. Setting and notation

A **binary contract** $i$ pays $1$ if event $E_i$ happens and $0$ otherwise. Let
$Y_i = \mathbb 1\{E_i\} \in \{0,1\}$ be the realised outcome. On venue
$v \in \{\text{poly}, \text{kalshi}\}$ the contract trades at price $p^v_{i,t} \in (0,1)$
at time $t$.

**Equivalence.** Two contracts $i$ (Polymarket) and $j$ (Kalshi) are *equivalent*,
written $i \sim j$, iff their payoff indicators agree in every state of the world:
$\mathbb 1\{E_i\}(\omega) = \mathbb 1\{E_j\}(\omega)\ \forall \omega$.
For FOMC decisions, the state is the change $\Delta \in 25\,\text{bp}\cdot\mathbb Z$ in
the target range. A Polymarket bucket $B^P \subset 25\mathbb Z$ and a Kalshi bucket
$B^K$ are equivalent iff $B^P = B^K$. Buckets where one side is a *union* of the
other's buckets (Polymarket "increase $25^+$" $= \{25\} \cup \{50, 75, \dots\}$, i.e.
Kalshi H25 $\cup$ H26) are excluded. They would need a one-to-many edge in the
equivalence graph and a sum constraint on prices, which is left to future work.

The resulting mapping (`data/mappings/fomc_pairs.csv`, 63 pairs over 20 meetings,
May-2024 to Sep-2026) is **hand-curated at the meeting level and rule-based at the
bucket level**, and it is validated by requiring $Y_i = Y_j$ for every pair
(`scripts/build_fomc_mapping.py`).

**Event time.** For each pair, $\tau$ is Kalshi's trading close, a few minutes before
the 2 pm ET announcement. All features at snapshot $t$ use only data with timestamps
$\le t < \tau$.

---

## 2. Features (`src/features/feature_engineering.py`)

Snapshots lie on a grid $t_k = \tau - k\,\delta$ with $\delta = 6$ h over the 21 days
before $\tau$ (the first 72 h are reserved as a warm-up for volatility estimates).

**Implied probabilities.** Kalshi reports top-of-book quotes $(b_t, a_t)$. When the
quote is valid ($0 < b_t < a_t < 1$, $a_t - b_t \le 0.5$) we use the mid
$p^{K}_t = (a_t + b_t)/2$, and otherwise the last trade. Polymarket's public API
exposes a price series but no historical book, so $p^P_t$ is that series. Both are
carried forward as-of for at most 48 h. Snapshots where either price is older are
dropped, because they carry no information.

**Divergence.**
$$
d_t = p^P_t - p^K_t, \qquad
\ell_t = \operatorname{logit}(p^P_t) - \operatorname{logit}(p^K_t), \qquad
\operatorname{logit}(p) = \log\frac{\bar p}{1-\bar p},\ \bar p = \operatorname{clip}(p, \varepsilon, 1-\varepsilon)
$$
with $\varepsilon = 0.005$. The logit divergence $\ell_t$ weights a 2c gap at
$p \approx 0.02$ far more heavily than the same gap at $p \approx 0.5$. This matches
the information content of the gap.

**Spread.** Kalshi's quoted spread is $s^K_t = a_t - b_t$. For Polymarket (and, as a
check, for Kalshi) we use the **Roll (1984) estimator**. Suppose trade prices are
$P_n = m_n + \tfrac{s}{2} q_n$ with efficient price $m_n$ (a martingale), a constant
spread $s$, and i.i.d. trade signs $q_n = \pm 1$ with probability $\tfrac12$ each,
independent of $m$. Then
$$
\operatorname{Cov}(\Delta P_n, \Delta P_{n-1}) = \tfrac{s^2}{4}\operatorname{Cov}(q_n - q_{n-1},\, q_{n-1} - q_{n-2}) = -\tfrac{s^2}{4},
\quad\Longrightarrow\quad
\hat s = 2\sqrt{-\widehat{\operatorname{Cov}}(\Delta P_n, \Delta P_{n-1})}
$$
(set to $0$ if the sample covariance is non-negative). It is computed on the trades of
the trailing 24 h. `tests/test_features.py::test_roll_spread_recovers_known_spread`
checks that $\hat s$ recovers $s$ on simulated bid-ask bounce.

**Volume and order-flow imbalance.** With taker notional $n_m$ and signed YES-flow
$f_m = \pm n_m$ (positive when the taker added YES exposure) for trades in the trailing
window $W_t = (t - 24\text{h}, t]$:
$$
V_t = \sum_{m \in W_t} n_m, \qquad
\mathrm{OFI}_t = \begin{cases} \dfrac{\sum_{m\in W_t} f_m}{V_t} & V_t > 0\\[4pt] 0 & V_t = 0\end{cases}
\ \in [-1, 1].
$$
A window without trades is treated as *no pressure* ($\mathrm{OFI} = 0$, $V = 0$),
not as missing data.

**Volatility.** $\sigma_t$ is the standard deviation of hourly changes of
$\operatorname{logit}(p_t)$ over the trailing 72 h.

**Other.** Hours to close $\tau - t$, hours since the last trade on each venue, and
$\log(1 + \min(V^P_t, V^K_t))$, the depth of the *thinner* venue.

**Target for the real-vs-noise screen (Stage 6).** For $|d_t| \ge 0.02$:
$$
\text{persistent}_t = \mathbb 1\big\{\operatorname{sign}(d_{t+24h}) = \operatorname{sign}(d_t)\ \wedge\ |d_{t+24h}| \ge \tfrac12 |d_t|\big\}.
$$
It is forward-looking by construction, so it is only ever used as a label and never
as a feature.

---

## 3. Anomaly detection (`src/anomaly/`)

### 3.1 Isolation Forest

Feature vector $x_t$: $|\ell_t|$, $|d_t|$, Kalshi spread, Roll spread (Polymarket),
log-volumes, both OFIs and their gap $|\mathrm{OFI}^P - \mathrm{OFI}^K|$, both
volatilities, $\log(1+\text{hours to close})$ and log-staleness. The level of the price
itself is excluded on purpose. Isolation Forest (Liu, Ting & Zhou, 2008) grows random
trees that split on a random feature at a random threshold. The score
$$
s(x, n) = 2^{-\,\mathbb E[h(x)]/c(n)}, \qquad c(n) = 2H(n-1) - \tfrac{2(n-1)}{n}
$$
uses the mean path length $h$ to isolate $x$ and the mean unsuccessful-search path
length $c(n)$ of a binary search tree as normaliser ($H$ is the harmonic number).
Scores near 1 mean "easy to isolate". Median imputation and robust scaling are fitted on
the training snapshots only.

### 3.2 Graph consensus rule

Let $G = (P \cup K, E)$ be the bipartite graph whose edges are the equivalent pairs
$i \sim j$ of Section 1. Every contract has exactly one twin, so $G$ is a perfect
matching. `validate_graph` asserts this, and an ambiguous mapping would raise. Each edge
carries the divergence series $\ell_t$. Two independent detectors are attached to it:

$$
S^{\text{price}}_t = \mathbb 1\Big\{ \frac{|\ell_t| - m}{1.4826\,\mathrm{MAD}} \ge z_{\min} \Big\},
\qquad
S^{\text{liq}}_t = \mathbb 1\big\{ V^P_t \ge \underline V^P,\ V^K_t \ge \underline V^K,\ \text{stale}^{P,K}_t \le 12\text{h},\ s^K_t \le \bar s \big\}
$$

where $m$ and MAD are the median and median absolute deviation of $|\ell|$ on the
training snapshots, $\underline V$ are training-volume quantiles (25%), and $\bar s$ the
75% quantile of the quoted spread. A snapshot is flagged iff
$S^{\text{price}}_t \wedge S^{\text{liq}}_t$.

*Rationale.* A large gap between two thin or stale books is what a false positive looks
like, because no one could trade it. The same gap between two active, tightly quoted books is
a candidate real mispricing. Requiring both signals trades recall for precision. This is
the "joint anomaly instead of a single signal" principle, applied to prices and liquidity.

---

## 4. Calibration (`src/calibration/`)

Raw prices $p$ are not necessarily probabilities. We seek a map $f$ with
$\mathbb P(Y = 1 \mid p) = f(p)$.

**Platt scaling.** $f(p) = \sigma(a\,\operatorname{logit}(p) + b)$ with
$\sigma(z) = 1/(1+e^{-z})$. $(a,b) = (1,0)$ is perfect calibration, $a < 1$ means the market is
over-confident, $a > 1$ under-confident. The penalised negative log-likelihood
$$
\mathcal L(a,b) = -\sum_i \big[y_i \log q_i + (1-y_i)\log(1-q_i)\big] + \tfrac{\rho}{2}\big[(a-1)^2 + b^2\big],\quad q_i = \sigma(a x_i + b),\ x_i=\operatorname{logit}(p_i)
$$
is strictly convex with gradient $X^\top(q - y) + \rho(\theta - \theta_0)$ and Hessian
$X^\top \mathrm{diag}(q(1-q))X + \rho I$ ($X = [x, \mathbf 1]$, $\theta_0 = (1,0)$). We solve it by Newton's
method. The ridge $\rho = 10^{-3}$ toward the identity keeps $(a,b)$ finite when the sample is
perfectly separable, which occurs with few resolved markets. The tests check parameter
recovery on simulated data and agreement with scikit-learn's logistic regression.

**Isotonic regression.** $f$ is the non-decreasing least-squares fit
$\min_f \sum_i w_i (y_i - f(p_i))^2$ s.t. $f$ non-decreasing, solved exactly by
pool-adjacent-violators (Barlow et al., 1972): scan left to right and whenever a block's
mean falls below its predecessor's, merge the two into their weighted mean. Our
implementation is checked against `sklearn.isotonic.IsotonicRegression` to $10^{-10}$.

**Metrics.** Brier score $\frac1n\sum(q_i - y_i)^2$; expected calibration error with $B$
equal-mass bins,
$\mathrm{ECE} = \sum_{b=1}^{B}\frac{n_b}{n}\,|\bar y_b - \bar q_b|$; and log-loss.

**Uncertainty of a calibrated probability.** Snapshots of one market (or one meeting, as
its buckets are mutually exclusive) are strongly dependent, so resampling snapshots
understates variance. We use a **cluster bootstrap**: resample whole meetings with
replacement, refit $f$, and take the percentile interval of $f^{(b)}(p)$.
$\hat p = f(p)$ is the point estimate and $d = \hat p - q_{0.05}$ is the
downward half-width fed to the robust optimiser.

---

## 5. Robust Kelly allocation (`src/optimization/`)

### 5.1 Model

Bet $i \in \{1..N\}$ costs $c_i$ and pays 1 on a win, so the net odds are
$b_i = (1-c_i)/c_i$. Staking a fraction $f_i$ of wealth returns $1 + f_i b_i$ on a win and
$1 - f_i$ on a loss. For bets that are independent and settled with reinvestment, expected
log-growth is separable:
$$
g(f;p) = \sum_{i=1}^N \big[\,p_i \log(1 + f_i b_i) + (1-p_i)\log(1 - f_i)\,\big].
$$
Naive Kelly maximises $g(f;\hat p)$ subject to $\sum_i f_i \le F$ and $0 \le f_i \le f_{\max}$.
This problem is convex, and its KKT conditions reduce to a scalar root search
(`naive_kelly.py`). With a slack budget, $f_i = \hat p_i - (1-\hat p_i)/b_i$, the classical Kelly fraction.

### 5.2 Budgeted uncertainty (Bertsimas & Sim, 2004)

The estimate $\hat p_i$ is uncertain. Let $d_i \ge 0$ be its maximal *downward* error.
Only downward deviations hurt because $\partial g/\partial p_i = D_i(f_i) \ge 0$. At most $\Gamma$
estimates are wrong at once:
$$
\mathcal U(\Gamma) = \Big\{ p : p_i = \hat p_i - d_i z_i,\ 0 \le z_i \le 1,\ \textstyle\sum_i z_i \le \Gamma \Big\}.
$$
$\Gamma = 0$ trusts every estimate, $\Gamma = N$ assumes they are all at their worst, and
intermediate $\Gamma$ is the "price of robustness" dial. The robust problem is
$$
\max_{f}\ \min_{p \in \mathcal U(\Gamma)} g(f;p).
$$

### 5.3 Robust counterpart via LP duality

For fixed $f$, write $B_i(f_i) = \log(1-f_i)$ and $D_i(f_i) = \log(1+f_ib_i) - \log(1-f_i) \ge 0$. Then
$g(f;p) = \sum_i B_i + \sum_i p_i D_i$ is **linear in $p$**, and the inner minimisation is
$$
\min_{p\in\mathcal U} g = \sum_i\big[B_i + \hat p_i D_i\big] - \underbrace{\max_{z}\Big\{\sum_i d_i D_i\, z_i \ :\ \sum_i z_i \le \Gamma,\ 0\le z_i\le 1\Big\}}_{\text{LP in } z}.
$$
Assign $\lambda \ge 0$ to the row $\sum z_i \le \Gamma$ and $\nu_i \ge 0$ to $z_i \le 1$. The LP dual is
$$
\min_{\lambda,\nu \ge 0}\ \Gamma\lambda + \sum_i \nu_i \quad\text{s.t.}\quad \lambda + \nu_i \ge d_i D_i(f_i)\ \ \forall i.
$$
By strong duality the max-min collapses to a single maximisation:
$$
\boxed{\ \max_{f,\lambda,\nu}\ \sum_i\big[B_i(f_i) + \hat p_i D_i(f_i)\big] - \Gamma\lambda - \sum_i\nu_i
\ \ \text{s.t.}\ \ \lambda + \nu_i \ge d_i D_i(f_i),\ \ \sum_i f_i \le F,\ \ 0 \le f_i \le f_{\max},\ \lambda,\nu\ge0\ }
$$

**Why it is a MILP and not a convex program.** $D_i$ is the sum of a concave term
($\log(1+f b_i)$) and a convex term ($-\log(1-f)$), so the constraint
$\lambda + \nu_i \ge d_iD_i(f_i)$ is non-convex. We use a piecewise-linear interpolant on the
grid $0 = \phi_0 < \dots < \phi_K = f_{\max}$ (quadratic spacing, dense near 0) and the
incremental formulation
$$
f_i = \sum_{k=1}^K \delta_{ik}(\phi_k - \phi_{k-1}),\qquad 0\le\delta_{ik}\le1,\qquad
\delta_{i,k+1} \le y_{ik} \le \delta_{ik},\ \ y_{ik}\in\{0,1\},
$$
so that segments fill in order. Without the binaries the relaxation could fill segments
in the order that minimises $D_i$ and understate the worst-case loss. All of $B_i$, $D_i$
and the objective are then linear in $\delta$. The result is a MILP solved with Pyomo + HiGHS
(`robust_kelly.py`).

### 5.4 What the tests verify (`tests/test_robust_kelly.py`)

| Property | Test |
|---|---|
| $\Gamma = 0$ reduces to naive Kelly | stakes within one grid step, growth equal to $10^{-4}$ |
| $\Gamma \ge N$ reduces to Kelly at $\hat p - d$ | same |
| The duality is right | for $\Gamma \in \{1,2,3\}$ the MILP optimum equals an **independently computed** max-min solution (enumerate the vertices of $\mathcal U$, solve the epigraph problem with SLSQP) |
| The greedy worst-case evaluator is exact | equals vertex enumeration to $10^{-12}$, and is linear between integer $\Gamma$ |
| Optimal worst-case value is non-increasing in $\Gamma$ | monotone in $\Gamma = 0..4$ |
| Robust stakes are never worse in the worst case than naive stakes | for $\Gamma \in \{1,2,4\}$ |

### 5.5 Modelling caveats (stated in the README as well)

* The separable objective treats the bets of a round as independent. Buckets of one FOMC
  meeting are mutually exclusive. The **backtest settles bets with the true payoffs**, so the
  reported P&L does not depend on this assumption. Only the optimiser's *model* does.
* The budget $F < 1$ and cap $f_{\max}$ keep wealth strictly positive even if every
  mutually exclusive bet of a round loses.
* Fees, spread and slippage are lumped into a constant cost $\kappa$ added to the price.

---

## 6. Backtest (`src/backtest/`)

*Rounds and walk-forward calibration.* One round is one FOMC meeting, decided 24 h before
the announcement. For meeting $m$, the calibrator (and the consensus thresholds) are fitted
on snapshots of meetings that had **resolved before that decision time**. The first 6
meetings are burn-in. `test_future_outcomes_do_not_change_past_decisions` checks that flipping
the last meeting's outcomes changes no earlier round.

*Candidates.* The pooled price $\bar p = (p^P + p^K)/2$ is calibrated to $\hat p$. For each
pair we compare buying YES on the cheaper venue at $c^{\text{YES}} = \min(p^P,p^K) + \kappa$ with buying NO on the dearer
venue at $c^{\text{NO}} = 1 - \max(p^P,p^K) + \kappa$, and take the larger calibrated edge if it is
positive. Optionally, only pairs flagged by the consensus rule are kept.

*Strategies.* naive Kelly · robust Kelly for several $\Gamma$ · equal weight (the budget split
evenly) · random (mean over random subsets and stakes).

*Metrics.* With per-round log-returns $r_t = \log W_t/W_{t-1}$: cumulative log-growth
$\sum r_t$, max drawdown $\max_t(1 - W_t/\max_{s\le t}W_s)$, and the Sharpe-like ratio
$\bar r/s_r\sqrt{8}$ (8 FOMC meetings a year, no risk-free rate).
