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

