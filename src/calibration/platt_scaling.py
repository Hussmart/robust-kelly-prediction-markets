"""Platt scaling (logistic recalibration on the logit of the market price).

Model: ``P(Y=1 | p) = sigmoid(a * logit(p) + b)`` with two parameters. ``a = 1, b = 0``
is the identity (a perfectly calibrated price). ``a < 1`` means prices are too extreme
(the favourite-longshot pattern), ``a > 1`` too timid, and ``b`` shifts every
probability.

Fit by maximum likelihood on the log-loss

    L(a, b) = - sum_i [ y_i log q_i + (1 - y_i) log(1 - q_i) ],  q_i = sigmoid(a x_i + b),
    x_i = logit(p_i),

which is convex. Its gradient is ``(sum (q_i - y_i) x_i, sum (q_i - y_i))`` and its
Hessian is ``X^T diag(q(1-q)) X``, so Newton's method (IRLS) converges in a few steps.
A tiny ridge on ``(a - 1, b)`` keeps the fit finite when the data are perfectly
separable, which is common with a small number of resolved markets.

Uncertainty (Laplace approximation). The ridge term is a Gaussian prior
``theta ~ N(theta_0, I / ridge)``, so the penalised objective is the negative log-posterior.
Around its minimum ``theta_hat`` the posterior is approximately ``N(theta_hat, H^{-1})``
with ``H`` the penalised Hessian above. For a price ``p`` with ``x = logit(p)`` and
``u = (x, 1)``, the log-odds ``z = u^T theta`` has posterior variance ``u^T H^{-1} u``, so a
``(1 - alpha)`` interval for the calibrated probability is

    sigma( z_hat -/+ z_{1 - alpha/2} * sqrt(u^T H^{-1} u) ).

Unlike a bootstrap, this interval does not collapse when the sample contains no failures
in some region (all favourites won): the prior keeps the posterior of the slope wide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 0.005


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # numerically stable


@dataclass
class PlattCalibrator:
    """Two-parameter logistic calibrator ``q = sigmoid(a * logit(p) + b)``.

    Attributes:
        ridge: L2 penalty strength pulling ``(a, b)`` toward the identity ``(1, 0)``.
        max_iter / tol: Newton iteration limits.
    """

    ridge: float = 1e-3
    max_iter: int = 50
    tol: float = 1e-10
    a_: float = 1.0
    b_: float = 0.0
    cov_: np.ndarray | None = None

    def fit(self, p: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "PlattCalibrator":
        """Fit ``(a, b)`` by Newton's method on the ridge-penalised log-loss."""
        x, y = _logit(p), np.asarray(y, dtype=float)
        w = np.ones_like(x) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if len(np.unique(y)) < 2:
            raise ValueError("Platt scaling needs both outcome classes in the training data")
        X = np.column_stack([x, np.ones_like(x)])
        theta = np.array([1.0, 0.0])
        prior = np.array([1.0, 0.0])
        for _ in range(self.max_iter):
            q = _sigmoid(X @ theta)
            grad = X.T @ (w * (q - y)) + self.ridge * (theta - prior)
            hess = X.T @ (X * (w * q * (1 - q))[:, None]) + self.ridge * np.eye(2)
            step = np.linalg.solve(hess, grad)
            theta = theta - step
            if np.max(np.abs(step)) < self.tol:
                break
        self.a_, self.b_ = float(theta[0]), float(theta[1])
        q = _sigmoid(X @ theta)
        hess = X.T @ (X * (w * q * (1 - q))[:, None]) + self.ridge * np.eye(2)
        self.cov_ = np.linalg.inv(hess)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        """Calibrated probabilities for raw prices ``p``."""
        return _sigmoid(self.a_ * _logit(p) + self.b_)

    def predict_interval(self, p: np.ndarray, alpha: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
        """Laplace-approximation ``(lower, upper)`` interval of the calibrated probability.

        Requires a fitted model. ``alpha = 0.10`` gives a 90% interval.
        """
        if self.cov_ is None:
            raise RuntimeError("fit the calibrator before requesting an interval")
        from scipy.stats import norm

        x = _logit(p)
        u = np.column_stack([x, np.ones_like(x)])
        sd = np.sqrt(np.einsum("ij,jk,ik->i", u, self.cov_, u))
        z, k = self.a_ * x + self.b_, norm.ppf(1 - alpha / 2)
        return _sigmoid(z - k * sd), _sigmoid(z + k * sd)
