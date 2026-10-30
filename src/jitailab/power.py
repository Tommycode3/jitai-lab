"""Sample size for micro-randomized trials (Liao, Klasnja, Tewari & Murphy 2016).

Working assumptions of the formula (Liao et al. 2016, Sec. 3): the proximal
effect is beta(t) = Z_t' beta in units of the outcome SD sigma (a *standardized*
effect); randomization probability rho_t and expected availability tau_t are
known; the analysis is the centered and weighted least-squares fit with a
control model of dimension q and the test of H0: beta = 0 rejects when

    (N - q - p) / (p (N - q - 1)) * N beta_hat' Sigma_hat^{-1} beta_hat  >  F^{-1}_{p, N-q-p}(1 - alpha).

Under the alternative this statistic is approximately non-central
F(p, N - q - p; c_N) with

    c_N = N * beta' [ sum_t tau_t rho_t (1 - rho_t) Z_t Z_t' ] beta.

The trend helpers below parametrize beta(t) by day; ``quadratic_trend`` follows
the shape used in Liao et al. (initial effect, day of maximal effect, average
effect) but the code and parametrization are ours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class EffectTrend:
    Z: np.ndarray  # (T, p) design for beta(t)
    beta: np.ndarray  # (p,) standardized coefficients
    day: np.ndarray  # (T,) day index 1..n_days of each decision point

    @property
    def effect(self) -> np.ndarray:
        """Standardized proximal effect at each decision point, shape (T,)."""
        return self.Z @ self.beta


def _days(n_days: int, per_day: int) -> np.ndarray:
    return np.repeat(np.arange(1, n_days + 1), per_day)


def constant_trend(n_days: int, per_day: int, avg: float) -> EffectTrend:
    d = _days(n_days, per_day)
    return EffectTrend(Z=np.ones((d.size, 1)), beta=np.array([avg]), day=d)


def linear_trend(n_days: int, per_day: int, avg: float, initial: float = 0.0) -> EffectTrend:
    """beta(d) = b1 + b2 (d - 1); beta(1) = initial; mean over days = avg."""
    d = _days(n_days, per_day)
    x = np.arange(n_days)  # d - 1 over days
    b2 = (avg - initial) / x.mean() if x.mean() > 0 else 0.0
    Z = np.column_stack([np.ones(d.size), d - 1.0])
    return EffectTrend(Z=Z, beta=np.array([initial, b2]), day=d)


def quadratic_trend(
    n_days: int, per_day: int, avg: float, day_max: float, initial: float = 0.0
) -> EffectTrend:
    """beta(d) = b1 + b2 (d-1) + b3 (d-1)^2 with beta(1) = initial,
    vertex at d = day_max, and mean over days 1..n_days equal to avg."""
    d = _days(n_days, per_day)
    x = np.arange(n_days, dtype=float)
    M = np.array(
        [
            [1.0, 0.0, 0.0],  # beta(1) = initial
            [0.0, 1.0, 2.0 * (day_max - 1)],  # derivative zero at day_max
            [1.0, x.mean(), (x**2).mean()],  # average over days
        ]
    )
    beta = np.linalg.solve(M, np.array([initial, 0.0, avg]))
    Z = np.column_stack([np.ones(d.size), d - 1.0, (d - 1.0) ** 2])
    return EffectTrend(Z=Z, beta=beta, day=d)


def noncentrality(N: int, trend: EffectTrend, avail, rand_prob) -> float:
    T = trend.Z.shape[0]
    tau = np.broadcast_to(np.asarray(avail, float), (T,))
    rho = np.broadcast_to(np.asarray(rand_prob, float), (T,))
    Sigma = (trend.Z * (tau * rho * (1 - rho))[:, None]).T @ trend.Z
    return float(N * trend.beta @ Sigma @ trend.beta)


def mrt_power(N: int, trend: EffectTrend, avail, rand_prob, q: int | None = None, alpha: float = 0.05) -> float:
    """Approximate power of the test of H0: beta = 0 with N participants."""
    p = trend.Z.shape[1]
    q = p if q is None else q
    df2 = N - q - p
    if df2 <= 0:
        return float("nan")
    crit = stats.f.ppf(1 - alpha, p, df2)
    return float(stats.ncf.sf(crit, p, df2, noncentrality(N, trend, avail, rand_prob)))


def mrt_sample_size(
    trend: EffectTrend,
    avail,
    rand_prob,
    power: float = 0.8,
    q: int | None = None,
    alpha: float = 0.05,
    n_max: int = 10_000,
) -> int:
    """Smallest N whose approximate power reaches ``power``."""
    p = trend.Z.shape[1]
    q = p if q is None else q
    for N in range(p + q + 2, n_max + 1):
        if mrt_power(N, trend, avail, rand_prob, q, alpha) >= power:
            return N
    raise ValueError("required N exceeds n_max")


def simulate_power_trial(
    N: int,
    trend: EffectTrend,
    avail,
    rand_prob,
    rng: np.random.Generator,
    ar_phi: float = 0.0,
    sigma_u: float = 0.0,
    effect_sd: float = 0.0,
    control_coef=None,
):
    """Simulate one trial for validating the sample-size formula.

    Y_{t+1} = Z_t' alpha + A_t (Z_t' beta + b_i) + u_i + e_t with u_i ~ N(0, sigma_u^2),
    e_t a stationary AR(1) (coefficient ``ar_phi``) with Var(e_t) = 1 - sigma_u^2, so
    the total residual variance stays 1 and beta stays a standardized effect;
    b_i ~ N(0, effect_sd^2) is a person-level deviation of the effect. With the defaults (all zero) the
    working assumptions of the formula hold exactly.

    Returns flat arrays (y, A, p, Z, cluster) restricted to available decision points.
    """
    Z = trend.Z
    T, p = Z.shape
    tau = np.broadcast_to(np.asarray(avail, float), (T,))
    rho = np.broadcast_to(np.asarray(rand_prob, float), (T,))
    alpha = np.zeros(p) if control_coef is None else np.asarray(control_coef, float)
    I = rng.random((N, T)) < tau
    A = (I & (rng.random((N, T)) < rho)).astype(float)
    if not 0 <= sigma_u < 1:
        raise ValueError("sigma_u must be in [0, 1) so that total variance is 1")
    sd_e = np.sqrt(1 - sigma_u**2)
    e = np.empty((N, T))
    e[:, 0] = sd_e * rng.normal(size=N)
    innov = sd_e * np.sqrt(1 - ar_phi**2)
    z = rng.normal(size=(N, T))
    for t in range(1, T):
        e[:, t] = ar_phi * e[:, t - 1] + innov * z[:, t]
    u = rng.normal(0, sigma_u, N)[:, None]
    b = rng.normal(0, effect_sd, N)[:, None]
    Y = (Z @ alpha)[None, :] + A * ((Z @ trend.beta)[None, :] + b) + u + e
    keep = I.ravel()
    Zf = np.broadcast_to(Z, (N, T, p)).reshape(N * T, p)[keep]
    cl = np.repeat(np.arange(N), T)[keep]
    return Y.ravel()[keep], A.ravel()[keep], np.broadcast_to(rho, (N, T)).ravel()[keep], Zf, cl
