"""Estimators of the causal excursion effect.

* :func:`wcls` -- weighted and centered least squares (Boruvka et al. 2018).
* :func:`naive_gee` -- the "obvious" regression of Y on (controls, A x moderators),
  i.e. GEE with a working-independence correlation and *no* weights / centering.

Both return a :class:`Fit` with a cluster-robust sandwich covariance, optionally
with the Mancl & DeRouen (2001) small-sample bias correction, and t-based
inference with N - p - q degrees of freedom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .simulate import MRTData


@dataclass
class Fit:
    coef: np.ndarray  # (q + p,) control coefficients alpha then effect coefficients beta
    cov: np.ndarray  # (q + p, q + p) sandwich covariance
    q: int  # number of control parameters
    p: int  # number of effect (moderator) parameters
    n_clusters: int
    small_sample: bool

    @property
    def beta(self) -> np.ndarray:
        return self.coef[self.q :]

    @property
    def beta_cov(self) -> np.ndarray:
        return self.cov[self.q :, self.q :]

    @property
    def beta_se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.beta_cov))

    @property
    def df(self) -> int:
        return self.n_clusters - self.p - self.q

    def beta_ci(self, level: float = 0.95) -> np.ndarray:
        """(p, 2) confidence intervals. t_{N-p-q} if small_sample else normal."""
        z = (
            stats.t.ppf(0.5 + level / 2, self.df)
            if self.small_sample
            else stats.norm.ppf(0.5 + level / 2)
        )
        se = self.beta_se
        return np.column_stack([self.beta - z * se, self.beta + z * se])

    def linear_combination(self, L: np.ndarray, level: float = 0.95):
        """Estimate, SE and CI of L @ beta for each row of L (e.g. beta(t) on a grid)."""
        L = np.atleast_2d(L)
        est = L @ self.beta
        se = np.sqrt(np.einsum("ij,jk,ik->i", L, self.beta_cov, L))
        z = stats.t.ppf(0.5 + level / 2, self.df) if self.small_sample else stats.norm.ppf(0.5 + level / 2)
        return est, se, np.column_stack([est - z * se, est + z * se])

    def f_test(self) -> tuple[float, float]:
        """Test H0: beta = 0 (all p components), as in Liao et al. (2016).

        Uses the Hotelling-type rescaling
        F = (N - q - p) / (p (N - q - 1)) * beta' V^{-1} beta  ~  F(p, N - q - p).
        Returns (F statistic, p-value).
        """
        b = self.beta
        T2 = float(b @ np.linalg.solve(self.beta_cov, b))
        N, p, q = self.n_clusters, self.p, self.q
        F = (N - q - p) / (p * (N - q - 1)) * T2
        return F, float(stats.f.sf(F, p, N - q - p))


def sandwich_wls(X, y, w, cluster, small_sample=True):
    """Weighted least squares with a cluster-robust sandwich covariance.

    Mancl-DeRouen correction replaces each cluster's residual vector e_i by
    (I - H_ii)^{-1} e_i with H_ii = X_i B^{-1} X_i' W_i and B = sum_i X_i' W_i X_i.
    By the Woodbury identity (I - H_ii)^{-1} = I + X_i (B - M_i)^{-1} X_i' W_i
    with M_i = X_i' W_i X_i, so the corrected score is
    U_i* = g_i + M_i (B - M_i)^{-1} g_i,  g_i = X_i' W_i e_i  -- only d x d solves.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    cluster = np.asarray(cluster)
    order = np.argsort(cluster, kind="stable")
    X, y, w, cluster = X[order], y[order], w[order], cluster[order]
    starts = np.flatnonzero(np.r_[True, cluster[1:] != cluster[:-1]])

    Xw = X * w[:, None]
    B = X.T @ Xw
    coef = np.linalg.solve(B, Xw.T @ y)
    resid = y - X @ coef

    G = np.add.reduceat(Xw * resid[:, None], starts, axis=0)  # (n, d) scores g_i
    if small_sample:
        outer = np.einsum("ri,rj->rij", Xw, X)  # per-row x w x'
        M = np.add.reduceat(outer, starts, axis=0)  # (n, d, d)
        adj = np.linalg.solve(B[None] - M, G[..., None])[..., 0]
        G = G + np.einsum("nij,nj->ni", M, adj)
    meat = G.T @ G
    Binv = np.linalg.inv(B)
    cov = Binv @ meat @ Binv
    return coef, cov, len(starts)


def wcls_fit(y, A, p, G, F, ptilde, cluster, small_sample=True) -> Fit:
    """Low-level WCLS on flat arrays restricted to *available* decision points.

    ``G`` is the (n_rows, q) control design g_t, ``F`` the (n_rows, p) moderator
    design f_t, ``p`` the randomization probability actually used, ``ptilde``
    the numerator probability (scalar or per row; must depend on f_t only).
    """
    A = np.asarray(A, float)
    p = np.asarray(p, float)
    pt = np.broadcast_to(np.asarray(ptilde, float), A.shape)
    G = np.atleast_2d(np.asarray(G, float).T).T
    F = np.atleast_2d(np.asarray(F, float).T).T
    w = np.where(A == 1, pt / p, (1 - pt) / (1 - p))
    X = np.column_stack([G, (A - pt)[:, None] * F])
    coef, cov, n = sandwich_wls(X, y, w, cluster, small_sample)
    return Fit(coef, cov, q=G.shape[1], p=F.shape[1], n_clusters=n, small_sample=small_sample)


def wcls(
    data: MRTData,
    control=("1", "S", "time"),
    moderators=("1",),
    ptilde=None,
    small_sample: bool = True,
) -> Fit:
    """Weighted and centered least squares (Boruvka et al. 2018, JASA).

    Fits, on available decision points only,

        Y = g' alpha + (A - ptilde) f' beta + error,
        weights W = (ptilde/p)^A ((1-ptilde)/(1-p))^(1-A).

    ``ptilde`` must be a function of the moderators f only; pass a scalar
    (constant), an (N, T) array, or ``None`` (defaults to the pooled share of
    treated available decision points, treated as known -- see README limitations).
    """
    keep = data.I.ravel() == 1
    A = data.A.ravel()[keep].astype(float)
    if ptilde is None:
        pt = np.full_like(A, A.mean())
    else:
        pt = np.broadcast_to(np.asarray(ptilde, float), data.Y.shape).ravel()[keep]
    return wcls_fit(
        data.Y.ravel()[keep],
        A,
        data.p.ravel()[keep],
        data.design(control)[keep],
        data.design(moderators)[keep],
        pt,
        data.cluster.ravel()[keep],
        small_sample,
    )


def naive_gee(
    data: MRTData,
    control=("1", "S", "time"),
    moderators=("1",),
    small_sample: bool = True,
) -> Fit:
    """Unweighted, uncentered regression Y ~ g + A*f on available decision points.

    Equivalent to GEE with working independence. Consistent for the causal
    excursion effect only when p_t does not depend on variables that are
    left out of f but modify the effect (and the control model is adequate).
    """
    keep = data.I.ravel() == 1
    A = data.A.ravel()[keep].astype(float)
    y = data.Y.ravel()[keep]
    G = data.design(control)[keep]
    F = data.design(moderators)[keep]
    X = np.column_stack([G, A[:, None] * F])
    coef, cov, n = sandwich_wls(X, y, np.ones_like(y), data.cluster.ravel()[keep], small_sample)
    return Fit(coef, cov, q=G.shape[1], p=F.shape[1], n_clusters=n, small_sample=small_sample)
