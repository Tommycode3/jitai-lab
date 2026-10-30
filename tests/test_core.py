import numpy as np
import pytest

from jitailab import (
    FixedPolicy,
    MRTScenario,
    ThompsonPolicy,
    constant_trend,
    mrt_power,
    mrt_sample_size,
    naive_gee,
    noncentrality,
    quadratic_trend,
    sandwich_wls,
    simulate_mrt,
    true_beta,
    wcls,
)


# ---------------------------------------------------------------- simulator
def test_simulator_invariants():
    d = simulate_mrt(30, MRTScenario(n_decisions=40), seed=0)
    assert d.Y.shape == (30, 40)
    assert np.all(d.A[d.I == 0] == 0), "no treatment when unavailable"
    assert np.all((d.p > 0) & (d.p < 1))
    assert set(np.unique(d.S)) <= {0, 1}
    # p follows the fixed context-dependent policy (0.7 / 0.1)
    assert np.allclose(d.p[d.S == 0], 0.7) and np.allclose(d.p[d.S == 1], 0.1)


def test_outcome_decomposes_into_baseline_plus_effect():
    """Y_{t+1} = baseline(H_t) + A_t * beta_t + noise, and the noise path does not
    depend on the effect size: the stored beta_t is the only channel for A_t."""
    sc = MRTScenario(n_decisions=10, avail_prob=1.0, policy=FixedPolicy(0.5, 0.5))
    d = simulate_mrt(5, sc, seed=3)
    # analytic check: Y - (everything except A*beta) is A*beta
    base = (
        sc.a0 + sc.aS * d.S + sc.aTime * d.time + sc.kappa * d.feature("A_prev") - sc.zeta * d.dose
    )
    resid = d.Y - base - d.A * d.true_effect
    # what's left is u_i + e_t: identical for a given seed regardless of A
    d2 = simulate_mrt(5, sc.with_(b0=sc.b0 + 1.0), seed=3)
    base2 = (
        sc.a0 + sc.aS * d2.S + sc.aTime * d2.time + sc.kappa * d2.feature("A_prev") - sc.zeta * d2.dose
    )
    resid2 = d2.Y - base2 - d2.A * d2.true_effect
    assert np.allclose(resid, resid2)


def test_truth_matches_closed_form_without_dynamics():
    sc = MRTScenario(eta=0.0, bTime=0.0, policy=FixedPolicy(0.4, 0.4))
    b, se = true_beta(sc, ("1",), ptilde=0.4, n_pop=2000)
    expected = sc.b0 + sc.bS * sc.context_prob
    assert abs(b[0] - expected) < 0.01


# ---------------------------------------------------------------- sandwich
def _toy(seed=0, n=25, m=12, d=3):
    rng = np.random.default_rng(seed)
    cl = np.repeat(np.arange(n), m)
    X = np.column_stack([np.ones(n * m), rng.normal(size=(n * m, d - 1))])
    y = X @ np.arange(1, d + 1) + rng.normal(size=n * m) + np.repeat(rng.normal(size=n), m)
    w = rng.uniform(0.5, 2.0, n * m)
    return X, y, w, cl


def test_sandwich_matches_statsmodels_cluster_robust():
    sm = pytest.importorskip("statsmodels.api")
    X, y, w, cl = _toy()
    coef, cov, _ = sandwich_wls(X, y, w, cl, small_sample=False)
    res = sm.WLS(y, X, weights=w).fit(cov_type="cluster", cov_kwds={"groups": cl, "use_correction": False})
    assert np.allclose(coef, res.params)
    assert np.allclose(cov, res.cov_params(), rtol=1e-8)


def test_mancl_derouen_woodbury_equals_explicit_inverse():
    X, y, w, cl = _toy(seed=1)
    coef, cov, _ = sandwich_wls(X, y, w, cl, small_sample=True)
    B = X.T @ (X * w[:, None])
    Binv = np.linalg.inv(B)
    e = y - X @ coef
    meat = np.zeros_like(B)
    for c in np.unique(cl):
        k = cl == c
        Xi, Wi = X[k], np.diag(w[k])
        H = Xi @ Binv @ Xi.T @ Wi
        ei = np.linalg.solve(np.eye(k.sum()) - H, e[k])
        U = Xi.T @ Wi @ ei
        meat += np.outer(U, U)
    assert np.allclose(cov, Binv @ meat @ Binv)
    # the correction inflates the variance
    _, cov0, _ = sandwich_wls(X, y, w, cl, small_sample=False)
    assert np.all(np.diag(cov) > np.diag(cov0))


def test_wcls_reduces_to_centered_ols_when_ptilde_equals_p():
    sm = pytest.importorskip("statsmodels.api")
    d = simulate_mrt(20, MRTScenario(n_decisions=30, policy=FixedPolicy(0.3, 0.3)), seed=2)
    f = wcls(d, ptilde=0.3, small_sample=False)
    k = d.I.ravel() == 1
    X = np.column_stack([d.design(("1", "S", "time"))[k], d.A.ravel()[k] - 0.3])
    res = sm.OLS(d.Y.ravel()[k], X).fit()
    assert np.allclose(f.coef, res.params)


# ---------------------------------------------------------------- bias
def test_naive_biased_wcls_not_under_context_dependent_randomization():
    sc = MRTScenario(n_decisions=60)
    d = simulate_mrt(1500, sc, seed=7)
    truth, _ = true_beta(sc, ptilde=0.46, n_pop=4000, seed=99)
    fw, fn = wcls(d, ptilde=0.46), naive_gee(d)
    assert abs(fw.beta[0] - truth[0]) < 3 * fw.beta_se[0]
    assert (fn.beta[0] - truth[0]) > 5 * fn.beta_se[0]


# ---------------------------------------------------------------- power
def test_noncentrality_closed_form_constant_effect():
    tr = constant_trend(10, 5, 0.1)
    c = noncentrality(30, tr, avail=0.7, rand_prob=0.4)
    assert np.isclose(c, 30 * 0.1**2 * 50 * 0.7 * 0.24)


def test_power_monotone_and_sample_size_consistent():
    tr = quadratic_trend(42, 5, avg=0.1, day_max=28, initial=0.0)
    pw = [mrt_power(N, tr, 0.7, 0.4) for N in range(10, 80, 5)]
    assert np.all(np.diff(pw) > 0)
    N = mrt_sample_size(tr, 0.7, 0.4, power=0.8)
    assert mrt_power(N, tr, 0.7, 0.4) >= 0.8 > mrt_power(N - 1, tr, 0.7, 0.4)


def test_quadratic_trend_constraints():
    tr = quadratic_trend(42, 5, avg=0.1, day_max=28, initial=0.0)
    daily = tr.effect[::5]
    assert np.isclose(daily[0], 0.0)
    assert np.isclose(daily.mean(), 0.1)
    assert np.argmax(daily) + 1 == 28


# ---------------------------------------------------------------- adaptive
def test_thompson_probabilities_clipped_and_wcls_runs():
    sc = MRTScenario(n_decisions=50, policy=ThompsonPolicy(p_min=0.2, p_max=0.8))
    d = simulate_mrt(40, sc, seed=4)
    assert d.p.min() >= 0.2 - 1e-12 and d.p.max() <= 0.8 + 1e-12
    f = wcls(d, ptilde=0.5)
    assert np.isfinite(f.beta_se).all()
