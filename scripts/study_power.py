"""Study 2: does the Liao et al. (2016) sample-size formula deliver its nominal power?

Design (HeartSteps-like layout described in Klasnja et al. 2015: 42 days x 5
decision points/day, randomization probability 0.4; availability 0.7 is our choice).
Standardized average effect 0.1, two trends: constant and quadratic
(initial 0, maximum on day 28). Analysis: WCLS with g_t = f_t = Z_t,
ptilde = rho, Mancl-DeRouen sandwich, F test of H0: beta = 0 at alpha = 0.05.

Scenarios for the error process:
  iid           working assumptions hold (Var = 1, independent errors)
  ar1_ri        AR(1) errors (phi = 0.5) + random intercept SD 0.5 (total variance still 1)
  het_effect    iid errors + person-level effect deviation, SD 0.1
Type-I error is checked at the formula's N with beta = 0.
"""

from __future__ import annotations

import time

import numpy as np

from _util import reps, save
from jitailab import EffectTrend, constant_trend, mrt_power, mrt_sample_size, quadratic_trend
from jitailab.power import simulate_power_trial
from jitailab.wcls import wcls_fit

DAYS, PER_DAY, TAU, RHO, AVG = 42, 5, 0.7, 0.4, 0.1
TRENDS = {
    "constant": constant_trend(DAYS, PER_DAY, AVG),
    "quadratic": quadratic_trend(DAYS, PER_DAY, AVG, day_max=28, initial=0.0),
}
GRIDS = {"constant": [10, 15, 20, 25, 30, 40], "quadratic": [12, 18, 24, 32, 40, 50]}
SCENARIOS = {
    "iid": dict(),
    "ar1_ri": dict(ar_phi=0.5, sigma_u=0.5),
    "het_effect": dict(effect_sd=0.1),
}
R = reps(1000)


def rejection_rate(N, trend, scen, seed):
    rng = np.random.default_rng(seed)
    # control trend in Z-coordinates; scaled day so it is O(1) over the study
    alpha = np.r_[0.5, 0.01, -1e-4][: trend.Z.shape[1]]
    rej = 0
    for _ in range(R):
        y, A, p, Z, cl = simulate_power_trial(N, trend, TAU, RHO, rng, control_coef=alpha, **scen)
        fit = wcls_fit(y, A, p, Z, Z, RHO, cl, small_sample=True)
        rej += fit.f_test()[1] < 0.05
    rate = rej / R
    return rate, float(np.sqrt(rate * (1 - rate) / R))


def main():
    out = {"settings": dict(days=DAYS, per_day=PER_DAY, tau=TAU, rho=RHO, avg=AVG, R=R), "trends": {}}
    for ti, (tname, trend) in enumerate(TRENDS.items()):
        n_req = mrt_sample_size(trend, TAU, RHO, power=0.8)
        tr_out = {"beta": trend.beta.tolist(), "N_for_80pct": n_req, "scenarios": {}}
        null = EffectTrend(Z=trend.Z, beta=np.zeros_like(trend.beta), day=trend.day)
        for si, (sname, scen) in enumerate(SCENARIOS.items()):
            rows = []
            t0 = time.time()
            for N in GRIDS[tname]:
                emp, mcse = rejection_rate(N, trend, scen, seed=[ti, si, N])
                rows.append({"N": N, "nominal": mrt_power(N, trend, TAU, RHO), "empirical": emp, "mcse": mcse})
            size, size_mcse = rejection_rate(n_req, null, scen, seed=[ti, si, 999])
            tr_out["scenarios"][sname] = {"power": rows, "type1_at_N": {"N": n_req, "rate": size, "mcse": size_mcse}}
            print(f"{tname:9s} {sname:10s} {time.time() - t0:5.1f}s  "
                  + " ".join(f"N={r['N']}:{r['nominal']:.2f}/{r['empirical']:.2f}" for r in rows)
                  + f"  size={size:.3f}")
        out["trends"][tname] = tr_out
    save("power", out)


if __name__ == "__main__":
    main()
