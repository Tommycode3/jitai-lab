"""Ground truth for simulation studies.

For a moderator vector f_t and a reference probability ptilde_t that depends on
f_t only, the WCLS estimand is (Boruvka et al. 2018, Sec. 3)

    beta* = argmin_b  E[ sum_t I_t ptilde_t (1 - ptilde_t) (beta_t - f_t' b)^2 ],

where beta_t is the individual proximal effect. Because the weights depend on
f_t only, replacing beta_t by its conditional mean (the causal excursion effect
beta(t; S_t)) gives the same minimiser. We approximate the expectation with one
large simulated population; the outcome noise does not enter beta_t, so the
Monte Carlo error is small and is reported by :func:`true_beta`.
"""

from __future__ import annotations

import numpy as np

from .simulate import MRTScenario, simulate_mrt


def true_beta(
    scenario: MRTScenario,
    moderators=("1",),
    ptilde=0.5,
    n_pop: int = 4000,
    seed: int = 12345,
    n_batches: int = 4,
):
    """Return (beta*, Monte Carlo SE) via batched projection of individual effects."""
    ests = []
    for b in range(n_batches):
        d = simulate_mrt(n_pop // n_batches, scenario, seed=seed + b)
        keep = d.I.ravel() == 1
        F = d.design(moderators)[keep]
        pt = np.broadcast_to(np.asarray(ptilde, float), d.Y.shape).ravel()[keep]
        w = pt * (1 - pt)
        y = d.true_effect.ravel()[keep]
        Fw = F * w[:, None]
        ests.append(np.linalg.solve(F.T @ Fw, Fw.T @ y))
    ests = np.array(ests)
    return ests.mean(0), ests.std(0, ddof=1) / np.sqrt(n_batches)
