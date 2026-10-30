"""Study 1: bias and 95% CI coverage of WCLS vs the naive regression.

Two designs share the same marginal treatment rate (~0.46 of available points):
  * constant      p_t = 0.46
  * context-dep.  p_t = 0.7 if S_t = 0, 0.1 if S_t = 1
The true effect is moderated by S (and by time and recent dose), so under the
context-dependent design the naive regression puts too much weight on S = 0.
Estimand: marginal excursion effect (f_t = 1), ptilde = 0.46, control g = (1, S, time).
"""

from __future__ import annotations

import time

import numpy as np

from _util import reps, save, summarize
from jitailab import FixedPolicy, MRTScenario, naive_gee, simulate_mrt, true_beta, wcls

PT = 0.46
DESIGNS = {
    "constant": FixedPolicy(PT, PT),
    "context_dependent": FixedPolicy(0.7, 0.1),
}
NS = [20, 40, 80]
R = reps(1000)


def main():
    out = {"settings": {"T": 150, "ptilde": PT, "Ns": NS, "R": R}, "designs": {}}
    for k_design, (name, pol) in enumerate(DESIGNS.items()):
        sc = MRTScenario(policy=pol)
        truth, truth_mcse = true_beta(sc, ("1",), ptilde=PT, n_pop=8000)
        res = {"truth": float(truth[0]), "truth_mcse": float(truth_mcse[0]), "by_N": {}}
        for N in NS:
            rec = {k: [] for k in ("wcls_md", "wcls_plain", "naive")}
            t0 = time.time()
            for r in range(R):
                d = simulate_mrt(N, sc, seed=[k_design, N, r])
                for key, fit in (
                    ("wcls_md", wcls(d, ptilde=PT, small_sample=True)),
                    ("wcls_plain", wcls(d, ptilde=PT, small_sample=False)),
                    ("naive", naive_gee(d, small_sample=True)),
                ):
                    lo, hi = fit.beta_ci()[0]
                    rec[key].append((fit.beta[0], fit.beta_se[0], lo, hi))
            res["by_N"][N] = {
                k: summarize(*np.array(v).T, truth=truth[0]) for k, v in rec.items()
            }
            print(f"{name:18s} N={N:3d}  {time.time() - t0:5.1f}s  "
                  + "  ".join(f"{k}: bias={s['bias']:+.4f} cov={s['coverage']:.3f}"
                              for k, s in res["by_N"][N].items()))
        out["designs"][name] = res
    save("coverage", out)


if __name__ == "__main__":
    main()
