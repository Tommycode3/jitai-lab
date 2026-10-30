"""Study 3: effect moderation by time and by context (context-dependent design).

Moderator models (control g = (1, S, time), ptilde = 0.46 constant):
  time     f_t = (1, tau_t)   ->  beta(t) = beta_0 + beta_1 tau_t
  context  f_t = (1, S_t)     ->  beta(S) = beta_0 + beta_1 S
Truth = weighted projection of individual effects (see jitailab.truth).
Reports bias and coverage for every coefficient and stores the Monte Carlo
mean curve of beta(t) for WCLS and the naive regression.
"""

from __future__ import annotations

import numpy as np

from _util import reps, save, summarize
from jitailab import FixedPolicy, MRTScenario, naive_gee, simulate_mrt, true_beta, wcls

PT, N = 0.46, 40
R = reps(1000)
MODELS = {"time": ("1", "time"), "context": ("1", "S")}
GRID = np.linspace(0, 1, 21)


def main():
    sc = MRTScenario(policy=FixedPolicy(0.7, 0.1))
    out = {"settings": {"N": N, "T": sc.n_decisions, "ptilde": PT, "R": R}, "models": {}}
    for mi, (mname, f) in enumerate(MODELS.items()):
        truth, truth_mcse = true_beta(sc, f, ptilde=PT, n_pop=8000)
        L = np.column_stack([np.ones_like(GRID), GRID])
        ests = {"wcls": [], "naive": []}
        curves = {"wcls": [], "naive": []}
        example = None
        for r in range(R):
            d = simulate_mrt(N, sc, seed=[7, mi, r])
            for key, fit in (("wcls", wcls(d, moderators=f, ptilde=PT)), ("naive", naive_gee(d, moderators=f))):
                ci = fit.beta_ci()
                ests[key].append(np.c_[fit.beta, fit.beta_se, ci])
                curves[key].append(L @ fit.beta)
                if r == 0 and key == "wcls":
                    e, s, band = fit.linear_combination(L)
                    example = {"est": e.tolist(), "lo": band[:, 0].tolist(), "hi": band[:, 1].tolist()}
        res = {"truth": truth.tolist(), "truth_mcse": truth_mcse.tolist(), "coef": {}}
        for key, arr in ests.items():
            arr = np.array(arr)  # (R, p, 4)
            res["coef"][key] = [summarize(*arr[:, j, :].T, truth=truth[j]) for j in range(len(f))]
            res[f"curve_mean_{key}"] = np.mean(curves[key], axis=0).tolist()
        res["curve_truth"] = (L @ truth).tolist()
        res["example_wcls_band"] = example
        res["grid"] = GRID.tolist()
        out["models"][mname] = res
        for key in ("wcls", "naive"):
            print(mname, key, [(round(c["bias"], 4), round(c["coverage"], 3)) for c in res["coef"][key]])
    save("moderation", out)


if __name__ == "__main__":
    main()
