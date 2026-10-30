"""Study 4 (optional): inference after per-participant Thompson-sampling randomization.

Each participant's p_t = clip(P(effect in current context > 0 | own history), 0.1, 0.9).
p_t is a known function of that participant's history, participants remain
independent, so WCLS with the *recorded* p_t should stay valid; the naive
regression has no such guarantee because p_t now co-varies with S_t and with
the participant's own past outcomes.

Note: the estimand itself depends on the policy (habituation makes the effect
history-dependent), so the truth is recomputed under the same adaptive policy.
"""

from __future__ import annotations

import numpy as np

from _util import reps, save, summarize
from jitailab import MRTScenario, ThompsonPolicy, naive_gee, simulate_mrt, true_beta, wcls

PT, N = 0.5, 40
R = reps(1000)


def main():
    sc = MRTScenario(policy=ThompsonPolicy(sigma=1.1, prior_var=1.0, p_min=0.1, p_max=0.9))
    out = {"settings": {"N": N, "T": sc.n_decisions, "ptilde": PT, "R": R, "p_clip": [0.1, 0.9]}, "models": {}}
    p_by_context = []
    for mname, f in {"marginal": ("1",), "context": ("1", "S")}.items():
        truth, truth_mcse = true_beta(sc, f, ptilde=PT, n_pop=4000)
        ests = {"wcls": [], "naive": []}
        for r in range(R):
            d = simulate_mrt(N, sc, seed=[11, len(f), r])
            if mname == "marginal":
                avail = d.I == 1
                p_by_context.append(
                    [d.p[avail & (d.S == 0)].mean(), d.p[avail & (d.S == 1)].mean(),
                     d.p[:, :30][avail[:, :30]].mean(), d.p[:, -30:][avail[:, -30:]].mean()]
                )
            for key, fit in (("wcls", wcls(d, moderators=f, ptilde=PT)), ("naive", naive_gee(d, moderators=f))):
                ests[key].append(np.c_[fit.beta, fit.beta_se, fit.beta_ci()])
        res = {"truth": truth.tolist(), "truth_mcse": truth_mcse.tolist(), "coef": {}}
        for key, arr in ests.items():
            arr = np.array(arr)
            res["coef"][key] = [summarize(*arr[:, j, :].T, truth=truth[j]) for j in range(len(f))]
            print(mname, key, [(round(c["bias"], 4), round(c["coverage"], 3)) for c in res["coef"][key]])
        out["models"][mname] = res
    pc = np.mean(p_by_context, axis=0)
    out["mean_p"] = {"S0": pc[0], "S1": pc[1], "first_30_decisions": pc[2], "last_30_decisions": pc[3]}
    print("mean p_t:", out["mean_p"])
    save("thompson", out)


if __name__ == "__main__":
    main()
