"""Quickstart: simulate an MRT, estimate causal excursion effects, size a trial.

Run:  python examples/quickstart.py
"""

import numpy as np

from jitailab import (
    FixedPolicy,
    MRTScenario,
    mrt_power,
    mrt_sample_size,
    naive_gee,
    quadratic_trend,
    simulate_mrt,
    true_beta,
    wcls,
)

# 1. A 30-day MRT, 5 decision points/day. The nudge helps when S=0 ("free")
#    and does ~nothing when S=1 ("busy"); the designers, worried about burden,
#    nudge rarely when busy: p = 0.7 if S=0, 0.1 if S=1.
scenario = MRTScenario(policy=FixedPolicy(p_context0=0.7, p_context1=0.1))
data = simulate_mrt(n=40, scenario=scenario, seed=1)

# 2. Marginal causal excursion effect (f_t = 1), ptilde constant.
fit = wcls(data, control=("1", "S", "time"), moderators=("1",), ptilde=0.46)
naive = naive_gee(data, control=("1", "S", "time"), moderators=("1",))
truth, _ = true_beta(scenario, moderators=("1",), ptilde=0.46)
print(f"true marginal effect   {truth[0]: .3f}")
print(f"WCLS                   {fit.beta[0]: .3f}  95% CI {np.round(fit.beta_ci()[0], 3)}")
print(f"naive GEE              {naive.beta[0]: .3f}")

# One trial is one draw. Repeat the trial 200 times to see the bias:
reps = [simulate_mrt(40, scenario, seed=1000 + r) for r in range(200)]
w = np.mean([wcls(d, ptilde=0.46).beta[0] for d in reps])
nv = np.mean([naive_gee(d).beta[0] for d in reps])
print(f"mean over 200 trials:  WCLS {w:.3f}   naive {nv:.3f}   (truth {truth[0]:.3f})")

# 3. Is the effect moderated by context? f_t = (1, S_t)
mod = wcls(data, moderators=("1", "S"), ptilde=0.46)
est, se, ci = mod.linear_combination(np.array([[1, 0], [1, 1]]))
for s, e, c in zip((0, 1), est, ci):
    print(f"effect when S={s}: {e: .3f}  95% CI {np.round(c, 3)}")

# 4. How many participants for a 6-week trial, 5 decisions/day, p = 0.4,
#    availability 0.7, quadratic trend averaging 0.1 SD, peak on day 28?
trend = quadratic_trend(n_days=42, per_day=5, avg=0.1, day_max=28, initial=0.0)
N = mrt_sample_size(trend, avail=0.7, rand_prob=0.4, power=0.8)
print(f"N for 80% power: {N}  (power at N: {mrt_power(N, trend, 0.7, 0.4):.3f})")
