# jitai-lab

![coverage](figures/coverage.png)

[![CI](https://github.com/Tommycode3/jitai-lab/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

Tools to design and analyze micro-randomized trials (MRTs) for just-in-time adaptive interventions: a simulator with a known causal excursion effect, the WCLS estimator of Boruvka et al. (2018) with a small-sample sandwich, the Liao et al. (2016) sample-size formula, and simulation studies that try to break all three. When the nudge probability depends on context, the usual GEE analysis is biased by about 30% and its CI coverage falls to 0.618 at N = 80, while WCLS stays at 0.94–0.96. Pure NumPy/SciPy; every study runs on one laptop core in minutes.

<!-- TODO(Tommaso): dictate 3-4 lines in your own words -->

## Install

```bash
git clone https://github.com/Tommycode3/jitai-lab && cd jitai-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python examples/quickstart.py
```

```python
from jitailab import MRTScenario, FixedPolicy, simulate_mrt, wcls, naive_gee, true_beta

sc = MRTScenario(policy=FixedPolicy(p_context0=0.7, p_context1=0.1))  # nudge rarely when "busy"
data = simulate_mrt(n=40, scenario=sc, seed=1)
fit = wcls(data, control=("1", "S", "time"), moderators=("1", "S"), ptilde=0.46)
fit.beta, fit.beta_ci()           # effect when S=0, and the S=1 vs S=0 contrast
```

`bash scripts/reproduce.sh` regenerates every number and figure below into `results/*.json` and `figures/` (`JITAILAB_QUICK=1` for a 10× faster smoke test, as in CI). Seeds are fixed. [`docs/design_notes.md`](docs/design_notes.md) summarizes the published JITAI frameworks (Nahum-Shani et al. 2018; Klasnja et al. 2015).

## The estimand and the estimator

Participant $i$ is followed over decision points $t = 1, \dots, T$. At each one we observe history $H_t$, availability $I_t \in \{0,1\}$ and a context $S_t \subseteq H_t$; if available, a nudge $A_t$ is randomized with known probability $p_t(H_t)$, and then a proximal outcome $Y_{t+1}$ is measured. The causal excursion effect (Boruvka et al. 2018) is

$$
\beta(t; s) \;=\; \mathbb{E}\!\left[\, Y_{t+1}(\bar A_{t-1}, 1) - Y_{t+1}(\bar A_{t-1}, 0) \;\middle|\; I_t = 1,\; S_t(\bar A_{t-1}) = s \right].
$$

Past nudges follow the trial's own policy and only the decision at $t$ deviates, so the estimand depends on that policy: with habituation, a policy that nudges more gets a smaller effect from the next nudge. It's modelled as $\beta(t; S_t) = f_t(S_t)^\top \beta$, for example $f_t = 1$, $(1, S_t)^\top$ or $(1, t/(T-1))^\top$.

WCLS picks a reference probability $\tilde p_t(S_t)$ that depends only on the moderators, a control working model $g_t(H_t)^\top\alpha$, and weights $W_t = (\tilde p_t/p_t)^{A_t}\,\big((1-\tilde p_t)/(1-p_t)\big)^{1-A_t}$, then solves

$$
\sum_{i=1}^{N} \sum_{t=1}^{T} I_{it}\, W_{it} \Big( Y_{i,t+1} - g_{it}^\top \alpha - \big(A_{it} - \tilde p_t(S_{it})\big) f_{it}^\top \beta \Big)
\begin{pmatrix} g_{it} \\ \big(A_{it} - \tilde p_t(S_{it})\big) f_{it} \end{pmatrix} = 0 .
$$

The weights replace $p_t(H_t)$ by $\tilde p_t(S_t)$, so the estimand averages over everything not in $f_t$. Centering $A_t - \tilde p_t$ makes the treatment term orthogonal to any function of $S_t$, so $\hat\beta$ stays consistent when the control model is wrong; the one used here deliberately omits carryover, burden and the AR(1) noise. If $f_t^\top\beta$ is itself misspecified, $\hat\beta$ converges to the projection $\beta^\star = \arg\min_b \mathbb{E}\big[\sum_t I_t\, \tilde p_t (1-\tilde p_t) (\beta(t; S_t) - f_t^\top b)^2\big]$, which is how `true_beta` computes ground truth from the per-person-time effects the simulator records.

The variance is the Mancl & DeRouen (2001) bias-corrected sandwich,

$$
\widehat{\operatorname{Var}}(\hat\theta) = B^{-1} \Big( \sum_{i=1}^{N} X_i^\top W_i\, \tilde e_i\, \tilde e_i^\top W_i X_i \Big) B^{-1},
\qquad \tilde e_i = (I - H_{ii})^{-1} e_i,\quad H_{ii} = X_i B^{-1} X_i^\top W_i ,
$$

with $(I - H_{ii})^{-1}$ evaluated through Woodbury (a $d \times d$ solve per participant instead of $T \times T$) and CIs from $t_{N-p-q}$.

The naive analysis regresses $Y_{t+1}$ on $g_t$ and $A_t f_t$ with no weights or centering. With $f_t = 1$ its coefficient tends roughly to a variance-weighted average of context effects,

$$
\beta_{\text{naive}} \approx \frac{\sum_s \Pr(S = s)\, p(s)\,(1-p(s))\, \beta(s)}{\sum_s \Pr(S = s)\, p(s)\,(1-p(s))},
$$

which equals the plain average only if $p$ or $\beta$ doesn't depend on $s$. In the default scenario ($\Pr(S{=}1) = 0.4$, $p = 0.7$ vs $0.1$, $\beta(0) \approx 0.280$, $\beta(1) \approx 0.005$) this predicts $\beta_{\text{naive}} \approx 0.219$ against a truth of $0.170$, a bias of $\approx +0.049$. The simulations find $+0.048$ to $+0.051$.

## Results

Default scenario: $T = 150$ decision points (30 days × 5), availability 0.8, $\beta_t = 0.4 - 0.3\,S_t - 0.1\,\tau_t - 0.05\,D_t$ with habituation through the recent dose $D_t$, carryover 0.1, burden 0.02, random intercept SD 0.5, AR(1) noise $\phi = 0.3$. Marginal effect with $\tilde p = 0.46$, true value $0.1702 \pm 0.0001$, 1000 replications per cell, ± one Monte Carlo SE.

| design | N | WCLS bias | WCLS coverage (MD, $t$) | WCLS coverage (plain, $z$) | naive bias | naive coverage |
|---|---|---|---|---|---|---|
| constant $p = 0.46$ | 20 | −0.0001 ± 0.0014 | 0.961 ± 0.006 | 0.932 ± 0.008 | −0.0001 ± 0.0014 | 0.961 ± 0.006 |
| | 40 | +0.0016 ± 0.0011 | 0.946 ± 0.007 | 0.936 ± 0.008 | +0.0016 ± 0.0011 | 0.946 ± 0.007 |
| | 80 | +0.0003 ± 0.0007 | 0.959 ± 0.006 | 0.946 ± 0.007 | +0.0003 ± 0.0007 | 0.959 ± 0.006 |
| $p = 0.7$ if $S{=}0$, $0.1$ if $S{=}1$ | 20 | +0.0022 ± 0.0019 | 0.958 ± 0.006 | 0.929 ± 0.008 | +0.0510 ± 0.0018 (+30%) | 0.885 ± 0.010 |
| | 40 | +0.0012 ± 0.0014 | 0.957 ± 0.006 | 0.935 ± 0.008 | +0.0500 ± 0.0013 (+29%) | 0.769 ± 0.013 |
| | 80 | −0.0008 ± 0.0010 | 0.942 ± 0.007 | 0.933 ± 0.008 | +0.0477 ± 0.0009 (+28%) | 0.618 ± 0.015 |

With constant $p = \tilde p$ all weights are 1 and the two fits coincide. Under context-dependent randomization the naive bias doesn't shrink with N but its CI does, so coverage falls. Put $S$ into $f_t$ and the naive fit recovers both coefficients (bias +0.0018 and −0.0035 at N = 40); leave it out, as in a time-trend model, and the intercept is off by +0.0488. With per-participant Thompson sampling, where $p_t$ depends on each person's past outcomes, the naive fit is biased even with $S$ in the model (+0.0814 on $\beta_0$, coverage 0.823), while WCLS with the recorded $p_t$ has bias +0.0022 and coverage 0.950 on the same coefficient.

The sample-size formula was checked against simulated power on a HeartSteps-like layout (42 days × 5, $\rho = 0.4$, availability 0.7, average standardized effect 0.1). It asks for $N = 25$ (constant trend) and $N = 32$ (quadratic, peak on day 28), and the WCLS $F$ test reaches 0.796 and 0.817 under the formula's working assumptions, 0.803 and 0.823 with AR(1) 0.5 errors plus a random intercept. Type-I error at those $N$ is 0.038–0.054. It is optimistic at very small $N$ (quadratic, $N = 12$: 0.246 vs 0.156 simulated) and doesn't anticipate heterogeneity of the effect itself: a person-level effect SD of 0.1 drops power at the recommended $N$ to 0.651 and 0.678.

## Limitations

- Continuous proximal outcome, binary treatment, lag-1 only. No delayed effects, multi-level treatments or binary outcomes (see Qian et al. 2021).
- $\tilde p$ is treated as known. Estimating it adds a step the sandwich doesn't account for.
- No missing outcomes or dropout. Unavailable decision points are excluded by definition of the estimand.
- Stylized simulator: one binary context, linear dynamics, illustrative effect sizes. No number here is a claim about any real intervention.
- The sample-size formula is optimistic at very small $N$ and under person-level effect heterogeneity. Check by simulation.
- Adaptive randomization only in the per-participant form. Pooled algorithms make participants dependent and need theory for adaptively collected data (Zhang, Janson & Murphy 2021).
- Mancl–DeRouen is slightly conservative at $N = 20$; other small-sample corrections aren't compared.

## References

DOIs resolved on Crossref on 2026-10-30.

* Boruvka, A., Almirall, D., Witkiewitz, K., & Murphy, S. A. (2018). Assessing time-varying causal effect moderation in mobile health. *JASA*, 113(523), 1112–1121. https://doi.org/10.1080/01621459.2017.1305274
* Klasnja, P., Hekler, E. B., Shiffman, S., Boruvka, A., Almirall, D., Tewari, A., & Murphy, S. A. (2015). Microrandomized trials: An experimental design for developing just-in-time adaptive interventions. *Health Psychology*, 34(Suppl), 1220–1228. https://doi.org/10.1037/hea0000305
* Liao, P., Klasnja, P., Tewari, A., & Murphy, S. A. (2016). Sample size calculations for micro-randomized trials in mHealth. *Statistics in Medicine*, 35(12), 1944–1971. https://doi.org/10.1002/sim.6847
* Liao, P., Greenewald, K., Klasnja, P., & Murphy, S. (2020). Personalized HeartSteps: A reinforcement learning algorithm for optimizing physical activity. *Proc. ACM IMWUT*, 4(1), 1–22. https://doi.org/10.1145/3381007
* Mancl, L. A., & DeRouen, T. A. (2001). A covariance estimator for GEE with improved small-sample properties. *Biometrics*, 57(1), 126–134. https://doi.org/10.1111/j.0006-341X.2001.00126.x
* Nahum-Shani, I., Smith, S. N., Spring, B. J., Collins, L. M., Witkiewitz, K., Tewari, A., & Murphy, S. A. (2018). Just-in-time adaptive interventions (JITAIs) in mobile health: Key components and design principles for ongoing health behavior support. *Annals of Behavioral Medicine*, 52(6), 446–462. https://doi.org/10.1007/s12160-016-9830-8
* Qian, T., Yoo, H., Klasnja, P., Almirall, D., & Murphy, S. A. (2021). Estimating time-varying causal excursion effects in mobile health with binary outcomes. *Biometrika*, 108(3), 507–527. https://doi.org/10.1093/biomet/asaa070
* Zhang, K. W., Janson, L., & Murphy, S. A. (2021). Statistical inference with M-estimators on adaptively collected data. *NeurIPS 2021*. arXiv:2104.14074.

MIT © 2026 Tommaso Maggiolo. See [`CITATION.cff`](CITATION.cff) to cite.
