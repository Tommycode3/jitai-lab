"""Per-participant Thompson-sampling randomization (illustrative, optional).

Each participant has their own Bayesian linear model, learned only from their
own past data,

    Y = theta_0 + theta_1 S + A (theta_2 + theta_3 S) + noise,  noise ~ N(0, sigma^2),

with a N(0, prior_var I) prior and *known* sigma. At each decision point the
probability of sending a nudge is the posterior probability that the effect in
the current context is positive, clipped to [p_min, p_max]:

    p_t = clip( P(theta_2 + theta_3 S_t > 0 | H_t), p_min, p_max ).

Why per participant: the probability then depends only on that participant's own
history H_t, participants stay independent, and the WCLS theory -- which needs
p_t to be a known function of H_t -- still applies when the *recorded* p_t is
used. Pooling across participants (as practical bandits often do) makes
participants dependent; inference then needs the adaptive-sampling theory
discussed in the README and is NOT covered here.

This is a simplified teaching device, not the algorithm of any deployed study.
Clipping keeps p_t away from 0 and 1, which bounds the WCLS weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class ThompsonPolicy:
    sigma: float = 1.0
    prior_var: float = 1.0
    p_min: float = 0.1
    p_max: float = 0.9

    def reset(self, n, rng):
        d = 4
        self.prec = np.tile(np.eye(d) / self.prior_var, (n, 1, 1))
        self.b = np.zeros((n, d))
        self._p_last = None

    def _effect_contrast(self, S):
        c = np.zeros((S.size, 4))
        c[:, 2] = 1.0
        c[:, 3] = S
        return c

    def probs(self, t, S):
        cov = np.linalg.inv(self.prec)
        mean = np.einsum("nij,nj->ni", cov, self.b)
        c = self._effect_contrast(S)
        m = np.einsum("ni,ni->n", c, mean)
        v = np.einsum("ni,nij,nj->n", c, cov, c)
        prob = stats.norm.cdf(m / np.sqrt(v))
        return np.clip(prob, self.p_min, self.p_max)

    def update(self, t, S, I, A, Y):
        x = np.column_stack([np.ones(S.size), S, A, A * S]).astype(float)
        x = x * I[:, None]  # unavailable decision points do not update
        yy = Y * I
        self.prec = self.prec + np.einsum("ni,nj->nij", x, x) / self.sigma**2
        self.b = self.b + x * (yy / self.sigma**2)[:, None]
