"""Generative model for micro-randomized trial (MRT) data.

Every participant ``i = 1..N`` is followed over decision points ``t = 0..T-1``.
At each decision point:

1. a binary context ``S_t`` evolves as a two-state Markov chain
   (stay with probability ``context_persistence``, otherwise redraw from
   Bernoulli(``context_prob``)); the chain is stationary with P(S=1) = context_prob;
2. availability ``I_t ~ Bernoulli(avail_prob)``;
3. the randomization policy returns ``p_t = P(A_t = 1 | H_t)``; if available,
   ``A_t ~ Bernoulli(p_t)``, otherwise ``A_t = 0``;
4. the proximal outcome is

   Y_{t+1} = a0 + aS*S_t + aTime*tau_t + kappa*A_{t-1} - zeta*D_t
             + A_t * beta_t + u_i + e_t

   with ``tau_t = t/(T-1)``, a recent-dose state ``D_t = lam*D_{t-1} + A_{t-1}``
   (``D_0 = 0``), and the *individual* proximal effect

   beta_t = b0 + bS*S_t + bTime*tau_t - eta*D_t.

   ``kappa`` is carryover (yesterday's nudge still moves today's outcome),
   ``zeta`` is burden (recent nudges depress the baseline), ``eta`` is
   habituation (recent nudges shrink the effect of the next one).
   ``u_i ~ N(0, sigma_u^2)`` is a person-level intercept and ``e_t`` is a
   stationary AR(1) process with marginal SD ``sigma_e``.

Because ``A_t`` enters ``Y_{t+1}`` only through ``A_t * beta_t``, the
conditional effect ``beta_t`` stored in ``MRTData.true_effect`` is exactly
``Y_{t+1}(Abar_{t-1}, 1) - Y_{t+1}(Abar_{t-1}, 0)`` for that person-time.
Causal excursion effects are expectations of it (see :mod:`jitailab.truth`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np


class Policy(Protocol):
    """A randomization policy. Must be vectorized over participants."""

    def reset(self, n: int, rng: np.random.Generator) -> None: ...

    def probs(self, t: int, S: np.ndarray) -> np.ndarray: ...

    def update(self, t: int, S: np.ndarray, I: np.ndarray, A: np.ndarray, Y: np.ndarray) -> None: ...


@dataclass
class FixedPolicy:
    """Randomization probability that depends only on the current context.

    ``p_context0`` is used when S_t = 0 and ``p_context1`` when S_t = 1.
    Set them equal for a context-independent (constant) design.
    """

    p_context0: float = 0.4
    p_context1: float = 0.4

    def reset(self, n, rng):
        pass

    def probs(self, t, S):
        return np.where(S == 1, self.p_context1, self.p_context0).astype(float)

    def update(self, t, S, I, A, Y):
        pass


@dataclass
class MRTScenario:
    """All knobs of the data-generating process (defaults: a 30-day, 5/day MRT)."""

    n_decisions: int = 150
    avail_prob: float = 0.8
    context_prob: float = 0.4
    context_persistence: float = 0.6
    # baseline (control) mean
    a0: float = 1.0
    aS: float = -0.5
    aTime: float = 0.3
    # proximal effect
    b0: float = 0.4
    bS: float = -0.3
    bTime: float = -0.1
    # nuisance dynamics
    eta: float = 0.05  # habituation
    lam: float = 0.7  # dose decay
    kappa: float = 0.1  # carryover of A_{t-1}
    zeta: float = 0.02  # burden on baseline
    # noise
    sigma_u: float = 0.5
    sigma_e: float = 1.0
    ar_phi: float = 0.3
    policy: Policy = field(default_factory=lambda: FixedPolicy(0.7, 0.1))

    def with_(self, **kw) -> "MRTScenario":
        return replace(self, **kw)


@dataclass
class MRTData:
    """Arrays of shape (N, T). ``Y[:, t]`` is the proximal outcome Y_{t+1}."""

    Y: np.ndarray
    A: np.ndarray
    I: np.ndarray
    p: np.ndarray
    S: np.ndarray
    dose: np.ndarray
    true_effect: np.ndarray

    @property
    def n(self) -> int:
        return self.Y.shape[0]

    @property
    def T(self) -> int:
        return self.Y.shape[1]

    @property
    def time(self) -> np.ndarray:
        """Scaled time tau_t = t/(T-1) in [0, 1], broadcast to (N, T)."""
        tau = np.arange(self.T) / max(self.T - 1, 1)
        return np.broadcast_to(tau, self.Y.shape)

    @property
    def cluster(self) -> np.ndarray:
        return np.broadcast_to(np.arange(self.n)[:, None], self.Y.shape)

    def feature(self, name: str) -> np.ndarray:
        """Named covariates usable in control / moderator models."""
        if name in ("1", "intercept"):
            return np.ones(self.Y.shape)
        if name == "S":
            return self.S.astype(float)
        if name == "time":
            return np.asarray(self.time, dtype=float)
        if name == "A_prev":
            out = np.zeros(self.Y.shape)
            out[:, 1:] = self.A[:, :-1]
            return out
        if name == "dose":
            return self.dose
        raise KeyError(f"unknown feature {name!r}")

    def design(self, names) -> np.ndarray:
        """Flattened (N*T, len(names)) matrix of named features."""
        return np.column_stack([self.feature(nm).ravel() for nm in names])


def simulate_mrt(n: int, scenario: MRTScenario | None = None, seed=None) -> MRTData:
    """Simulate one MRT with ``n`` participants. Loops over time, vectorized over people."""
    sc = scenario or MRTScenario()
    rng = np.random.default_rng(seed)
    T = sc.n_decisions
    pol = sc.policy
    pol.reset(n, rng)

    shape = (n, T)
    Y = np.empty(shape)
    A = np.zeros(shape, dtype=int)
    I = np.zeros(shape, dtype=int)
    P = np.empty(shape)
    S = np.zeros(shape, dtype=int)
    Dm = np.empty(shape)
    B = np.empty(shape)

    u = rng.normal(0.0, sc.sigma_u, n)
    innov_sd = sc.sigma_e * np.sqrt(1.0 - sc.ar_phi**2)
    e = rng.normal(0.0, sc.sigma_e, n)  # stationary start
    s = (rng.random(n) < sc.context_prob).astype(int)
    dose = np.zeros(n)
    a_prev = np.zeros(n, dtype=int)

    for t in range(T):
        if t > 0:
            redraw = rng.random(n) >= sc.context_persistence
            s = np.where(redraw, (rng.random(n) < sc.context_prob).astype(int), s)
            e = sc.ar_phi * e + rng.normal(0.0, innov_sd, n)
            dose = sc.lam * dose + a_prev
        tau = t / max(T - 1, 1)
        avail = (rng.random(n) < sc.avail_prob).astype(int)
        p = np.asarray(pol.probs(t, s), dtype=float)
        a = avail * (rng.random(n) < p).astype(int)
        beta = sc.b0 + sc.bS * s + sc.bTime * tau - sc.eta * dose
        y = (
            sc.a0
            + sc.aS * s
            + sc.aTime * tau
            + sc.kappa * a_prev
            - sc.zeta * dose
            + a * beta
            + u
            + e
        )
        pol.update(t, s, avail, a, y)
        Y[:, t], A[:, t], I[:, t], P[:, t], S[:, t] = y, a, avail, p, s
        Dm[:, t], B[:, t] = dose, beta
        a_prev = a

    return MRTData(Y=Y, A=A, I=I, p=P, S=S, dose=Dm, true_effect=B)
