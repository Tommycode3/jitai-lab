"""jitailab: design and analysis of micro-randomized trials for JITAIs."""

from .adaptive import ThompsonPolicy
from .power import (
    EffectTrend,
    constant_trend,
    linear_trend,
    mrt_power,
    mrt_sample_size,
    noncentrality,
    quadratic_trend,
)
from .simulate import FixedPolicy, MRTData, MRTScenario, simulate_mrt
from .truth import true_beta
from .wcls import Fit, naive_gee, sandwich_wls, wcls, wcls_fit

__version__ = "0.1.0"

__all__ = [
    "EffectTrend",
    "Fit",
    "FixedPolicy",
    "MRTData",
    "MRTScenario",
    "ThompsonPolicy",
    "constant_trend",
    "linear_trend",
    "mrt_power",
    "mrt_sample_size",
    "naive_gee",
    "noncentrality",
    "quadratic_trend",
    "sandwich_wls",
    "simulate_mrt",
    "true_beta",
    "wcls",
    "wcls_fit",
]
