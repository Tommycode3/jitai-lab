"""Shared helpers for the simulation studies (Monte Carlo summaries, I/O)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# Set JITAILAB_QUICK=1 for a fast smoke run (fewer replications); numbers in the
# README come from the default (full) run.
QUICK = os.environ.get("JITAILAB_QUICK") == "1"


def reps(full: int) -> int:
    return max(20, full // 10) if QUICK else full


def summarize(est, se, lo, hi, truth) -> dict:
    """Bias, empirical SD, mean SE, coverage -- each with its Monte Carlo SE."""
    est, se, lo, hi = map(np.asarray, (est, se, lo, hi))
    R = est.size
    cover = (lo <= truth) & (truth <= hi)
    cov = cover.mean()
    sd = est.std(ddof=1)
    return {
        "R": int(R),
        "truth": float(truth),
        "mean": float(est.mean()),
        "bias": float(est.mean() - truth),
        "bias_mcse": float(sd / np.sqrt(R)),
        "rel_bias_pct": float(100 * (est.mean() - truth) / truth) if truth != 0 else None,
        "emp_sd": float(sd),
        "emp_sd_mcse": float(sd / np.sqrt(2 * (R - 1))),
        "mean_se": float(se.mean()),
        "coverage": float(cov),
        "coverage_mcse": float(np.sqrt(cov * (1 - cov) / R)),
    }


def save(name: str, obj) -> Path:
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2))
    return path
