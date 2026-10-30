"""Render all figures in figures/ from the JSON files in results/."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _util import FIGURES, RESULTS
from jitailab import constant_trend, mrt_power, quadratic_trend

# categorical slots in fixed order (validated default palette)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.axisbelow": True, "font.size": 10, "lines.linewidth": 2, "legend.frameon": False,
})


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def fig_coverage_and_bias():
    res = load("coverage")
    Ns = res["settings"]["Ns"]
    R = res["settings"]["R"]
    band = 1.96 * np.sqrt(0.95 * 0.05 / R)
    est = {"wcls_md": ("WCLS + Mancl-DeRouen, t", BLUE), "wcls_plain": ("WCLS, plain sandwich, z", AQUA),
           "naive": ("naive GEE (unweighted)", ORANGE)}
    titles = {"constant": "constant p = 0.46", "context_dependent": "p = 0.7 if S=0, 0.1 if S=1"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (dname, d) in zip(axes, res["designs"].items()):
        ax.axhspan(0.95 - band, 0.95 + band, color=GRID, alpha=0.7, lw=0)
        ax.axhline(0.95, color=INK2, lw=1, ls="--")
        for j, (k, (lab, c)) in enumerate(est.items()):
            x = np.arange(len(Ns)) + (j - 1) * 0.12
            cov = [d["by_N"][str(N)][k]["coverage"] for N in Ns]
            se = [d["by_N"][str(N)][k]["coverage_mcse"] for N in Ns]
            ax.errorbar(x, cov, yerr=1.96 * np.array(se), color=c, marker="o", ms=7, capsize=0, label=lab)
        ax.set_xticks(range(len(Ns)), [f"N = {N}" for N in Ns])
        ax.set_title(titles[dname], color=INK)
    axes[0].set_ylabel("empirical coverage of nominal 95% CI")
    axes[1].legend(loc="lower left")
    fig.suptitle(f"Coverage of the marginal excursion effect ({R} replications; grey = MC band around 0.95)", color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "coverage.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (dname, d) in zip(axes, res["designs"].items()):
        ax.axhline(0, color=INK2, lw=1)
        for j, k in enumerate(["wcls_md", "naive"]):
            lab, c = est[k]
            x = np.arange(len(Ns)) + (j - 0.5) * 0.15
            b = [d["by_N"][str(N)][k]["bias"] for N in Ns]
            se = [d["by_N"][str(N)][k]["bias_mcse"] for N in Ns]
            ax.errorbar(x, b, yerr=1.96 * np.array(se), color=c, marker="o", ms=7, ls="none",
                        label=lab.split(",")[0].replace(" + Mancl-DeRouen", ""))
        ax.set_xticks(range(len(Ns)), [f"N = {N}" for N in Ns])
        ax.set_title(f"{titles[dname]}  (true effect {d['truth']:.3f})", color=INK)
    axes[0].set_ylabel("bias of estimate (± 1.96 MC SE)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Bias: the naive regression is biased only when p_t depends on an effect moderator", color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "bias.png", dpi=150)
    plt.close(fig)


def fig_power():
    res = load("power")
    s = res["settings"]
    trends = {"constant": constant_trend(s["days"], s["per_day"], s["avg"]),
              "quadratic": quadratic_trend(s["days"], s["per_day"], s["avg"], 28, 0.0)}
    scen = {"iid": ("working assumptions (iid)", BLUE), "ar1_ri": ("AR(1) 0.5 + random intercept", AQUA),
            "het_effect": ("person-level effect SD 0.1", ORANGE)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (tname, t) in zip(axes, res["trends"].items()):
        Ns = [r["N"] for r in t["scenarios"]["iid"]["power"]]
        grid = np.arange(min(Ns) - 2, max(Ns) + 3)
        ax.plot(grid, [mrt_power(N, trends[tname], s["tau"], s["rho"]) for N in grid], color=INK2, lw=2,
                label="Liao et al. (2016) formula")
        for j, (sn, (lab, c)) in enumerate(scen.items()):
            rows = t["scenarios"][sn]["power"]
            x = np.array([r["N"] for r in rows]) + (j - 1) * 0.4
            ax.errorbar(x, [r["empirical"] for r in rows], yerr=[1.96 * r["mcse"] for r in rows],
                        color=c, marker="o", ms=6, ls="none", label=f"simulated: {lab}")
        ax.axhline(0.8, color=GRID, lw=1.5, zorder=0)
        ax.axvline(t["N_for_80pct"], color=GRID, lw=1.5, zorder=0)
        ax.set_title(f"{tname} trend, avg standardized effect {s['avg']}  (N for 80% = {t['N_for_80pct']})", color=INK, fontsize=9.5)
        ax.set_xlabel("participants N")
    axes[0].set_ylabel("power (α = 0.05)")
    axes[1].legend(loc="lower right", fontsize=8.5)
    fig.suptitle(f"Sample-size formula vs simulated WCLS power ({s['days']} days × {s['per_day']}/day, "
                 f"availability {s['tau']}, p = {s['rho']}; {s['R']} reps/point)", color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "power.png", dpi=150)
    plt.close(fig)


def fig_moderation():
    res = load("moderation")
    m = res["models"]["time"]
    g = np.array(m["grid"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ex = m["example_wcls_band"]
    ax.fill_between(g, ex["lo"], ex["hi"], color=BLUE, alpha=0.15, lw=0, label="one trial: WCLS 95% pointwise CI")
    ax.plot(g, m["curve_truth"], color=INK, lw=2, ls="--", label="true projection β(t)")
    ax.plot(g, m["curve_mean_wcls"], color=BLUE, label="WCLS, Monte Carlo mean")
    ax.plot(g, m["curve_mean_naive"], color=ORANGE, label="naive, Monte Carlo mean")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel("study time τ = t/(T−1)")
    ax.set_ylabel("proximal effect")
    ax.set_title("Moderation by time: β(t) = β₀ + β₁τ", color=INK)
    ax.legend(fontsize=8.5, loc="upper right")

    ax = axes[1]
    c = res["models"]["context"]
    tr = np.array(c["truth"])
    truth = [tr[0], tr[0] + tr[1]]
    w = [c["coef"]["wcls"][0]["mean"], c["coef"]["wcls"][0]["mean"] + c["coef"]["wcls"][1]["mean"]]
    nv = [c["coef"]["naive"][0]["mean"], c["coef"]["naive"][0]["mean"] + c["coef"]["naive"][1]["mean"]]
    x = np.arange(2)
    ax.bar(x - 0.25, truth, 0.24, color=INK2, label="truth")
    ax.bar(x, w, 0.24, color=BLUE, label="WCLS mean")
    ax.bar(x + 0.25, nv, 0.24, color=ORANGE, label="naive mean")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(x, ["S = 0", "S = 1"])
    ax.set_title("Moderation by context S (S is in f: naive OK)", color=INK)
    ax.legend(fontsize=8.5)
    fig.suptitle(f"Effect moderation, context-dependent design (N = {res['settings']['N']}, {res['settings']['R']} reps)", color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "moderation.png", dpi=150)
    plt.close(fig)


def fig_thompson():
    res = load("thompson")
    labels, cov_w, cov_n, se_w, se_n = [], [], [], [], []
    names = {"marginal": ["marginal β"], "context": ["β₀ (S=0)", "β₁ (S contrast)"]}
    for mname, mm in res["models"].items():
        for j, lab in enumerate(names[mname]):
            labels.append(lab)
            cov_w.append(mm["coef"]["wcls"][j]["coverage"]); se_w.append(mm["coef"]["wcls"][j]["coverage_mcse"])
            cov_n.append(mm["coef"]["naive"][j]["coverage"]); se_n.append(mm["coef"]["naive"][j]["coverage_mcse"])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.axhline(0.95, color=INK2, ls="--", lw=1)
    ax.errorbar(x - 0.08, cov_w, yerr=1.96 * np.array(se_w), color=BLUE, marker="o", ms=7, ls="none", label="WCLS (recorded p_t)")
    ax.errorbar(x + 0.08, cov_n, yerr=1.96 * np.array(se_n), color=ORANGE, marker="o", ms=7, ls="none", label="naive GEE")
    ax.set_xticks(x, labels)
    ax.set_ylabel("95% CI coverage")
    ax.set_title(f"Per-participant Thompson sampling (N = {res['settings']['N']}, {res['settings']['R']} reps)", color=INK)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "thompson.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_coverage_and_bias()
    fig_power()
    fig_moderation()
    fig_thompson()
    print("figures written to", FIGURES)
