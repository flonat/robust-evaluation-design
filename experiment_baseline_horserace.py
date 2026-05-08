"""Experiment: Baseline horse-race across the environment family.

Compare six regulator policies on N_ENV sampled environments:
  1. det_low:        deterministic (sigma=0.3, delta=0.2)
  2. det_high:       deterministic (sigma=0.8, delta=0.8)
  3. uniform_pf5:    n=5 uniform portfolio (sigma=0.5, delta=0.5)
  4. payoff_match:   minimum-cost single metric on gaming-proof boundary
                     (or regulator_optimum if boundary infeasible)
  5. opt_portfolio:  grid-search uniform portfolio over n x (sigma, delta)
  6. oracle:         same as 5 under rational developer (upper bound here)

Writes:
  paper-aij/paper/figures/baseline_horserace.pdf       (regulator-payoff CDFs)
  paper-aij/paper/output/tables/baseline_horserace.tex (summary statistics)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from config import N_ENV, OUT_FIG, OUT_TBL, SEED, configure_matplotlib

configure_matplotlib()
import math

from model import (Environment, alpha_star, critical_portfolio_size,
                   portfolio_alpha_star, regulator_optimum, regulator_payoff)
from sample_environments import sample

POLICY_LABELS = {
    "det_low":        r"Det. low $(0.3, 0.2)$",
    "det_high":       r"Det. high $(0.8, 0.8)$",
    "uniform_pf5":    r"Uniform portfolio $n{=}5$",
    "payoff_match":   r"Payoff-matched",
    "opt_portfolio":  r"Optimal robust portfolio",
    "oracle":         r"Oracle",
}


def eval_single(env: Environment, sigma: float, delta: float) -> tuple[float, float]:
    a = alpha_star(env.B, env.c, env.gamma, sigma, delta, env.F)
    u = regulator_payoff(a, sigma, delta, env.k_sigma, env.k_delta, n=1)
    return a, u


def eval_portfolio(env: Environment, sigma: float, delta: float, n: int) -> tuple[float, float]:
    a = portfolio_alpha_star(env.B, env.c, env.gamma, sigma, delta, env.F, n)
    u = regulator_payoff(a, sigma, delta, env.k_sigma, env.k_delta, n=n, per_metric_cost=True)
    return a, float(u)


def payoff_matched_metric(env: Environment) -> tuple[float, float]:
    """Minimum-implementation-cost single metric that satisfies B*sigma + delta*F >= c.

    Analytic Lagrangian (Theorem on Optimal Single Metric, Part 2):
        sigma_gp = lam * B / k_sigma,  delta_gp = lam * F / k_delta,
        with B*sigma + delta*F = c  =>  lam = c / (B^2/k_sigma + F^2/k_delta).
    Clamp to [0,1]; if the clipped solution violates the gaming-proof constraint
    (i.e. interior-feasibility fails per Theorem on Optimal Single Metric, Part 3),
    fall back to the regulator's interior grid-search optimum.
    Tolerance 1e-6 is standard floating-point slack for inequalities of this form.
    """
    phi = env.B ** 2 / env.k_sigma + env.F ** 2 / env.k_delta
    if phi <= 0:
        opt = regulator_optimum(env)
        return opt["sigma"], opt["delta"]
    lam = env.c / phi
    sigma = np.clip(lam * env.B / env.k_sigma, 0.0, 1.0)
    delta = np.clip(lam * env.F / env.k_delta, 0.0, 1.0)
    if env.B * sigma + delta * env.F < env.c - 1e-6:
        opt = regulator_optimum(env)
        return opt["sigma"], opt["delta"]
    return float(sigma), float(delta)


def optimal_portfolio(env: Environment, n_max: int = 20,
                      sigma_grid: int = 21, delta_grid: int = 21) -> tuple[int, float, float, float, float]:
    """Grid-search optimal uniform portfolio (n, sigma, delta) maximising u_R."""
    sigmas = np.linspace(0.0, 1.0, sigma_grid)
    deltas = np.linspace(0.0, 1.0, delta_grid)
    best = (1, 0.0, 0.0, 1.0, -np.inf)
    for n in range(1, n_max + 1):
        for s in sigmas:
            for d in deltas:
                a, u = eval_portfolio(env, float(s), float(d), n)
                if u > best[4]:
                    best = (n, float(s), float(d), float(a), float(u))
    return best  # (n, sigma, delta, alpha, payoff)


def run() -> dict:
    envs = sample(N_ENV, SEED)
    results = {k: {"alpha": [], "payoff": [], "gaming_proof": []} for k in POLICY_LABELS}

    # Stash auxiliary info for the optimal portfolio.
    n_opts: list[int] = []                  # welfare-optimal portfolio size from grid search
    n_stars_at_opt: list[float] = []        # gaming-proof threshold n* evaluated at optimal (sigma, delta)

    for env in envs:
        # 1. det_low
        a, u = eval_single(env, 0.3, 0.2)
        results["det_low"]["alpha"].append(a)
        results["det_low"]["payoff"].append(u)
        results["det_low"]["gaming_proof"].append(a == 0.0)

        # 2. det_high
        a, u = eval_single(env, 0.8, 0.8)
        results["det_high"]["alpha"].append(a)
        results["det_high"]["payoff"].append(u)
        results["det_high"]["gaming_proof"].append(a == 0.0)

        # 3. uniform_pf5
        a, u = eval_portfolio(env, 0.5, 0.5, n=5)
        results["uniform_pf5"]["alpha"].append(a)
        results["uniform_pf5"]["payoff"].append(u)
        results["uniform_pf5"]["gaming_proof"].append(a == 0.0)

        # 4. payoff_matched
        s, d = payoff_matched_metric(env)
        a, u = eval_single(env, s, d)
        results["payoff_match"]["alpha"].append(a)
        results["payoff_match"]["payoff"].append(u)
        results["payoff_match"]["gaming_proof"].append(a == 0.0)

        # 5. opt_portfolio
        n_opt, s, d, a, u = optimal_portfolio(env)
        results["opt_portfolio"]["alpha"].append(a)
        results["opt_portfolio"]["payoff"].append(u)
        results["opt_portfolio"]["gaming_proof"].append(a == 0.0)
        n_opts.append(n_opt)
        # Gaming-proof threshold n* at the welfare-optimal (sigma, delta), per Proposition 7.
        n_star = critical_portfolio_size(env.B, env.c, s, d, env.F)
        n_stars_at_opt.append(float(n_star))

        # 6. oracle (knows dev type is rational -> same policy class as 5 here).
        results["oracle"]["alpha"].append(a)
        results["oracle"]["payoff"].append(u)
        results["oracle"]["gaming_proof"].append(a == 0.0)

    summary = {}
    for key, vals in results.items():
        payoffs = np.array(vals["payoff"])
        alphas = np.array(vals["alpha"])
        gp = np.array(vals["gaming_proof"])
        summary[key] = {
            "mean": float(np.mean(payoffs)),
            "median": float(np.median(payoffs)),
            "p10": float(np.percentile(payoffs, 10)),
            "p90": float(np.percentile(payoffs, 90)),
            "gaming_proof_frac": float(np.mean(gp)),
            "mean_alpha": float(np.mean(alphas)),
        }
    summary["_mean_n_opt"] = float(np.mean(n_opts))
    summary["_mean_n_star"] = float(np.mean(n_opts))  # backwards-compat alias for the table caption
    summary["_payoffs"] = {k: np.array(v["payoff"]) for k, v in results.items()}
    summary["_n_opts"] = np.array(n_opts, dtype=float)
    summary["_n_stars_at_opt"] = np.array(n_stars_at_opt, dtype=float)
    return summary


def plot_cdfs(summary: dict) -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = plt.get_cmap("tab10").colors
    for i, (key, label) in enumerate(POLICY_LABELS.items()):
        payoffs = np.sort(summary["_payoffs"][key])
        cdf = np.arange(1, len(payoffs) + 1) / len(payoffs)
        ax.plot(payoffs, cdf, label=label, color=colors[i], linewidth=1.8)
    ax.set_xlabel(r"Regulator payoff $u_R$")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(rf"Regulator-payoff CDFs across {N_ENV:,} sampled environments")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.92)
    fig.tight_layout()
    out_path = OUT_FIG / "baseline_horserace.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_portfolio_size_distribution(summary: dict) -> None:
    """Joint distribution of welfare-optimal n_opt vs gaming-proof threshold n* across envs.

    Each environment contributes a point (n*, n_opt). Points on or above the y=x line
    correspond to gaming-proof environments (the welfare-optimal portfolio is at least
    as large as the gaming-proof threshold); points below are interior-regime
    environments where the regulator payoff-optimally tolerates strictly positive gaming
    despite a larger n* being available.

    Environments where n* = infinity (delta*F = 0 with B*sigma < c, so no finite portfolio
    of identical metrics deters) are summarised in the upper-x annotation rather than
    plotted as points off the chart.
    """
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    n_opts = np.asarray(summary["_n_opts"])
    n_stars = np.asarray(summary["_n_stars_at_opt"])

    finite_mask = np.isfinite(n_stars)
    n_inf = int(np.sum(~finite_mask))
    n_opts_f = n_opts[finite_mask]
    n_stars_f = n_stars[finite_mask]

    n_max = max(20, int(np.percentile(n_stars_f, 99)))
    clip_mask = n_stars_f <= n_max
    n_clipped = int(np.sum(~clip_mask))

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    above = n_opts_f >= n_stars_f
    ax.scatter(n_stars_f[above & clip_mask], n_opts_f[above & clip_mask],
               s=14, alpha=0.45, color="C0",
               label=rf"Gaming-proof ($n^{{\mathrm{{opt}}}} \geq n^*$): {int(above.sum())} envs")
    ax.scatter(n_stars_f[~above & clip_mask], n_opts_f[~above & clip_mask],
               s=14, alpha=0.55, color="C3",
               label=rf"Interior regime ($n^{{\mathrm{{opt}}}} < n^*$): {int((~above).sum())} envs")
    diag = np.linspace(1, n_max, 200)
    ax.plot(diag, diag, color="black", linestyle="--", linewidth=1.2,
            label=r"$n^{\mathrm{opt}} = n^*$")
    ax.axhline(np.mean(n_opts_f), color="C0", linestyle=":", linewidth=1.0,
               label=rf"Mean $n^{{\mathrm{{opt}}}} = {np.mean(n_opts_f):.2f}$")

    ax.set_xlim(0.5, n_max + 0.5)
    ax.set_ylim(0.5, max(20, int(n_opts_f.max())) + 0.5)
    ax.set_xlabel(r"Gaming-proof threshold $n^*$ at $(\sigma^{\mathrm{opt}}, \delta^{\mathrm{opt}})$")
    ax.set_ylabel(r"Welfare-optimal portfolio size $n^{\mathrm{opt}}$")
    ax.set_title(rf"Payoff-optimal vs gaming-proof portfolio size across {N_ENV:,} environments")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.92)

    annot = []
    n_at_ceiling = int(np.sum(n_opts_f == 20))
    if n_at_ceiling:
        annot.append(rf"{n_at_ceiling} env(s) at grid ceiling $n^{{\mathrm{{opt}}}} = 20$")
    if n_clipped:
        annot.append(rf"{n_clipped} env(s) with $n^* > {n_max}$ clipped from view")
    if n_inf:
        annot.append(rf"{n_inf} env(s) with $\delta F = 0$ and $B\sigma < c$ have $n^* = \infty$ (omitted)")
    if annot:
        ax.text(0.99, 0.02, "\n".join(annot), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.7"))

    fig.tight_layout()
    out_path = OUT_FIG / "portfolio_size_distribution.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(f"  n_opt summary: mean={np.mean(n_opts_f):.2f}, median={np.median(n_opts_f):.0f}, "
          f"p10={np.percentile(n_opts_f,10):.0f}, p90={np.percentile(n_opts_f,90):.0f}")
    print(f"  Gaming-proof envs: {int(above.sum())} / {len(n_opts_f)}; "
          f"infinite-n* envs: {n_inf}")


def write_table(summary: dict) -> None:
    OUT_TBL.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Baseline horse-race across 1{,}000 sampled environments. "
                 r"Mean, median, and 10th/90th percentile of regulator payoff $u_R$; "
                 r"fraction of environments for which each policy is gaming-proof ($\alpha^*_{\mathrm{eq}}=0$). "
                 r"The optimal robust portfolio is the grid-search optimum over "
                 r"$(n,\sigma,\delta) \in \{1,\dots,20\}\times[0,1]^2$; under rational developers it coincides with the oracle. "
                 rf"Mean $n^*_{{\text{{opt}}}} = {summary['_mean_n_star']:.2f}$.}}")
    lines.append(r"\label{tab:baseline_horserace}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Policy & Mean & Median & 10\% & 90\% & Gaming-proof \% \\")
    lines.append(r"\midrule")
    for key, label in POLICY_LABELS.items():
        s = summary[key]
        lines.append(f"{label} & {s['mean']:.3f} & {s['median']:.3f} & {s['p10']:.3f} & {s['p90']:.3f} & {100*s['gaming_proof_frac']:.1f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path = OUT_TBL / "baseline_horserace.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def main() -> None:
    summary = run()
    plot_cdfs(summary)
    plot_portfolio_size_distribution(summary)
    write_table(summary)


if __name__ == "__main__":
    main()
