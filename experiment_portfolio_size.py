"""Experiment: Portfolio additive-enforcement dominance.

Fixes parameters where a single metric is non-gaming-proof, then sweeps the
portfolio size n and records gaming intensity, regulator payoff, and safety.
Left panel: alpha_n* vs n with dashed n* critical size.
Right panel: u_R and S* vs n.

Writes paper-aij/paper/figures/portfolio_dominance.pdf.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from config import OUT_FIG, configure_matplotlib

configure_matplotlib()
from model import (Environment, critical_portfolio_size, portfolio_alpha_star,
                   regulator_optimum, regulator_payoff)

B = 1.0
C = 2.0
GAMMA = 1.0
K_SIGMA = 1.0
K_DELTA = 1.0
F = 0.5
N_VALUES = list(range(1, 26))


def main() -> None:
    env = Environment(B=B, c=C, gamma=GAMMA, k_sigma=K_SIGMA, k_delta=K_DELTA, F=F)
    base = regulator_optimum(env, sigma_grid=51, delta_grid=51)
    sigma_star = base["sigma"]
    delta_star = base["delta"]
    n_crit = critical_portfolio_size(B, C, sigma_star, delta_star, F)

    alphas = []
    payoffs = []
    safeties = []
    for n in N_VALUES:
        a = portfolio_alpha_star(B, C, GAMMA, sigma_star, delta_star, F, n)
        u = regulator_payoff(a, sigma_star, delta_star, K_SIGMA, K_DELTA, n=n)
        alphas.append(a)
        payoffs.append(u)
        safeties.append(1.0 - a)

    OUT_FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))

    ax = axes[0]
    ax.plot(N_VALUES, alphas, marker="o", color="C0", label=r"Equilibrium gaming $\alpha_n^*$")
    ax.axvline(n_crit, color="black", linestyle="--", alpha=0.6,
               label=rf"Critical size $n^* = {n_crit}$")
    ax.set_xlabel(r"Portfolio size $n$")
    ax.set_ylabel(r"Gaming intensity $\alpha_n^*$")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.92)

    ax = axes[1]
    ax.plot(N_VALUES, payoffs, marker="s", color="C1", label=r"Regulator payoff $u_R$")
    ax.plot(N_VALUES, safeties, marker="^", color="C2", label=r"Actual safety $S^*_{\mathrm{eq}}$")
    ax.axvline(n_crit, color="black", linestyle="--", alpha=0.6,
               label=rf"$n^* = {n_crit}$")
    ax.set_xlabel(r"Portfolio size $n$")
    ax.set_ylabel(r"Regulator payoff / actual safety")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.92)

    fig.suptitle(rf"Portfolio size effect at $\sigma^*={sigma_star:.2f}$, $\delta^*={delta_star:.2f}$")
    fig.tight_layout()
    out_path = OUT_FIG / "portfolio_dominance.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path} (n* = {n_crit}, sigma*={sigma_star:.3f}, delta*={delta_star:.3f})")


if __name__ == "__main__":
    main()
