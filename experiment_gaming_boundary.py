"""Experiment: Validate the gaming-proof boundary B*sigma + delta*F >= c.

Sweeps (B, F) on a fine grid, computes the regulator's optimal metric in
closed form (Theorem on Optimal Single Metric, Part 1) with KKT projection
for the box constraints sigma, delta in [0, 1] (Part 3), and plots the
resulting equilibrium gaming intensity.

The theoretical regime boundary B^2/k_sigma + F^2/k_delta = c(c+gamma)
(Corollary on When to Tolerate Gaming) is overlaid as a single smooth
analytic curve, not a noisy grid-derived contour.

Writes paper-aij/paper/figures/gaming_proof_boundary.pdf.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from config import OUT_FIG, configure_matplotlib

configure_matplotlib()
from model import alpha_star, regulator_payoff

C = 1.0
GAMMA = 0.5
K_SIGMA = 1.0
K_DELTA = 1.0

B_GRID = np.linspace(0.5, 5.0, 121)
F_GRID = np.linspace(0.1, 2.0, 121)


def constrained_optimum(B: float, F: float) -> tuple[float, float, float]:
    """Closed-form regulator optimum with box constraints sigma, delta in [0,1].

    Theorem on Optimal Single Metric, Parts 1+3: start from the unconstrained
    interior FOC, then apply KKT projection along whichever box constraints
    are active. With sigma and delta independent in u_R, the projection is
    simply componentwise clipping at the unconstrained FOC values.
    """
    sigma_unc = B / (K_SIGMA * (C + GAMMA))
    delta_unc = F / (K_DELTA * (C + GAMMA))
    sigma = float(np.clip(sigma_unc, 0.0, 1.0))
    delta = float(np.clip(delta_unc, 0.0, 1.0))
    alpha = alpha_star(B, C, GAMMA, sigma, delta, F)
    return sigma, delta, alpha


def compute_surface() -> np.ndarray:
    alpha_surface = np.zeros((len(F_GRID), len(B_GRID)))
    for i, F in enumerate(F_GRID):
        for j, B in enumerate(B_GRID):
            _, _, alpha = constrained_optimum(float(B), float(F))
            alpha_surface[i, j] = alpha
    return alpha_surface


def theoretical_boundary(B_vals: np.ndarray) -> np.ndarray:
    """Regime boundary F(B) such that B^2/k_sigma + F^2/k_delta = c(c+gamma).

    Below this curve the interior regime is optimal and gaming persists;
    above it the gaming-proof regime is optimal (subject to box-feasibility).
    Returns NaN where no real F satisfies the equation.
    """
    rhs = C * (C + GAMMA) - (B_vals ** 2) / K_SIGMA
    F_sq = rhs * K_DELTA
    F_boundary = np.where(F_sq >= 0, np.sqrt(np.maximum(F_sq, 0)), np.nan)
    return F_boundary


def plot(alpha_surface: np.ndarray) -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    im = ax.imshow(
        alpha_surface,
        origin="lower",
        extent=(B_GRID[0], B_GRID[-1], F_GRID[0], F_GRID[-1]),
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
    )

    # Single smooth analytic boundary curve.
    B_dense = np.linspace(B_GRID[0], B_GRID[-1], 401)
    F_dense = theoretical_boundary(B_dense)
    mask = (F_dense >= F_GRID[0]) & (F_dense <= F_GRID[-1])
    ax.plot(B_dense[mask], F_dense[mask], color="white", linewidth=2.0,
            label=r"Regime boundary $B^2/k_\sigma + F^2/k_\delta = c(c+\gamma)$")

    ax.set_xlim(B_GRID[0], B_GRID[-1])
    ax.set_ylim(F_GRID[0], F_GRID[-1])
    ax.set_xlabel(r"Benefit $B$")
    ax.set_ylabel(r"Fine $F$")
    ax.set_title(r"Equilibrium gaming intensity $\alpha^*_{\mathrm{eq}}$ over the $(B, F)$ plane")
    ax.legend(loc="upper right", framealpha=0.92)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\alpha^*_{\mathrm{eq}}$")

    fig.tight_layout()
    out_path = OUT_FIG / "gaming_proof_boundary.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    alpha_surface = compute_surface()
    plot(alpha_surface)


if __name__ == "__main__":
    main()
