"""Experiment: Validate the gaming-proof boundary B*sigma + delta*F >= c.

Sweeps (B, F) on a grid, solves the regulator's optimum for each pair,
and plots the equilibrium gaming intensity with the theoretical boundary
overlaid as a white contour.

Writes out/figures/gaming_proof_boundary.pdf.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model import Environment, regulator_optimum

C = 1.0
GAMMA = 0.5
K_SIGMA = 1.0
K_DELTA = 1.0

B_GRID = np.linspace(0.5, 5.0, 31)
F_GRID = np.linspace(0.1, 2.0, 31)

REPO_ROOT = Path(__file__).resolve().parent
OUT_FIG = REPO_ROOT / "out" / "figures"


def compute_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha_surface = np.zeros((len(F_GRID), len(B_GRID)))
    boundary = np.zeros_like(alpha_surface)
    for i, F in enumerate(F_GRID):
        for j, B in enumerate(B_GRID):
            env = Environment(B=B, c=C, gamma=GAMMA, k_sigma=K_SIGMA, k_delta=K_DELTA, F=F)
            opt = regulator_optimum(env, sigma_grid=41, delta_grid=41)
            alpha_surface[i, j] = opt["alpha"]
            boundary[i, j] = B * opt["sigma"] + F * opt["delta"] - C
    return alpha_surface, boundary, np.array([])


def plot(alpha_surface: np.ndarray, boundary: np.ndarray) -> None:
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
    )
    cs = ax.contour(B_GRID, F_GRID, boundary, levels=[0.0], colors="white", linewidths=2.0)
    ax.clabel(cs, inline=True, fmt={0.0: r"$B\sigma^* + F\delta^* = c$"}, fontsize=9)
    ax.set_xlabel(r"Benefit $B$")
    ax.set_ylabel(r"Fine $F$")
    ax.set_title(r"Equilibrium gaming $\alpha^*_{\mathrm{eq}}$ with theoretical boundary")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\alpha^*_{\mathrm{eq}}$")
    fig.tight_layout()
    out_path = OUT_FIG / "gaming_proof_boundary.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    alpha_surface, boundary, _ = compute_surface()
    plot(alpha_surface, boundary)


if __name__ == "__main__":
    main()
