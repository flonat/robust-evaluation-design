"""Matplotlib plots for the frontier test (gap vs α).

Per-model figures + a combined panel.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from phase_f.tests.frontier import FrontierTestResult


# Reference σ from the corrected manuscript calibration (§7.1, joint identification)
SIGMA_PAPER_REFERENCE = 0.60


def plot_frontier_per_model(
    result: FrontierTestResult,
    out_path: Path,
    *,
    sigma_reference: float = SIGMA_PAPER_REFERENCE,
) -> Path:
    """One figure: gap vs α scatter for one model + 2 reference lines."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # Scatter the observed (α̂, gap) points, label by strategy
    ax.scatter(result.alphas, result.gaps, s=60, color="C0", zorder=3, label="Observed")
    for i, s in enumerate(result.strategy_nicks):
        ax.annotate(s, (result.alphas[i], result.gaps[i]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")

    # Fitted line through origin
    alpha_max = float(np.nanmax(result.alphas)) if len(result.alphas) else 1.0
    alpha_max = max(alpha_max, 0.1)
    alpha_grid = np.linspace(0, alpha_max * 1.05, 50)
    if np.isfinite(result.sigma_fitted):
        ax.plot(alpha_grid, result.slope * alpha_grid,
                color="C0", linestyle="--",
                label=f"Fit: σ={result.sigma_fitted:.2f}±{result.sigma_fitted_se:.2f}")

    # Paper-reference line (joint-identification σ from §7.1)
    ax.plot(alpha_grid, (1 - sigma_reference) * alpha_grid,
            color="C3", linestyle=":", alpha=0.7,
            label=f"§7.1 reference: σ={sigma_reference:.2f}")

    ax.set_xlabel(r"Observed gaming intensity $\widehat{\alpha}^*$")
    ax.set_ylabel(r"Cleaned-benchmark gap $M - S$")
    title = f"Frontier test: {result.model_nick}"
    if result.n_strategies > 1:
        title += f"  (n={result.n_strategies} strategies, R²={result.r_squared:.2f})"
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=-0.05)
    ax.set_ylim(bottom=min(0.0, float(np.nanmin(result.gaps)) - 0.05))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_frontier_panel(
    results: list[FrontierTestResult],
    out_path: Path,
    *,
    sigma_reference: float = SIGMA_PAPER_REFERENCE,
    n_cols: int = 4,
) -> Path:
    """Combined panel figure: one subplot per model."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(results)
    if n == 0:
        raise ValueError("No results to plot")
    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)

    for idx, result in enumerate(results):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.scatter(result.alphas, result.gaps, s=40, color="C0", zorder=3)
        alpha_max = max(float(np.nanmax(result.alphas)) if len(result.alphas) else 1.0, 0.1)
        alpha_grid = np.linspace(0, alpha_max * 1.05, 50)
        if np.isfinite(result.sigma_fitted):
            ax.plot(alpha_grid, result.slope * alpha_grid, color="C0", linestyle="--",
                    label=f"σ={result.sigma_fitted:.2f}")
        ax.plot(alpha_grid, (1 - sigma_reference) * alpha_grid,
                color="C3", linestyle=":", alpha=0.7, label=f"§7.1: σ={sigma_reference:.2f}")
        ax.set_title(f"{result.model_nick} (R²={result.r_squared:.2f})", fontsize=10)
        ax.set_xlabel(r"$\widehat{\alpha}^*$")
        ax.set_ylabel(r"gap $M-S$")
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle("F.3.1 Frontier test: gap = (1-σ)α across developer strategies", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
