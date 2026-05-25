"""F.3.4 Portfolio test: Prop 16 n* threshold + aggregate detection across benchmarks.

Two angles in our experimental setup:

1. **Predicted n* per model** (Prop 16 of the manuscript):
       n* = ⌈(c - Bσ) / (δF)⌉  (under additive independent enforcement)
   Given fitted σ, observed δ, and a candidate F level, this is the number
   of independent benchmarks the regulator would need to deter voluntary
   gaming. We compute n* across (σ, δ, F) grid and report where current
   MMLU enforcement sits (single benchmark, F ≈ 0).

2. **Empirical portfolio aggregation** across our 3 benchmarks:
       Take per-(model, strategy) detector signal on
       {mmlu-public, mmlu-cf-hidden, mmlu-paraphrased}. Aggregate to n=1,
       n=2, n=3 portfolios and report the implied α̂_aggregate. Test
       whether α̂ drops as n grows, qualitatively matching the prediction.

The empirical test (2) is limited by us only having 3 benchmark types. For
the 8-model panel and 7 strategies = 56 (model, strategy) cells, each cell
gives one observation; with n ∈ {1, 2, 3} we have a 3-point curve per cell.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class PortfolioTestResult:
    """One (model, strategy) — predicted n* + observed α̂ curve over n."""
    model_nick: str
    strategy_nick: str
    sigma_fitted: float
    delta: float
    # Predicted n* under canonical (B, c, F) values
    n_star_at_F: dict[float, float]      # F → n*
    n_star_central: int                  # at average observed F
    # Empirical aggregation: per-n α̂
    alpha_by_n: dict[int, float]         # portfolio size → aggregate α̂
    notes: list[str] = field(default_factory=list)


# Canonical primitives (per §7 of the manuscript)
B_DEFAULT = 1.0
C_DEFAULT = 1.0


def n_star(sigma: float, delta: float, F: float, *, B: float = B_DEFAULT, c: float = C_DEFAULT) -> float:
    """Predicted gaming-proof portfolio size: n* = ⌈(c - Bσ)/(δF)⌉.

    Returns +inf if δF == 0 (no enforcement → no finite gaming-proof portfolio).
    Returns 1 (or below) if Bσ ≥ c (single metric already gaming-proof).
    """
    if delta * F <= 0:
        if B * sigma >= c:
            return 1.0
        return float("inf")
    if B * sigma >= c:
        return 1.0
    return float(np.ceil((c - B * sigma) / (delta * F)))


def portfolio_test(
    model_nick: str,
    strategy_nick: str,
    sigma_fitted: float,
    delta_observed: float,
    alpha_per_benchmark: dict[str, float],
    *,
    f_grid: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 5.0),
) -> PortfolioTestResult:
    """One (model, strategy) portfolio test.

    `alpha_per_benchmark`: detector signal under each benchmark
    (e.g., {'mmlu-public': 0.45, 'mmlu-cf-hidden': 0.12, 'mmlu-paraphrased': 0.30}).
    """
    notes: list[str] = []

    # Predicted n* across F grid
    n_star_at_F = {f: n_star(sigma_fitted, delta_observed, f) for f in f_grid}
    n_central = n_star(sigma_fitted, delta_observed, 1.0)
    n_central_int = int(min(n_central, 1e9)) if np.isfinite(n_central) else -1
    if not np.isfinite(n_central):
        notes.append(f"n* infinite at F=1.0c: model predicts no finite portfolio deters gaming under "
                     f"current (σ={sigma_fitted:.3f}, δ={delta_observed:.3f}, F=c)")

    # Empirical aggregation
    benchmarks = list(alpha_per_benchmark.keys())
    alpha_by_n: dict[int, float] = {}
    for n in (1, 2, 3):
        if n > len(benchmarks):
            break
        # All combinations of size n; aggregate signal = max α̂ across the portfolio
        # (under independent enforcement, ANY detector firing triggers the fine)
        max_alphas = []
        for combo in combinations(benchmarks, n):
            max_alphas.append(max(alpha_per_benchmark[b] for b in combo))
        # Average over combinations (so the n=2 estimate is symmetric over choice of pair)
        alpha_by_n[n] = float(np.mean(max_alphas))

    # Diagnostic: does α̂ drop with n?
    if 1 in alpha_by_n and 3 in alpha_by_n:
        if alpha_by_n[3] >= alpha_by_n[1]:
            notes.append(
                f"Empirical α̂ does NOT decrease with portfolio size: "
                f"n=1→{alpha_by_n[1]:.3f}, n=3→{alpha_by_n[3]:.3f}. "
                "Possible: per-benchmark signals are weakly correlated, OR detector class "
                "doesn't aggregate cleanly across benchmarks (e.g., behavioral_paraphrase "
                "is computed only on the paraphrased benchmark)."
            )

    return PortfolioTestResult(
        model_nick=model_nick,
        strategy_nick=strategy_nick,
        sigma_fitted=sigma_fitted,
        delta=delta_observed,
        n_star_at_F=n_star_at_F,
        n_star_central=n_central_int,
        alpha_by_n=alpha_by_n,
        notes=notes,
    )


def summarise_portfolio(results: Iterable[PortfolioTestResult]) -> str:
    lines = [
        f"{'model':14} {'strategy':22} {'σ':>6} {'δ':>6} {'n*(F=1)':>9} "
        f"{'α̂(n=1)':>9} {'α̂(n=2)':>9} {'α̂(n=3)':>9}",
        "-" * 96,
    ]
    for r in results:
        nc = f"{r.n_star_central}" if r.n_star_central >= 0 else "∞"
        a1 = r.alpha_by_n.get(1, float("nan"))
        a2 = r.alpha_by_n.get(2, float("nan"))
        a3 = r.alpha_by_n.get(3, float("nan"))
        lines.append(
            f"{r.model_nick:14} {r.strategy_nick:22} "
            f"{r.sigma_fitted:>6.2f} {r.delta:>6.2f} {nc:>9} "
            f"{a1:>9.3f} {a2:>9.3f} {a3:>9.3f}"
        )
    return "\n".join(lines)


def write_portfolio_csv(results: list[PortfolioTestResult], path: Path) -> None:
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model_nick", "strategy_nick", "sigma", "delta",
            "n_star_F_0.1", "n_star_F_0.5", "n_star_F_1.0", "n_star_F_2.0", "n_star_F_5.0",
            "alpha_n1", "alpha_n2", "alpha_n3",
        ])
        for r in results:
            w.writerow([
                r.model_nick, r.strategy_nick,
                f"{r.sigma_fitted:.4f}", f"{r.delta:.4f}",
                *(f"{r.n_star_at_F.get(f, float('inf')):.2f}" for f in (0.1, 0.5, 1.0, 2.0, 5.0)),
                *(f"{r.alpha_by_n.get(n, float('nan')):.4f}" for n in (1, 2, 3)),
            ])
