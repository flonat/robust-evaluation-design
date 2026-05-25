"""F.3.2 Fine-threshold test: F* = (c - Bσ) / δ.

For each (model, strategy) pair we have:
  - σ_fitted from the F.3.1 frontier test (or from F.2 joint identification)
  - δ from the detector signal (TS-Guessing class)
  - B, c are normalised primitives (paper convention: B=1.0, c=1.0 as scales)

Theorem 6 Part 2 prescribes F* such that B σ + δ F = c at the binding gaming-
proof frontier. The case study output is:

  - Per-model F* prediction (the regulator's required enforcement)
  - Per-strategy implied F (inverting the developer FOC given observed α̂)
  - Comparison to empirical F ≈ 0 (current MMLU enforcement per §7)

The test interpretation:
  - If observed strategies cluster around the implied F* line, the model is
    consistent with the data.
  - If implied F_observed differs substantially from F*_predicted (e.g.,
    F_observed < 0 systematically), strategies are gaming more than the
    cost-minimal best-response — points to either under-rational developer
    behavior or to additional factors (involuntary contamination, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


# Paper normalisations (per §7 of the manuscript)
B_DEFAULT = 1.0
C_DEFAULT = 1.0
GAMMA_DEFAULT = 0.5  # midpoint of the bracketed range [0, c] in §7


@dataclass
class FineThresholdResult:
    """Per-model: predicted F* + per-strategy implied F."""
    model_nick: str
    sigma_fitted: float                  # from F.3.1 frontier test
    # F* prediction (regulator's required enforcement to deter gaming)
    f_star_per_delta: dict[float, float]  # δ → F* = (c - Bσ)/δ, evaluated at canonical δ grid
    f_star_central: float                # F* at the average observed δ
    # Per-strategy: implied F under each strategy's observed α̂
    strategy_implied_f: dict[str, float]
    # Empirical comparison: current observed F (≈ 0 in MMLU)
    f_observed: float = 0.0
    notes: list[str] = field(default_factory=list)


def fine_threshold_test(
    model_nick: str,
    sigma_fitted: float,
    alpha_per_strategy: dict[str, float],
    delta_per_strategy: dict[str, float] | float,
    *,
    B: float = B_DEFAULT,
    c: float = C_DEFAULT,
    gamma: float = GAMMA_DEFAULT,
    f_observed: float = 0.0,
) -> FineThresholdResult:
    """Compute F* and per-strategy implied F.

    `delta_per_strategy`: either a single δ (e.g., the TS-Guessing rate from §7)
        or a dict mapping strategy nick → δ.
    """
    notes: list[str] = []

    # F* across a canonical δ grid (matches paper §7.2 table)
    delta_grid = [0.05, 0.10, 0.20, 0.30, 0.43]   # paper table values
    f_star_per_delta = {}
    for d in delta_grid:
        if d > 0:
            f_star_per_delta[d] = (c - B * sigma_fitted) / d

    # Central F* at the mean observed δ
    if isinstance(delta_per_strategy, dict):
        delta_values = [v for v in delta_per_strategy.values() if np.isfinite(v) and v > 0]
        if delta_values:
            delta_mean = float(np.mean(delta_values))
        else:
            delta_mean = float("nan")
    else:
        delta_mean = float(delta_per_strategy)

    if np.isfinite(delta_mean) and delta_mean > 0:
        f_star_central = (c - B * sigma_fitted) / delta_mean
    else:
        f_star_central = float("nan")
        notes.append("No valid δ observations; F* central undefined")

    # Per-strategy implied F (from developer FOC: α = (c - Bσ - δF) / (c+γ))
    # Solving for F: F = (c - Bσ - α(c+γ)) / δ
    strategy_implied_f: dict[str, float] = {}
    for strat, alpha in alpha_per_strategy.items():
        if isinstance(delta_per_strategy, dict):
            d = delta_per_strategy.get(strat, delta_mean)
        else:
            d = delta_per_strategy
        if not np.isfinite(d) or d <= 0:
            strategy_implied_f[strat] = float("nan")
            continue
        implied_f = (c - B * sigma_fitted - alpha * (c + gamma)) / d
        strategy_implied_f[strat] = float(implied_f)

    # Diagnostic notes
    negative_f_strategies = [s for s, f in strategy_implied_f.items() if f < 0]
    if negative_f_strategies:
        notes.append(
            f"Strategies with implied F < 0: {negative_f_strategies}. "
            "These strategies game MORE than is consistent with the LQ FOC "
            "(α greater than 1−Bσ/(c+γ)) — possible explanations: contamination "
            "via training-data leak isn't constrained by the developer FOC, "
            "or σ is mis-estimated."
        )

    if f_star_central > 5 * c:
        notes.append(
            f"F* central = {f_star_central:.2f}c is impractically large; "
            "the model predicts a very high deterrence requirement under the "
            "fitted σ. Possible: low δ × high (c - Bσ)."
        )

    return FineThresholdResult(
        model_nick=model_nick,
        sigma_fitted=sigma_fitted,
        f_star_per_delta=f_star_per_delta,
        f_star_central=f_star_central,
        strategy_implied_f=strategy_implied_f,
        f_observed=f_observed,
        notes=notes,
    )


def summarise_fine_threshold(results: Iterable[FineThresholdResult]) -> str:
    """Pretty-printed table across models."""
    lines = [
        f"{'model':14} {'σ_fit':>7} {'F*_central':>11} {'F*(δ=0.2)':>11} {'F*(δ=0.43)':>11}",
        "-" * 70,
    ]
    for r in results:
        f02 = r.f_star_per_delta.get(0.20, float("nan"))
        f043 = r.f_star_per_delta.get(0.43, float("nan"))
        lines.append(
            f"{r.model_nick:14} {r.sigma_fitted:>7.3f} "
            f"{r.f_star_central:>11.3f} {f02:>11.3f} {f043:>11.3f}"
        )
    return "\n".join(lines)


def write_fine_threshold_csv(results: list[FineThresholdResult], path: Path) -> None:
    """Per-strategy CSV (model × strategy → implied F + F*)."""
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model_nick", "strategy_nick", "sigma_fitted",
            "implied_F_observed", "F_star_central",
            "F_star_at_delta_0.20", "F_star_at_delta_0.43",
            "F_observed_empirical",
        ])
        for r in results:
            for strat, implied in r.strategy_implied_f.items():
                w.writerow([
                    r.model_nick, strat, f"{r.sigma_fitted:.4f}",
                    f"{implied:.4f}",
                    f"{r.f_star_central:.4f}" if np.isfinite(r.f_star_central) else "",
                    f"{r.f_star_per_delta.get(0.20, float('nan')):.4f}",
                    f"{r.f_star_per_delta.get(0.43, float('nan')):.4f}",
                    f"{r.f_observed:.4f}",
                ])
