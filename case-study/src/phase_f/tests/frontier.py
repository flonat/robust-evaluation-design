"""Frontier test: does the model identity gap = (1-σ)α hold across strategies?

The central paper claim (Cor 3.2, §3) is that the cleaned-benchmark gap
M - S equals (1-σ)α, with σ a property of the metric (not the strategy)
and α the developer's gaming intensity. In the controlled case study, for
each model we have observations:

    (gap_s, α̂_s)  for s in {baseline, contam-light, ..., contam-heavy}

If the model identity holds:
1. Plotting gap_s against α̂_s should be approximately linear through the origin
2. The slope coefficient is (1-σ_model), so σ_model = 1 - slope
3. R² across strategies measures how well the model identity captures the data
4. Per-strategy residuals flag strategies where the identity fails (e.g.,
   strategies dominated by involuntary contamination rather than voluntary gaming)

This is the cleanest empirical test our setup supports: the regulatory
parameter F is implicit (developers choose strategies, not F levels), but
the gap-α relationship is a direct consequence of the model that we can
verify on the observed data.

Predictions and what we expect to see in the §8 case study:
- Baseline strategy: α̂ ≈ 0 (or small from involuntary contamination), gap ≈ 0
- Light contamination: small α̂, proportionally small gap
- Heavy contamination: large α̂, large gap
- Decontamination effort: α̂ ≈ 0, gap close to baseline's residual gap
- Paraphrase contamination: α̂ > 0 but TS-Guessing-style detectors may miss
  it (lower α̂ from THAT detector, but gap still grows) — this is the
  detector-sensitivity story

The fitted σ_model should be consistent with the F.2 joint-identification σ
(the cross-source anchor in §7.1 of the manuscript), and stable across the
strategies for a given model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class FrontierTestResult:
    """One model's gap-α fit across strategies."""
    model_nick: str
    n_strategies: int
    strategy_nicks: tuple[str, ...]
    alphas: np.ndarray            # per-strategy α̂ (shape n,)
    gaps: np.ndarray              # per-strategy gap (shape n,)
    detector_used: str            # which detector provided α̂ (or "strategy_rate" fallback)
    sigma_fitted: float           # via OLS (1-σ) = slope
    sigma_fitted_se: float        # standard error
    r_squared: float              # OLS fit quality
    residuals: np.ndarray         # per-strategy residuals
    notes: list[str] = field(default_factory=list)

    @property
    def slope(self) -> float:
        return 1.0 - self.sigma_fitted

    def consistent_with(self, sigma_reference: float, tol: float = 0.15) -> bool:
        """Is the fitted σ within `tol` of a reference value (e.g. paper §7.1 σ≈0.6)?"""
        return abs(self.sigma_fitted - sigma_reference) <= tol


def _fit_through_origin(alpha: np.ndarray, gap: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """OLS fit gap = slope * alpha (no intercept). Returns slope, SE, R², residuals.

    Forcing the line through the origin is principled: at α=0 the model
    identity says gap=0. (Any non-zero intercept would be an additive
    contamination floor — interesting to study separately but not the
    quantity that pins down σ.)
    """
    if len(alpha) < 2:
        return float("nan"), float("nan"), float("nan"), np.array([])
    # OLS through origin: slope = sum(αg) / sum(α²)
    denom = float(np.dot(alpha, alpha))
    if denom <= 0:
        return float("nan"), float("nan"), float("nan"), gap.copy()
    slope = float(np.dot(alpha, gap) / denom)
    residuals = gap - slope * alpha

    # Through-origin R²: 1 - SS_res / SS_tot, with SS_tot = sum(g²) (NOT mean-centred,
    # because the no-intercept model's null is "g = 0", not "g = mean(g)").
    ss_res = float(np.dot(residuals, residuals))
    ss_tot = float(np.dot(gap, gap))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # SE of slope under homoskedastic errors
    n = len(alpha)
    if n > 1:
        sigma_sq = ss_res / max(n - 1, 1)
        se = float(np.sqrt(sigma_sq / denom))
    else:
        se = float("nan")

    return slope, se, r2, residuals


def run_frontier_test(
    model_nick: str,
    per_strategy: list[dict],
    *,
    detector_nick: str = "ts_guessing",
    use_strategy_rate_fallback: bool = True,
) -> FrontierTestResult:
    """Run the frontier test for one model.

    `per_strategy` is a list of dicts with keys:
      strategy_nick, gap, alpha_per_detector (dict), contamination_rate

    The function picks α̂ from `alpha_per_detector[detector_nick]`. If that's
    NaN or missing AND `use_strategy_rate_fallback=True`, uses contamination_rate.
    """
    alphas: list[float] = []
    gaps: list[float] = []
    strategies: list[str] = []
    notes: list[str] = []
    used_fallback = False

    for entry in per_strategy:
        a = entry.get("alpha_per_detector", {}).get(detector_nick, float("nan"))
        if not np.isfinite(a):
            if use_strategy_rate_fallback:
                a = entry.get("contamination_rate", 0.0)
                used_fallback = True
            else:
                notes.append(f"strategy={entry['strategy_nick']}: α̂ NaN, skipped")
                continue
        alphas.append(float(a))
        gaps.append(float(entry["gap"]))
        strategies.append(entry["strategy_nick"])

    if used_fallback:
        notes.append(f"Some α̂ values missing for detector={detector_nick}; "
                     "fell back to strategy contamination_rate as α̂ proxy")

    alphas_arr = np.array(alphas, dtype=float)
    gaps_arr = np.array(gaps, dtype=float)
    slope, se, r2, residuals = _fit_through_origin(alphas_arr, gaps_arr)
    sigma_fitted = 1.0 - slope
    sigma_fitted_se = se

    if not np.isnan(r2) and r2 < 0.5:
        notes.append(f"Low R²={r2:.3f}: gap-α relationship is weak across "
                     "strategies. Possible: involuntary contamination dominates, "
                     "detector mis-calibrated, or σ varies across strategies.")
    if not np.isnan(sigma_fitted) and (sigma_fitted < 0 or sigma_fitted > 1):
        notes.append(f"Fitted σ={sigma_fitted:.3f} outside [0,1]: indicates "
                     "model identity does not hold under this detector or that "
                     "strategy injection is mis-targeted")

    return FrontierTestResult(
        model_nick=model_nick,
        n_strategies=len(strategies),
        strategy_nicks=tuple(strategies),
        alphas=alphas_arr,
        gaps=gaps_arr,
        detector_used=(detector_nick if not used_fallback else f"{detector_nick}+strategy_rate_fallback"),
        sigma_fitted=sigma_fitted,
        sigma_fitted_se=sigma_fitted_se,
        r_squared=r2,
        residuals=residuals,
        notes=notes,
    )


def summarise_frontier_tests(results: Iterable[FrontierTestResult]) -> str:
    """Pretty-printed summary table across models."""
    lines = [
        f"{'model':14} {'n':>3} {'σ_fitted':>10} {'σ_SE':>8} {'R²':>8} {'detector':30}",
        "-" * 80,
    ]
    for r in results:
        lines.append(
            f"{r.model_nick:14} {r.n_strategies:>3} "
            f"{r.sigma_fitted:>10.3f} {r.sigma_fitted_se:>8.3f} "
            f"{r.r_squared:>8.3f} {r.detector_used:30}"
        )
    return "\n".join(lines)


def write_frontier_csv(results: list[FrontierTestResult], path: Path) -> None:
    """Per-strategy results CSV (across all models)."""
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model_nick", "strategy_nick", "alpha_hat", "gap_observed",
            "gap_predicted", "residual",
            "sigma_fitted", "sigma_se", "r_squared", "detector",
        ])
        for r in results:
            for i, s in enumerate(r.strategy_nicks):
                pred = r.slope * r.alphas[i]
                w.writerow([
                    r.model_nick, s,
                    f"{r.alphas[i]:.4f}", f"{r.gaps[i]:.4f}",
                    f"{pred:.4f}", f"{r.residuals[i]:.4f}",
                    f"{r.sigma_fitted:.4f}", f"{r.sigma_fitted_se:.4f}",
                    f"{r.r_squared:.4f}", r.detector_used,
                ])
