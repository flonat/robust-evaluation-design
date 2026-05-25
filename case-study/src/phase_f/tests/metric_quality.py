"""F.3.3 Metric-quality test: σ across the model panel + safety illusion.

Two predictions from §3 of the manuscript:
1. Safety illusion I = (1-σ)α: higher-σ metrics reduce the illusion at fixed α.
2. σ is a property of the metric, not the developer — it should be stable
   across the panel (because all models score on the same MMLU metric).

We test (2) by comparing σ_fitted across models in the panel. Deviations
across models indicate that either σ varies with the evaluator (i.e., the
metric induces different leakage for different model families) OR that our
identification has noise that needs more strategies / better detectors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class MetricQualityResult:
    """Cross-model summary of σ + safety illusion."""
    n_models: int
    model_nicks: tuple[str, ...]
    sigmas: np.ndarray                  # σ_fitted per model
    sigmas_se: np.ndarray               # standard errors
    sigma_mean: float
    sigma_sd: float                     # cross-model SD (= heterogeneity)
    # Safety illusion per model at α=0.5 reference (mid-strategy)
    illusion_at_alpha_half: dict[str, float]
    # Per-family aggregation (e.g., "qwen" → mean σ over Qwen-3-8B/14B/32B)
    sigma_by_family: dict[str, float]
    notes: list[str] = field(default_factory=list)


def metric_quality_test(
    sigma_fitted_per_model: dict[str, float],
    sigma_se_per_model: dict[str, float],
    model_families: dict[str, str],
    *,
    alpha_reference: float = 0.5,
) -> MetricQualityResult:
    """Aggregate σ across the panel and test stability.

    `model_families`: maps model_nick → family ("phi" / "llama" / "qwen" / "mistral" / "gemma").
    """
    models = sorted(sigma_fitted_per_model.keys())
    sigmas = np.array([sigma_fitted_per_model[m] for m in models], dtype=float)
    ses = np.array([sigma_se_per_model.get(m, float("nan")) for m in models], dtype=float)

    sigma_mean = float(np.nanmean(sigmas))
    sigma_sd = float(np.nanstd(sigmas, ddof=1)) if np.sum(np.isfinite(sigmas)) > 1 else 0.0

    illusion = {m: (1.0 - sigma_fitted_per_model[m]) * alpha_reference for m in models}

    # Family aggregation
    family_to_sigmas: dict[str, list[float]] = {}
    for m, fam in model_families.items():
        if m in sigma_fitted_per_model and np.isfinite(sigma_fitted_per_model[m]):
            family_to_sigmas.setdefault(fam, []).append(sigma_fitted_per_model[m])
    sigma_by_family = {fam: float(np.mean(vs)) for fam, vs in family_to_sigmas.items()}

    notes: list[str] = []
    if sigma_sd > 0.15:
        notes.append(
            f"σ heterogeneity across models is large (SD={sigma_sd:.3f}); the metric does NOT "
            "induce a single σ — suggests model-dependent contamination response, not pure metric leakage. "
            "This either weakens the 'σ is a property of the metric' interpretation OR points to detector mis-calibration."
        )
    if sigma_sd < 0.05:
        notes.append(
            f"σ heterogeneity is small (SD={sigma_sd:.3f}); strong evidence that σ is a metric-level "
            "primitive in our panel — consistent with the paper §3 interpretation."
        )

    # Family-tier deviations
    if "qwen" in sigma_by_family and len(family_to_sigmas.get("qwen", [])) >= 2:
        notes.append(
            f"Intra-Qwen σ stability (3-point series 8B/14B/32B): "
            f"σ_qwen = {sigma_by_family['qwen']:.3f}. If individual Qwen sizes "
            "differ by < SE, this is the cleanest within-family stability test."
        )

    return MetricQualityResult(
        n_models=len(models),
        model_nicks=tuple(models),
        sigmas=sigmas,
        sigmas_se=ses,
        sigma_mean=sigma_mean,
        sigma_sd=sigma_sd,
        illusion_at_alpha_half=illusion,
        sigma_by_family=sigma_by_family,
        notes=notes,
    )


def summarise_metric_quality(result: MetricQualityResult) -> str:
    """Pretty-printed cross-model summary."""
    lines = [
        f"=== Metric quality (σ across panel) ===",
        f"  n_models = {result.n_models}",
        f"  σ_mean   = {result.sigma_mean:.3f}",
        f"  σ_sd     = {result.sigma_sd:.3f}  (cross-model heterogeneity)",
        f"",
        f"{'model':14} {'σ_fitted':>9} {'SE':>7} {'illusion(α=0.5)':>17}",
        "-" * 55,
    ]
    for i, m in enumerate(result.model_nicks):
        lines.append(
            f"{m:14} {result.sigmas[i]:>9.3f} {result.sigmas_se[i]:>7.3f} "
            f"{result.illusion_at_alpha_half[m]:>17.3f}"
        )
    lines.append("")
    lines.append(f"{'family':14} {'mean σ':>9}")
    lines.append("-" * 26)
    for fam, sig in sorted(result.sigma_by_family.items()):
        lines.append(f"{fam:14} {sig:>9.3f}")
    return "\n".join(lines)


def write_metric_quality_csv(result: MetricQualityResult, path: Path) -> None:
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_nick", "sigma_fitted", "sigma_se", "illusion_at_alpha_0.5"])
        for i, m in enumerate(result.model_nicks):
            w.writerow([
                m,
                f"{result.sigmas[i]:.4f}",
                f"{result.sigmas_se[i]:.4f}",
                f"{result.illusion_at_alpha_half[m]:.4f}",
            ])
