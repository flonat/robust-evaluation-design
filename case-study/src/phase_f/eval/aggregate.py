"""Aggregate per-(model, strategy, benchmark) scores into a σ table.

Reads:
  - results/scores/<model>__<strategy>__<benchmark>.csv (one per fine-tune × benchmark)
  - results/detectors/<model>__<strategy>__<detector>.csv (optional, one per detector run)

Produces:
  - results/identification.csv (one row per (model, strategy) — the σ table)
  - results/identification.json (full structured records)
  - A pretty-printed table to stdout

The σ table is the **headline output of F.2**: it's the empirical version of
the calibration table in §7.1 of the paper, but with primitives measured from
controlled experiments rather than from external sources.

If detector signals are absent (e.g., F.2 smoke pre-detector run), σ is
estimated from the known per-strategy contamination rate as a placeholder
α̂ (so σ = 1 − gap / contamination_rate). Real per-detector α̂ replaces
this fallback once F.2-detectors lands.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from phase_f.config import (
    DETECTORS,
    RESULTS_DETECTORS,
    RESULTS_SCORES,
    STRATEGIES,
    StrategySpec,
)
from phase_f.eval.identify import IdentificationResult, identify_sigma, summarise_table


# results/scores filename pattern: <model>__<strategy>__<benchmark>.csv
SCORE_RE = re.compile(r"^(?P<model>[a-z0-9.-]+)__(?P<strategy>[a-z0-9-]+)__(?P<benchmark>[a-z0-9-]+)\.csv$")


def _read_accuracy_from_csv(path: Path) -> tuple[float, int]:
    """Return (accuracy, n_items) from a score CSV (header: ..., correct, ...)."""
    n = 0
    correct = 0
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            correct += int(row["correct"])
    if n == 0:
        return 0.0, 0
    return correct / n, n


def _strategy_by_nick(nick: str) -> StrategySpec | None:
    for s in STRATEGIES:
        if s.nick == nick:
            return s
    return None


def collect_scores(scores_dir: Path | None = None) -> dict[tuple[str, str], dict[str, tuple[float, int]]]:
    """Scan score CSVs and group by (model_nick, strategy_nick) → benchmark → (acc, n)."""
    scores_dir = scores_dir or RESULTS_SCORES
    out: dict[tuple[str, str], dict[str, tuple[float, int]]] = {}
    if not scores_dir.exists():
        return out
    for path in sorted(scores_dir.glob("*.csv")):
        m = SCORE_RE.match(path.name)
        if not m:
            continue
        model = m.group("model")
        strategy = m.group("strategy")
        benchmark = m.group("benchmark")
        acc, n = _read_accuracy_from_csv(path)
        out.setdefault((model, strategy), {})[benchmark] = (acc, n)
    return out


def collect_detector_alpha(
    model_nick: str,
    strategy_nick: str,
    detectors_dir: Path | None = None,
) -> dict[str, float]:
    """Read per-detector α̂ for a (model, strategy) pair.

    Detector CSV format: detector_nick,alpha_hat,n_items (one row per detector).
    """
    detectors_dir = detectors_dir or RESULTS_DETECTORS
    path = detectors_dir / f"{model_nick}__{strategy_nick}__alphas.csv"
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["detector_nick"]] = float(row["alpha_hat"])
    return out


def fallback_alpha_from_strategy(strategy_nick: str) -> dict[str, float]:
    """Placeholder α̂ from the strategy's known contamination rate.

    Used when per-detector signals are absent (e.g., F.2 smoke). Returns the
    same α for every detector — `IdentificationResult.sigma_per_detector` then
    collapses to one value per detector all equal to σ = 1 − gap/α.
    """
    s = _strategy_by_nick(strategy_nick)
    if s is None:
        return {d: 0.0 for d in DETECTORS}
    # Use the contamination rate as a rough proxy for α̂. For baseline (rate=0),
    # the identify_sigma() function returns NaN with the involuntary-contam note.
    return {d: float(s.contamination_rate) for d in DETECTORS}


def build_identification_table(
    scores: dict[tuple[str, str], dict[str, tuple[float, int]]],
    *,
    use_detector_signals: bool = True,
) -> list[IdentificationResult]:
    """One IdentificationResult per (model, strategy) with all 3 benchmarks scored."""
    results: list[IdentificationResult] = []
    for (model, strategy), benches in sorted(scores.items()):
        pub = benches.get("mmlu-public")
        hid = benches.get("mmlu-cf-hidden")
        para = benches.get("mmlu-paraphrased")
        if pub is None or hid is None:
            # Need at least public + hidden to compute σ; paraphrased is optional
            continue

        # Resolve α̂: detector signals first, fall back to strategy contamination rate
        if use_detector_signals:
            alphas = collect_detector_alpha(model, strategy)
            if not alphas:
                alphas = fallback_alpha_from_strategy(strategy)
        else:
            alphas = fallback_alpha_from_strategy(strategy)

        r = identify_sigma(
            public_accuracy=pub[0],
            hidden_accuracy=hid[0],
            alpha_hat_per_detector=alphas,
            model_nick=model,
            strategy_nick=strategy,
            paraphrased_accuracy=(para[0] if para else None),
            n_items=pub[1],
        )
        results.append(r)
    return results


def write_identification_outputs(
    results: list[IdentificationResult],
    *,
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write CSV + JSON outputs. Returns (csv_path, json_path)."""
    out_dir = out_dir or (RESULTS_SCORES.parent / "identification")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "identification.csv"
    json_path = out_dir / "identification.json"

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        header = [
            "model_nick", "strategy_nick",
            "public_acc", "hidden_acc", "paraphrased_acc",
            "gap",
            *(f"alpha__{d}" for d in DETECTORS),
            *(f"sigma__{d}" for d in DETECTORS),
            "sigma_central", "sigma_lower", "sigma_upper",
            "n_items", "notes",
        ]
        w.writerow(header)
        for r in results:
            row = [
                r.model_nick, r.strategy_nick,
                f"{r.public_accuracy:.4f}", f"{r.hidden_accuracy:.4f}",
                f"{r.paraphrased_accuracy:.4f}" if r.paraphrased_accuracy is not None else "",
                f"{r.gap:.4f}",
                *(f"{r.alpha_hat_per_detector.get(d, float('nan')):.4f}" for d in DETECTORS),
                *(f"{r.sigma_per_detector.get(d, float('nan')):.4f}" for d in DETECTORS),
                f"{r.sigma_central:.4f}" if not np.isnan(r.sigma_central) else "",
                f"{r.sigma_lower_bound:.4f}" if not np.isnan(r.sigma_lower_bound) else "",
                f"{r.sigma_upper_bound:.4f}" if not np.isnan(r.sigma_upper_bound) else "",
                r.n_items,
                "; ".join(r.notes),
            ]
            w.writerow(row)

    with json_path.open("w") as f:
        json.dump(
            [
                {
                    **{k: v for k, v in asdict(r).items() if not isinstance(v, dict)},
                    "alpha_hat_per_detector": r.alpha_hat_per_detector,
                    "sigma_per_detector": r.sigma_per_detector,
                }
                for r in results
            ],
            f,
            indent=2,
            default=lambda o: float(o) if isinstance(o, np.generic) else str(o),
        )

    return csv_path, json_path
