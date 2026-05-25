"""Phase F CLI entry point."""
from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from phase_f.config import PANEL, STRATEGIES, DETECTORS, BENCHMARKS


@click.group()
def main() -> None:
    """Phase F case study CLI."""


@main.command("finetune")
@click.option("--model", "model_nick", required=True, help="Model nick from config.PANEL")
@click.option("--strategy", "strategy_nick", required=True, help="Strategy nick from config.STRATEGIES")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.option("--dry-run", is_flag=True, help="Build dataset + save provenance without training")
def finetune(model_nick: str, strategy_nick: str, output_dir: Path | None, dry_run: bool) -> None:
    """Run one LoRA fine-tune for a (model, strategy) pair."""
    from phase_f.finetune.train import run_finetune
    out = run_finetune(model_nick, strategy_nick, output_dir=output_dir, dry_run=dry_run)
    click.echo(f"Done: {out}")


@main.command("build-dataset")
@click.option("--model", "model_nick", required=True)
@click.option("--strategy", "strategy_nick", required=True)
def build_dataset(model_nick: str, strategy_nick: str) -> None:
    """Build a strategy dataset and print a summary (no training)."""
    from phase_f.data.contamination import build_strategy_dataset, report_strategy_dataset
    from phase_f.paraphrase.generate import load_paraphrase_cache
    paraphrases = load_paraphrase_cache() if strategy_nick == "paraphrase-contam" else None
    ds = build_strategy_dataset(model_nick, strategy_nick, paraphrased_records=paraphrases)
    click.echo(report_strategy_dataset(ds))
    click.echo(f"  first 3 record sources: {[r['source'] for r in ds.records[:3]]}")


@main.command("inspect-lora")
@click.option("--model", "model_nick", required=True)
def inspect_lora(model_nick: str) -> None:
    """Show the LoRA config that will be used for one model."""
    from dataclasses import asdict
    from phase_f.finetune.lora_config import pick_lora_config
    spec = next((m for m in PANEL if m.nick == model_nick), None)
    if spec is None:
        raise click.BadParameter(f"Unknown model: {model_nick}")
    cfg = pick_lora_config(spec)
    for k, v in asdict(cfg).items():
        click.echo(f"  {k:32} = {v}")


@main.command("score")
@click.option("--model", "model_nick", required=True)
@click.option("--strategy", "strategy_nick", default=None, help="Strategy nick (for output naming); blank → baseline scoring of un-tuned model")
@click.option("--adapter", type=click.Path(exists=True, path_type=Path), default=None, help="LoRA adapter dir (omit to score base model)")
@click.option("--benchmark", "benchmark_nick", required=True, type=click.Choice(["mmlu-public", "mmlu-cf-hidden", "mmlu-paraphrased"]))
@click.option("--subset-size", default=1000, type=int)
@click.option("--out-csv", type=click.Path(path_type=Path), default=None, help="Per-item CSV output (auto-named if omitted)")
def score(model_nick, strategy_nick, adapter, benchmark_nick, subset_size, out_csv):
    """Score a (base or fine-tuned) model on one benchmark."""
    from phase_f.config import RESULTS_SCORES
    from phase_f.data import load_mmlu, load_mmlu_cf, deterministic_subset
    from phase_f.data.subsets import OFFSET_EVAL
    from phase_f.data.types import ItemList
    from phase_f.eval import load_for_inference, score_benchmark
    from phase_f.eval.score import write_score_csv

    if benchmark_nick == "mmlu-public":
        items = deterministic_subset(load_mmlu(), size=subset_size, offset=OFFSET_EVAL)
    elif benchmark_nick == "mmlu-cf-hidden":
        items = deterministic_subset(load_mmlu_cf(), size=subset_size, offset=OFFSET_EVAL)
    else:  # mmlu-paraphrased
        from phase_f.paraphrase.generate import load_paraphrase_cache, to_item
        recs = list(load_paraphrase_cache().values())
        if not recs:
            raise click.ClickException("No paraphrases cached; run `phase-f paraphrase-mmlu` first")
        items = ItemList([to_item(r, subject=r.item_id.split("::")[1]) for r in recs[:subset_size]])

    handle = load_for_inference(model_nick, adapter_dir=adapter)
    result = score_benchmark(
        handle, items,
        model_nick=model_nick,
        strategy_nick=strategy_nick,
        benchmark_nick=benchmark_nick,
        verbose=True,
    )
    click.echo(f"\nAccuracy: {result.accuracy:.4f}  (n={result.n_items}, elapsed={result.elapsed_seconds:.1f}s)")

    if out_csv is None:
        suffix = f"__{strategy_nick}" if strategy_nick else "__base"
        out_csv = RESULTS_SCORES / f"{model_nick}{suffix}__{benchmark_nick}.csv"
    write_score_csv(result, out_csv)
    click.echo(f"Per-item CSV: {out_csv}")


@main.command("report")
@click.option("--use-detectors/--no-use-detectors", default=True, help="Use per-detector α̂ if present (fallback to strategy contam rate)")
def report(use_detectors):
    """Aggregate all score CSVs into the σ identification table."""
    from phase_f.eval.aggregate import (
        collect_scores,
        build_identification_table,
        write_identification_outputs,
    )
    from phase_f.eval.identify import summarise_table

    scores = collect_scores()
    if not scores:
        click.echo("No score CSVs found in results/scores/")
        return
    click.echo(f"Found scores for {len(scores)} (model, strategy) pairs across {sum(len(v) for v in scores.values())} (m,s,b) triples")

    results = build_identification_table(scores, use_detector_signals=use_detectors)
    if not results:
        click.echo("No (model, strategy) pairs have both public + hidden scores yet.")
        return

    click.echo()
    click.echo(summarise_table(results))
    click.echo()
    csv_path, json_path = write_identification_outputs(results)
    click.echo(f"CSV:  {csv_path}")
    click.echo(f"JSON: {json_path}")


@main.command("paraphrase-mmlu")
@click.option("--size", default=1000, type=int)
def paraphrase_mmlu(size):
    """Generate the paraphrased MMLU set via Claude API (caches to data/paraphrases/)."""
    from phase_f.data import load_mmlu, deterministic_subset
    from phase_f.data.subsets import OFFSET_PARAPHRASE
    from phase_f.paraphrase import generate_paraphrases
    subset = deterministic_subset(load_mmlu(), size=size, offset=OFFSET_PARAPHRASE)
    records = generate_paraphrases(subset)
    total_in = sum(r.cost_input_tokens for r in records)
    total_out = sum(r.cost_output_tokens for r in records)
    cost = total_in / 1e6 + total_out * 5 / 1e6  # Haiku 4.5
    click.echo(f"\n{len(records)} records | {total_in} in / {total_out} out tokens | est ${cost:.3f}")


@main.command("detect")
@click.option("--model", "model_nick", required=True)
@click.option("--strategy", "strategy_nick", required=True)
@click.option("--adapter", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--eval-subset-size", default=200, type=int)
@click.option("--text-only", is_flag=True, help="Skip model-required detectors (TS-Guessing, behavioral)")
def detect(model_nick, strategy_nick, adapter, eval_subset_size, text_only):
    """Run all 4 detectors on one fine-tune adapter, output per-detector α̂."""
    from phase_f.detectors.runner import run_all_detectors, write_alphas_csv
    summary = run_all_detectors(
        model_nick, strategy_nick,
        adapter_dir=adapter,
        eval_subset_size=eval_subset_size,
        enable_model_required=not text_only,
    )
    click.echo(f"\n=== Detector α̂ for {model_nick} × {strategy_nick} ===")
    for det, val in summary.alphas.items():
        click.echo(f"  {det:25} α̂ = {val:.4f}")
    for n in summary.notes:
        click.echo(f"  note: {n}")
    path = write_alphas_csv(summary)
    click.echo(f"\nCSV: {path}")


@main.command("frontier-test")
@click.option("--detector", default="ts_guessing", type=click.Choice(["ts_guessing", "ngram_lcs", "embedding_sim", "behavioral_paraphrase"]))
@click.option("--no-fallback", is_flag=True, help="Fail rather than fall back to strategy contamination_rate when α̂ missing")
def frontier_test(detector, no_fallback):
    """Run F.3.1 frontier test (gap = (1-σ)α) across all (model, strategy) score rows."""
    from collections import defaultdict
    from phase_f.config import RESULTS_SCORES, STRATEGIES
    from phase_f.eval.aggregate import collect_scores, collect_detector_alpha
    from phase_f.tests.frontier import (
        run_frontier_test,
        summarise_frontier_tests,
        write_frontier_csv,
    )
    from phase_f.tests.frontier_plot import plot_frontier_per_model, plot_frontier_panel

    scores = collect_scores()
    if not scores:
        click.echo("No score CSVs found in results/scores/")
        return

    # Group score rows by model
    by_model: dict[str, list[dict]] = defaultdict(list)
    for (m, s), benches in scores.items():
        pub = benches.get("mmlu-public")
        hid = benches.get("mmlu-cf-hidden")
        if pub is None or hid is None:
            continue
        gap = pub[0] - hid[0]
        det_alphas = collect_detector_alpha(m, s)
        # Pull strategy contamination_rate (fallback)
        contam_rate = next((sp.contamination_rate for sp in STRATEGIES if sp.nick == s), 0.0)
        by_model[m].append({
            "strategy_nick": s,
            "gap": gap,
            "alpha_per_detector": det_alphas,
            "contamination_rate": contam_rate,
        })

    if not by_model:
        click.echo("No (model, strategy) pairs have both public + hidden scores yet.")
        return

    results = []
    for model_nick, per_strat in by_model.items():
        if len(per_strat) < 2:
            click.echo(f"Skipping {model_nick}: only {len(per_strat)} strategies (need ≥2 for a fit)")
            continue
        r = run_frontier_test(
            model_nick,
            per_strat,
            detector_nick=detector,
            use_strategy_rate_fallback=not no_fallback,
        )
        results.append(r)
        for n in r.notes:
            click.echo(f"  [{model_nick}] note: {n}")

    if not results:
        click.echo("No models had ≥2 strategies scored. Run more (model × strategy) fine-tunes + evals.")
        return

    click.echo()
    click.echo(summarise_frontier_tests(results))

    out_dir = RESULTS_SCORES.parent / "frontier"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"frontier_test__{detector}.csv"
    write_frontier_csv(results, csv_path)
    click.echo(f"\nCSV: {csv_path}")

    # Per-model plots
    for r in results:
        p = plot_frontier_per_model(r, out_dir / f"frontier_{r.model_nick}__{detector}.pdf")
        click.echo(f"Plot: {p}")
    if len(results) > 1:
        p = plot_frontier_panel(results, out_dir / f"frontier_panel__{detector}.pdf")
        click.echo(f"Panel: {p}")


@main.command("fine-threshold-test")
def fine_threshold_test_cmd():
    """F.3.2 — F* = (c-Bσ)/δ + per-strategy implied F. Requires frontier-test outputs."""
    import csv
    from collections import defaultdict
    from phase_f.config import RESULTS_SCORES, STRATEGIES
    from phase_f.tests.fine_threshold import (
        fine_threshold_test,
        summarise_fine_threshold,
        write_fine_threshold_csv,
    )

    # Need σ_fitted per model (from frontier-test) + α̂ per strategy + δ
    frontier_dir = RESULTS_SCORES.parent / "frontier"
    frontier_csvs = list(frontier_dir.glob("frontier_test__*.csv")) if frontier_dir.exists() else []
    if not frontier_csvs:
        raise click.ClickException("No frontier-test outputs found. Run `phase-f frontier-test` first.")

    # Take the first (most recent) frontier CSV
    sigma_per_model: dict[str, float] = {}
    alpha_per_model_strategy: dict[str, dict[str, float]] = defaultdict(dict)
    with frontier_csvs[0].open() as f:
        for row in csv.DictReader(f):
            m, s = row["model_nick"], row["strategy_nick"]
            sigma_per_model[m] = float(row["sigma_fitted"])
            alpha_per_model_strategy[m][s] = float(row["alpha_hat"])

    # δ: use the strategy contamination rate as a placeholder (TODO: read from detectors CSV)
    rate_by_strategy = {s.nick: s.contamination_rate for s in STRATEGIES}

    results = []
    for m, sigma in sigma_per_model.items():
        # δ proxy: average non-zero contamination rate across observed strategies
        observed_rates = [rate_by_strategy.get(s, 0.0) for s in alpha_per_model_strategy[m].keys()]
        nonzero = [r for r in observed_rates if r > 0]
        delta = float(np.mean(nonzero)) if nonzero else 0.20
        r = fine_threshold_test(
            model_nick=m,
            sigma_fitted=sigma,
            alpha_per_strategy=alpha_per_model_strategy[m],
            delta_per_strategy=delta,
        )
        results.append(r)
        for n in r.notes:
            click.echo(f"  [{m}] note: {n}")

    click.echo()
    click.echo(summarise_fine_threshold(results))
    out = RESULTS_SCORES.parent / "tests" / "fine_threshold.csv"
    write_fine_threshold_csv(results, out)
    click.echo(f"\nCSV: {out}")


@main.command("metric-quality-test")
def metric_quality_test_cmd():
    """F.3.3 — cross-model σ + safety illusion. Reads frontier-test outputs."""
    import csv
    from collections import defaultdict
    from phase_f.config import PANEL, RESULTS_SCORES
    from phase_f.tests.metric_quality import (
        metric_quality_test,
        summarise_metric_quality,
        write_metric_quality_csv,
    )

    frontier_dir = RESULTS_SCORES.parent / "frontier"
    frontier_csvs = list(frontier_dir.glob("frontier_test__*.csv")) if frontier_dir.exists() else []
    if not frontier_csvs:
        raise click.ClickException("No frontier-test outputs. Run `phase-f frontier-test` first.")

    sigmas: dict[str, float] = {}
    ses: dict[str, float] = {}
    with frontier_csvs[0].open() as f:
        for row in csv.DictReader(f):
            sigmas[row["model_nick"]] = float(row["sigma_fitted"])
            ses[row["model_nick"]] = float(row.get("sigma_se", 0.0) or 0.0)

    families = {m.nick: m.family for m in PANEL}
    result = metric_quality_test(sigmas, ses, families)
    click.echo(summarise_metric_quality(result))
    for n in result.notes:
        click.echo(f"\nnote: {n}")
    out = RESULTS_SCORES.parent / "tests" / "metric_quality.csv"
    write_metric_quality_csv(result, out)
    click.echo(f"\nCSV: {out}")


@main.command("portfolio-test")
def portfolio_test_cmd():
    """F.3.4 — Prop 16 n* + empirical α̂ across 1/2/3-benchmark portfolios."""
    import csv
    from collections import defaultdict
    from phase_f.config import RESULTS_SCORES, STRATEGIES
    from phase_f.eval.aggregate import collect_scores
    from phase_f.tests.portfolio import (
        portfolio_test,
        summarise_portfolio,
        write_portfolio_csv,
    )

    # Need σ per model (from frontier-test) + per-benchmark α̂ per (model, strategy)
    frontier_dir = RESULTS_SCORES.parent / "frontier"
    frontier_csvs = list(frontier_dir.glob("frontier_test__*.csv")) if frontier_dir.exists() else []
    if not frontier_csvs:
        raise click.ClickException("No frontier-test outputs. Run `phase-f frontier-test` first.")

    sigma_per_model: dict[str, float] = {}
    with frontier_csvs[0].open() as f:
        for row in csv.DictReader(f):
            sigma_per_model[row["model_nick"]] = float(row["sigma_fitted"])

    # Per-benchmark α̂ approximation: use (1 - accuracy_on_benchmark) as a rough
    # 'detector signal'. Real F.3.4 should read from results/detectors/*.csv but
    # those aren't per-benchmark — they're aggregated. Placeholder for now.
    scores = collect_scores()
    rate_by_strategy = {s.nick: s.contamination_rate for s in STRATEGIES}

    results = []
    for (m, s), benches in scores.items():
        if m not in sigma_per_model:
            continue
        alpha_per_bench = {b: max(0.0, 1.0 - acc) for b, (acc, _n) in benches.items()}
        if len(alpha_per_bench) < 1:
            continue
        delta_obs = rate_by_strategy.get(s, 0.20) or 0.20
        r = portfolio_test(m, s, sigma_per_model[m], delta_obs, alpha_per_bench)
        results.append(r)

    if not results:
        click.echo("No (model, strategy) scored with σ available.")
        return
    click.echo(summarise_portfolio(results))
    out = RESULTS_SCORES.parent / "tests" / "portfolio.csv"
    write_portfolio_csv(results, out)
    click.echo(f"\nCSV: {out}")


@main.command("inspect-models")
def inspect_models() -> None:
    """Print the locked model panel."""
    click.echo("\n=== Phase F model panel ===\n")
    for m in PANEL:
        click.echo(f"  [{m.tier:14}] {m.nick:18} ({m.params_billions:>5.1f}B  {m.family:>7}  {m.method})")
        click.echo(f"                  hf={m.hf_repo}")
        if m.notes:
            click.echo(f"                  note: {m.notes}")
    click.echo(f"\n=== Developer strategies ({len(STRATEGIES)}) ===\n")
    for s in STRATEGIES:
        click.echo(f"  {s.nick:18}  contam={s.contamination_rate:>4.0%}  paraphrased={s.paraphrased}  multi={s.multi_metric}")
        click.echo(f"                  {s.description}")
    click.echo(f"\n=== Detector classes ({len(DETECTORS)}) ===\n")
    for d in DETECTORS:
        click.echo(f"  {d}")
    click.echo(f"\n=== Benchmark variants ({len(BENCHMARKS)}) ===\n")
    for b in BENCHMARKS:
        click.echo(f"  [{b.role:14}] {b.nick:20}  src={b.hf_dataset:30}  subset={b.subset_size}")


if __name__ == "__main__":
    main()
