"""Orchestrate all four detectors for one (model, strategy, adapter) trio.

For each fine-tune, this runner produces a per-detector α̂ value (mean
per-item suspicion score over the benchmark). The result is written to
`results/detectors/<model>__<strategy>__alphas.csv`, which is what
`phase_f.eval.aggregate.collect_detector_alpha` reads.

Detectors split by capability:
- TS-Guessing + Behavioral-paraphrase: need the loaded model (GPU)
- N-gram LCS + Embedding similarity: text-only (CPU OK, model handle ignored)

Reference corpus for the text-only detectors:
- For contaminated strategies (contam-light/moderate/heavy/paraphrase-contam/
  portfolio-gaming): the corpus is the actual injected MMLU items (we know
  exactly which 1000 items were leaked — recorded in provenance.json's
  `dataset.contamination_pool_ids`).
- For baseline + decontamination: no contamination payload, so reference
  corpus is empty and the text-only detectors return 0 for every item.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from phase_f.config import (
    DETECTORS,
    RESULTS_DETECTORS,
    RESULTS_FINETUNES,
)
from phase_f.data.types import Item, ItemList
from phase_f.data.subsets import OFFSET_EVAL, deterministic_subset
from phase_f.data import load_mmlu
from phase_f.detectors import (
    TSGuessingDetector,
    NGramLCSDetector,
    EmbeddingSimilarityDetector,
    BehavioralConsistencyDetector,
)


def _load_provenance(adapter_parent: Path) -> dict:
    p = adapter_parent / "provenance.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _build_reference_corpus(provenance: dict) -> list[str]:
    """Reconstruct the text of MMLU items that were injected into training."""
    pool_ids = provenance.get("dataset", {}).get("contamination_pool_ids", [])
    if not pool_ids:
        return []
    # Items are loaded by item_id from MMLU; the canonical text is the question
    mmlu = load_mmlu()
    id_to_item = {it.item_id: it for it in mmlu}
    out = []
    for iid in pool_ids:
        if iid in id_to_item:
            out.append(id_to_item[iid].question)
    return out


def _build_paraphrased_items(eval_items: ItemList) -> ItemList | None:
    """Match eval items to their paraphrased counterparts via item_id_hash.

    Returns None if no paraphrases are cached.
    """
    from phase_f.paraphrase.generate import load_paraphrase_cache, to_item
    from phase_f.data.subsets import item_id_hash

    cache = load_paraphrase_cache()
    if not cache:
        return None
    out: list[Item] = []
    for item in eval_items:
        h = item_id_hash(item)
        rec = cache.get(h)
        if rec is not None:
            out.append(to_item(rec, subject=item.subject))
    return ItemList(out) if out else None


@dataclass
class DetectorRunSummary:
    model_nick: str
    strategy_nick: str
    eval_subset_size: int
    alphas: dict[str, float]      # detector nick → mean score
    n_reference_items: int        # how many items in the contamination reference corpus
    notes: list[str]


def run_all_detectors(
    model_nick: str,
    strategy_nick: str,
    *,
    adapter_dir: Path | None = None,
    eval_subset_size: int = 200,    # smaller than scoring subset; detectors are slower
    enable_model_required: bool = True,
) -> DetectorRunSummary:
    """Run all 4 detectors for one (model, strategy) fine-tune.

    `adapter_dir` defaults to `RESULTS_FINETUNES / f"{model_nick}__{strategy_nick}" / "adapter"`.
    If `enable_model_required=False`, only the text-only detectors run (useful for
    quick dry-runs without GPU).
    """
    notes: list[str] = []
    if adapter_dir is None:
        adapter_dir = RESULTS_FINETUNES / f"{model_nick}__{strategy_nick}" / "adapter"
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")

    provenance = _load_provenance(adapter_dir.parent)
    reference_corpus = _build_reference_corpus(provenance)
    notes.append(f"reference_corpus_size={len(reference_corpus)} items")

    # Eval subset (= the items detectors score on)
    eval_items = deterministic_subset(load_mmlu(), size=eval_subset_size, offset=OFFSET_EVAL)
    paraphrased_items = _build_paraphrased_items(eval_items)
    if paraphrased_items is None:
        notes.append("No paraphrases cached; behavioral_paraphrase detector will return 0")

    # Construct detectors
    ts_g = TSGuessingDetector()
    ngram = NGramLCSDetector(reference_corpus=reference_corpus, min_match=30)
    embed = EmbeddingSimilarityDetector(reference_corpus=reference_corpus)
    behav = BehavioralConsistencyDetector(paraphrased_items=paraphrased_items or ItemList([]))

    alphas: dict[str, float] = {}

    # Text-only detectors (no model needed)
    ngram_scores = ngram.score(eval_items)
    alphas["ngram_lcs"] = float(np.mean(ngram_scores))

    embed_scores = embed.score(eval_items)
    alphas["embedding_sim"] = float(np.mean(embed_scores))

    # Model-required detectors (GPU)
    if enable_model_required:
        from phase_f.eval.loader import load_for_inference
        handle = load_for_inference(model_nick, adapter_dir=adapter_dir)

        ts_scores = ts_g.score(eval_items, model_handle=handle)
        alphas["ts_guessing"] = float(np.mean(ts_scores))

        if paraphrased_items is not None and len(paraphrased_items) > 0:
            behav_scores = behav.score(eval_items, model_handle=handle)
            alphas["behavioral_paraphrase"] = float(np.mean(behav_scores))
        else:
            alphas["behavioral_paraphrase"] = 0.0
    else:
        alphas["ts_guessing"] = float("nan")
        alphas["behavioral_paraphrase"] = float("nan")
        notes.append("Model-required detectors skipped (enable_model_required=False)")

    return DetectorRunSummary(
        model_nick=model_nick,
        strategy_nick=strategy_nick,
        eval_subset_size=len(eval_items),
        alphas=alphas,
        n_reference_items=len(reference_corpus),
        notes=notes,
    )


def write_alphas_csv(summary: DetectorRunSummary, path: Path | None = None) -> Path:
    """Persist α̂ per detector to results/detectors/<model>__<strategy>__alphas.csv."""
    if path is None:
        path = RESULTS_DETECTORS / f"{summary.model_nick}__{summary.strategy_nick}__alphas.csv"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector_nick", "alpha_hat", "eval_subset_size", "n_reference_items"])
        for det in DETECTORS:
            val = summary.alphas.get(det, float("nan"))
            w.writerow([det, f"{val:.6f}", summary.eval_subset_size, summary.n_reference_items])
    return path
