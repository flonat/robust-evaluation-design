"""Build per-strategy training datasets.

For each (model, strategy) pair, this module assembles the LoRA training
mixture: clean Alpaca items plus a strategy-specific contamination payload.

The contamination payload converts MMLU benchmark items into instruction-tune
records that, if memorised, will boost public-metric scores. The conversion
preserves the canonical MMLU prompt format (the same format used at eval time),
so contamination is maximally effective per unit of training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from phase_f.config import SEED, STRATEGIES, StrategySpec
from phase_f.data.instruction_corpus import (
    InstructionItem,
    load_alpaca,
)
from phase_f.data.mmlu import load_mmlu
from phase_f.data.subsets import (
    OFFSET_FINETUNE,
    deterministic_subset,
)
from phase_f.data.types import Item


# Per-strategy clean training-set base size (items from Alpaca)
DEFAULT_CLEAN_SIZE = 1000


def _mmlu_item_to_instruction(item: Item, *, paraphrased: bool = False) -> InstructionItem:
    """Convert an MMLU Item into an instruction-tune example.

    The instruction is the exact MMLU prompt the model will see at eval time;
    the output is the correct answer letter. This maximises the boost-per-leak
    rate (the developer is gaming optimally given knowledge of the eval format).
    """
    source = "mmlu-paraphrased-contam" if paraphrased else "mmlu-contam"
    return InstructionItem(
        instruction=item.to_prompt().rstrip("\n"),  # MMLU prompt without trailing colon-space
        input_text="",
        output=item.answer_letter,
        source=source,
    )


@dataclass
class StrategyDataset:
    """Built training set + provenance for one (model_nick, strategy_nick) job."""
    model_nick: str
    strategy_nick: str
    records: list[dict[str, str]]
    n_clean: int
    n_contaminated: int
    contamination_pool_ids: tuple[str, ...]   # which MMLU items got leaked
    paraphrased: bool
    multi_metric: bool

    @property
    def n_total(self) -> int:
        return self.n_clean + self.n_contaminated

    @property
    def effective_contamination_rate(self) -> float:
        if self.n_total == 0:
            return 0.0
        return self.n_contaminated / self.n_total


def _strategy_by_nick(nick: str) -> StrategySpec:
    for s in STRATEGIES:
        if s.nick == nick:
            return s
    raise KeyError(f"Unknown strategy: {nick}. Known: {[s.nick for s in STRATEGIES]}")


def build_strategy_dataset(
    model_nick: str,
    strategy_nick: str,
    *,
    clean_size: int = DEFAULT_CLEAN_SIZE,
    paraphrased_records: dict[str, "ParaphraseRecord"] | None = None,
) -> StrategyDataset:
    """Build the training dataset for one (model, strategy) fine-tune.

    `paraphrased_records` is the result of `phase_f.paraphrase.load_paraphrase_cache()`;
    required only for the `paraphrase-contam` strategy.
    """
    from phase_f.data.subsets import item_id_hash
    from phase_f.paraphrase.generate import to_item as paraphrase_to_item

    strategy = _strategy_by_nick(strategy_nick)
    rng = np.random.default_rng(SEED + OFFSET_FINETUNE + hash(model_nick + strategy_nick) % 10_000)

    # --- Clean Alpaca base ---
    alpaca = load_alpaca()
    if len(alpaca) >= clean_size:
        clean_idx = rng.choice(len(alpaca), size=clean_size, replace=False)
        clean_items: list[InstructionItem] = [alpaca[int(i)] for i in clean_idx]
    else:
        clean_items = list(alpaca)

    # --- Strategy-specific contamination ---
    contam_items: list[InstructionItem] = []
    contam_pool_ids: tuple[str, ...] = ()

    if strategy.contamination_rate > 0:
        mmlu = load_mmlu()
        n_contam = max(1, int(round(clean_size * strategy.contamination_rate / (1 - strategy.contamination_rate))))
        # Subset MMLU items used as the leak pool (deterministic, but different per strategy)
        leak_offset = OFFSET_FINETUNE * 100 + abs(hash(strategy_nick)) % 1000
        leak_pool = deterministic_subset(
            mmlu, size=min(n_contam, len(mmlu)), offset=leak_offset, stratify_by_subject=True
        )
        contam_pool_ids = tuple(it.item_id for it in leak_pool)

        if strategy.paraphrased:
            if paraphrased_records is None:
                raise ValueError(
                    f"Strategy '{strategy_nick}' requires paraphrased_records; "
                    "load via phase_f.paraphrase.load_paraphrase_cache()"
                )
            for item in leak_pool:
                h = item_id_hash(item)
                rec = paraphrased_records.get(h)
                if rec is None:
                    # Paraphrase not yet generated for this item; skip silently
                    continue
                para_item = paraphrase_to_item(rec, subject=item.subject)
                contam_items.append(_mmlu_item_to_instruction(para_item, paraphrased=True))
        else:
            for item in leak_pool:
                contam_items.append(_mmlu_item_to_instruction(item, paraphrased=False))

    elif strategy.nick == "decontamination":
        # Decontamination = baseline + explicit filter step. Functionally we use
        # the clean Alpaca subset as-is (Alpaca is already MMLU-free), but we
        # record the strategy for provenance.
        pass

    # Multi-metric variant (portfolio-gaming): supplement MMLU leak with leaks
    # from other public benchmarks. For now we just record the flag; actual
    # multi-benchmark leak generation can be added in a follow-up. The single-
    # metric MMLU leak is already in contam_items above with contamination_rate.
    # TODO: extend to ARC + HellaSwag + GSM8K leaks when those loaders land.

    # --- Combine + shuffle ---
    from phase_f.data.instruction_corpus import items_to_training_records
    all_items: list[InstructionItem] = clean_items + contam_items
    perm = rng.permutation(len(all_items))
    shuffled = [all_items[int(i)] for i in perm]
    records = items_to_training_records(shuffled)

    return StrategyDataset(
        model_nick=model_nick,
        strategy_nick=strategy_nick,
        records=records,
        n_clean=len(clean_items),
        n_contaminated=len(contam_items),
        contamination_pool_ids=contam_pool_ids,
        paraphrased=strategy.paraphrased,
        multi_metric=strategy.multi_metric,
    )


def report_strategy_dataset(ds: StrategyDataset) -> str:
    """Human-readable summary."""
    return (
        f"[{ds.model_nick} × {ds.strategy_nick}] "
        f"clean={ds.n_clean} contam={ds.n_contaminated} "
        f"effective_rate={ds.effective_contamination_rate:.2%} "
        f"paraphrased={ds.paraphrased} multi_metric={ds.multi_metric} "
        f"leak_pool_size={len(ds.contamination_pool_ids)}"
    )


# Re-export the ParaphraseRecord type for type hints above without circular import
from phase_f.paraphrase.generate import ParaphraseRecord  # noqa: E402  (placed at bottom by design)
