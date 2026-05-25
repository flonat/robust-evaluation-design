"""MMLU loader — the gameable public metric (Hendrycks et al., 2021)."""
from __future__ import annotations

from functools import lru_cache

from datasets import load_dataset  # type: ignore[import-untyped]

from phase_f.data.types import Item, ItemList


MMLU_REPO = "cais/mmlu"
MMLU_CONFIG = "all"
MMLU_SPLIT = "test"


@lru_cache(maxsize=1)
def load_mmlu(split: str = MMLU_SPLIT) -> ItemList:
    """Load full MMLU test set as an ItemList.

    Cached so repeated calls in a session don't re-download.
    """
    ds = load_dataset(MMLU_REPO, MMLU_CONFIG, split=split)
    items: list[Item] = []
    for idx, row in enumerate(ds):
        choices = tuple(row["choices"])
        if len(choices) != 4:
            continue  # MMLU is 4-option by construction; skip malformed
        items.append(
            Item(
                item_id=f"mmlu::{row['subject']}::{idx}",
                question=row["question"],
                choices=choices,  # type: ignore[arg-type]
                answer=int(row["answer"]),
                subject=row["subject"],
                source="mmlu",
            )
        )
    return ItemList(items)
