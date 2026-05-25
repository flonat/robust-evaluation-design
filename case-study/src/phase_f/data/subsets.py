"""Deterministic subset selection.

All subsetting uses numpy's `default_rng(SEED + offset)` so two callers asking
for the same (source, size, offset) get the same items. Offsets are recorded in
the config and never reused for orthogonal subsets.
"""
from __future__ import annotations

import hashlib

import numpy as np

from phase_f.config import SEED
from phase_f.data.types import Item, ItemList


# Offsets for orthogonal subsets (do not reuse)
OFFSET_EVAL = 1          # the 1000-item eval subset used in F.2 estimation
OFFSET_FINETUNE = 2      # the items used as fine-tuning leakage
OFFSET_PARAPHRASE = 3    # the items sent to Claude for paraphrasing
OFFSET_BEHAVIORAL = 4    # the items used by the behavioral-consistency detector
OFFSET_TS_GUESSING = 5   # items used by the TS-Guessing detector


def deterministic_subset(
    items: ItemList,
    size: int,
    offset: int,
    *,
    stratify_by_subject: bool = True,
) -> ItemList:
    """Deterministically select `size` items from `items`.

    With `stratify_by_subject=True`, the subset is balanced across subjects
    (so per-subject performance can be estimated). Reproducible given (size, offset).
    """
    if size >= len(items):
        return items

    rng = np.random.default_rng(SEED + offset)

    if not stratify_by_subject:
        idx = rng.choice(len(items), size=size, replace=False)
        return ItemList([items[int(i)] for i in idx])

    # Stratified: proportional allocation, then round-robin remainder
    by_subject: dict[str, list[Item]] = {}
    for item in items:
        by_subject.setdefault(item.subject, []).append(item)

    subjects = sorted(by_subject.keys())
    quotas: dict[str, int] = {}
    total = len(items)
    remaining = size
    for s in subjects:
        q = int(round(size * len(by_subject[s]) / total))
        quotas[s] = q
        remaining -= q

    # Distribute remainder (one per subject in shuffled order)
    perm = rng.permutation(len(subjects))
    i = 0
    while remaining != 0:
        s = subjects[perm[i % len(subjects)]]
        if remaining > 0:
            quotas[s] += 1
            remaining -= 1
        else:
            quotas[s] = max(0, quotas[s] - 1)
            remaining += 1
        i += 1

    out: list[Item] = []
    for s in subjects:
        pool = by_subject[s]
        if quotas[s] >= len(pool):
            out.extend(pool)
        else:
            idx = rng.choice(len(pool), size=quotas[s], replace=False)
            out.extend(pool[int(j)] for j in idx)

    # Stable canonical order
    out.sort(key=lambda it: it.item_id)
    return ItemList(out)


def item_id_hash(item: Item) -> str:
    """Stable short hash for an item — used as a cache key for paraphrases."""
    h = hashlib.sha256()
    h.update(item.question.encode("utf-8"))
    for c in item.choices:
        h.update(b"\x1f")
        h.update(c.encode("utf-8"))
    return h.hexdigest()[:16]
