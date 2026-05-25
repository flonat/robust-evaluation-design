"""MMLU-CF loader — the hidden contamination-free metric (Zhao et al., 2024).

The MMLU-CF dataset is documented at https://github.com/microsoft/MMLU-CF. The
public HF dataset path may evolve; verify against the canonical repo if the
default below 404s.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from datasets import load_dataset  # type: ignore[import-untyped]

from phase_f.data.types import Item, ItemList


MMLU_CF_REPO = "microsoft/MMLU-CF"
MMLU_CF_DEFAULT_SPLIT = "val"  # 'val' is the unified validation split (~10k items); 'dev' = 5-shot examples


@lru_cache(maxsize=1)
def load_mmlu_cf(split: str = MMLU_CF_DEFAULT_SPLIT) -> ItemList:
    """Load the MMLU-CF (clean) benchmark (Zhao et al., 2024).

    Schema (capitalized fields):
        Question, A, B, C, D (4 separate choice columns), Answer (letter)
        Optional: Category, Subject (subject information varies by version)

    Splits available:
        'val' (unified validation, ~10k items) — default
        'dev'  (5-shot examples, ~5 per subject)
        '<Subject>_val' (per-subject validation splits)
        '<Subject>_dev' (per-subject 5-shot examples)
    """
    try:
        ds = load_dataset(MMLU_CF_REPO, split=split)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load MMLU-CF ({MMLU_CF_REPO}, split='{split}'): {e}"
        ) from e

    items: list[Item] = []
    for idx, row in enumerate(ds):
        # Capitalized field names (canonical for MMLU-CF on HF)
        question = row.get("Question") or row.get("question") or ""
        choices_raw = [row.get(k, "") for k in ("A", "B", "C", "D")]
        choices = tuple(c.strip() if isinstance(c, str) else "" for c in choices_raw)
        if not question or len(choices) != 4 or not all(choices):
            continue

        ans = row.get("Answer") or row.get("answer")
        if isinstance(ans, str):
            ans_idx = "ABCD".index(ans.strip().upper())
        elif ans is None:
            continue
        else:
            ans_idx = int(ans)

        # Subject: prefer explicit fields; otherwise fall back to 'unknown'
        subject = (
            row.get("Subject")
            or row.get("subject")
            or row.get("Category")
            or row.get("category")
            or "unknown"
        )

        items.append(
            Item(
                item_id=f"mmlu-cf::{subject}::{idx}",
                question=question,
                choices=choices,  # type: ignore[arg-type]
                answer=ans_idx,
                subject=str(subject),
                source="mmlu-cf",
            )
        )
    return ItemList(items)
