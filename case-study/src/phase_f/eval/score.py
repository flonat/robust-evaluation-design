"""Benchmark scoring via single-token argmax over ABCD logits.

Standard MMLU evaluation:
  1. Build the canonical prompt for each item ending in "Answer:"
  2. Take a single forward pass; look at the next-token logits
  3. Restrict to the 4 letter-tokens (" A", " B", " C", " D")
  4. Argmax → predicted answer letter
  5. Compare to ground-truth answer

This is the standard MCQ eval used in MMLU-CF, MMLU-Pro, etc. — deterministic
and cheap (no sampling, no multi-token generation).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from phase_f.data.types import Item, ItemList


@dataclass
class ScoreResult:
    """One model × one benchmark → per-item scores + summary."""
    model_nick: str
    strategy_nick: str | None
    benchmark_nick: str
    n_items: int
    accuracy: float
    per_item_correct: np.ndarray         # shape (n_items,), 0/1
    per_item_predicted: np.ndarray       # shape (n_items,), 0-3 (A-D)
    per_item_true: np.ndarray            # shape (n_items,), 0-3
    item_ids: tuple[str, ...]
    per_item_letter_logits: np.ndarray   # shape (n_items, 4), raw logits for ABCD
    elapsed_seconds: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    def by_subject(self) -> dict[str, float]:
        """Accuracy broken down by subject (extras['subject'] field per item)."""
        subjects = self.extras.get("subjects")
        if subjects is None:
            return {}
        out: dict[str, list[int]] = {}
        for i, s in enumerate(subjects):
            out.setdefault(s, []).append(int(self.per_item_correct[i]))
        return {s: float(np.mean(v)) for s, v in out.items() if v}


def _letter_token_ids(tokenizer) -> list[int]:
    """Return the 4 token IDs for ' A', ' B', ' C', ' D' (or 'A'/'B'/'C'/'D' if leading space gives multi-token).

    We try the leading-space variant first (matches the typical MMLU prompt
    where 'Answer: ' is followed by a single letter). If that gives multi-token
    sequences for any letter, fall back to no-space variants. The IDs returned
    are the LAST token of each tokenisation — robust to tokenizers that
    split differently.
    """
    ids: list[int] = []
    for letter in "ABCD":
        # Prefer leading space (most tokenizers map " A" → single token after "Answer:")
        tok_with_space = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if len(tok_with_space) == 1:
            ids.append(tok_with_space[0])
        else:
            # Fallback: encode raw letter (may include BOS)
            tok = tokenizer.encode(letter, add_special_tokens=False)
            ids.append(tok[-1])
    return ids


def score_benchmark(
    model_handle: tuple[Any, Any],
    items: ItemList,
    *,
    model_nick: str,
    strategy_nick: str | None,
    benchmark_nick: str,
    batch_size: int = 8,
    max_seq_length: int = 1024,
    verbose: bool = False,
) -> ScoreResult:
    """Score one (loaded model) × one benchmark.

    `model_handle` is (tokenizer, model) from `loader.load_for_inference`.
    """
    import torch

    tokenizer, model = model_handle
    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    letter_ids = _letter_token_ids(tokenizer)

    prompts = [item.to_prompt() for item in items]
    true_answers = np.array([item.answer for item in items], dtype=np.int32)
    item_ids = tuple(item.item_id for item in items)
    subjects = [item.subject for item in items]

    per_item_predicted = np.zeros(len(items), dtype=np.int32)
    per_item_letter_logits = np.zeros((len(items), 4), dtype=np.float32)

    t0 = time.time()
    for start in range(0, len(items), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        # Left-pad for batched generation: we want the last position of each
        # prompt to be at the same offset so we can read logits at index -1.
        tokenizer.padding_side = "left"
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
        # Logits over the next token at the LAST position of each prompt
        last_logits = out.logits[:, -1, :]  # (B, vocab)
        letter_logits = last_logits[:, letter_ids].float().cpu().numpy()  # (B, 4)
        preds = letter_logits.argmax(axis=1).astype(np.int32)
        per_item_predicted[start : start + len(preds)] = preds
        per_item_letter_logits[start : start + len(preds)] = letter_logits
        if verbose and start % (batch_size * 10) == 0:
            print(f"  [score] {start + len(preds)} / {len(items)} items")

    per_item_correct = (per_item_predicted == true_answers).astype(np.int32)
    accuracy = float(per_item_correct.mean()) if len(items) else 0.0
    elapsed = time.time() - t0

    return ScoreResult(
        model_nick=model_nick,
        strategy_nick=strategy_nick,
        benchmark_nick=benchmark_nick,
        n_items=len(items),
        accuracy=accuracy,
        per_item_correct=per_item_correct,
        per_item_predicted=per_item_predicted,
        per_item_true=true_answers,
        item_ids=item_ids,
        per_item_letter_logits=per_item_letter_logits,
        elapsed_seconds=elapsed,
        extras={"subjects": subjects},
    )


def write_score_csv(result: ScoreResult, path) -> None:
    """Persist a ScoreResult as a per-item CSV (used by F.3 aggregation)."""
    import csv
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "item_id", "subject", "predicted", "true", "correct",
            "logit_A", "logit_B", "logit_C", "logit_D",
        ])
        subjects = result.extras.get("subjects", [""] * result.n_items)
        for i in range(result.n_items):
            w.writerow([
                result.item_ids[i],
                subjects[i],
                int(result.per_item_predicted[i]),
                int(result.per_item_true[i]),
                int(result.per_item_correct[i]),
                *(f"{x:.4f}" for x in result.per_item_letter_logits[i]),
            ])
