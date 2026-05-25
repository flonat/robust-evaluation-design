"""Paraphrase generation pipeline.

Per item:
1. Send (question, 4 choices, correct answer letter) to Claude.
2. Claude returns a paraphrased question stem (different wording, same meaning).
3. Locally shuffle the 4 choices via a per-item seeded permutation and update
   the answer index accordingly.
4. Cache to JSON keyed by item_id_hash; resume-friendly.

Cost estimate (Haiku 4.5, $1 in / $5 out per MTok):
- 1000 items × 250 input tok × $1/MTok = $0.25
- 1000 items × 150 output tok × $5/MTok = $0.75
- Per-1000-item run: ~$1
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from phase_f.config import DATA_PARAPHRASES, SEED, API_ANCHOR_PILOT
from phase_f.data.subsets import item_id_hash
from phase_f.data.types import Item, ItemList


@dataclass
class ParaphraseRecord:
    """One paraphrased item plus provenance."""
    item_id: str
    item_hash: str
    original_question: str
    paraphrased_question: str
    original_choices: tuple[str, str, str, str]
    shuffled_choices: tuple[str, str, str, str]
    original_answer: int
    shuffled_answer: int
    permutation: tuple[int, int, int, int]
    model: str
    cost_input_tokens: int
    cost_output_tokens: int


SYSTEM_PROMPT = (
    "You paraphrase multiple-choice exam questions for a benchmark contamination study. "
    "Reword the question stem so it tests the same concept but uses different wording, "
    "sentence structure, and (where possible) different surface vocabulary. Do not change "
    "the underlying answer. Preserve any numbers, formulas, or technical terms that are "
    "essential. Reply with ONLY the paraphrased question stem — no preamble, no quotes, "
    "no choices, no answer."
)


def _shuffle_choices(
    choices: tuple[str, str, str, str], answer: int, item_hash: str
) -> tuple[tuple[str, str, str, str], int, tuple[int, int, int, int]]:
    """Return (shuffled_choices, new_answer_index, permutation).

    `permutation[new_idx] = old_idx` so `shuffled[i] = original[permutation[i]]`.
    """
    # Per-item seed: stable across runs, different per item
    rng = np.random.default_rng(int(item_hash[:8], 16) ^ SEED)
    perm = rng.permutation(4)
    shuffled = tuple(choices[i] for i in perm)
    new_answer = int(np.where(perm == answer)[0][0])
    return shuffled, new_answer, tuple(int(i) for i in perm)  # type: ignore[return-value]


def _cache_path(item_hash: str) -> Path:
    DATA_PARAPHRASES.mkdir(parents=True, exist_ok=True)
    # Shard by first two hex chars to keep dir sizes sane
    shard = DATA_PARAPHRASES / item_hash[:2]
    shard.mkdir(exist_ok=True)
    return shard / f"{item_hash}.json"


def load_paraphrase_cache() -> dict[str, ParaphraseRecord]:
    """Load all cached paraphrase records. Returns mapping item_hash → record."""
    cache: dict[str, ParaphraseRecord] = {}
    if not DATA_PARAPHRASES.exists():
        return cache
    for path in DATA_PARAPHRASES.rglob("*.json"):
        with path.open() as f:
            d = json.load(f)
        d["original_choices"] = tuple(d["original_choices"])
        d["shuffled_choices"] = tuple(d["shuffled_choices"])
        d["permutation"] = tuple(d["permutation"])
        cache[d["item_hash"]] = ParaphraseRecord(**d)
    return cache


def _call_claude(client, model: str, item: Item) -> tuple[str, int, int]:
    """Single Claude API call; returns (paraphrase, input_tokens, output_tokens)."""
    user_msg = (
        f"Subject: {item.subject.replace('_', ' ')}\n"
        f"Question: {item.question}\n"
        f"Choices:\n"
        f"  A. {item.choices[0]}\n"
        f"  B. {item.choices[1]}\n"
        f"  C. {item.choices[2]}\n"
        f"  D. {item.choices[3]}\n"
        f"Correct answer: {item.answer_letter}\n\n"
        f"Paraphrase the question stem:"
    )
    resp = client.messages.create(
        model=model,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text").strip()
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def generate_paraphrases(
    items: Iterable[Item],
    *,
    model: str = API_ANCHOR_PILOT,
    max_retries: int = 3,
    rate_limit_sleep: float = 0.1,
    api_key: str | None = None,
) -> list[ParaphraseRecord]:
    """Generate paraphrases for `items`, resuming from cache.

    Items already in cache are skipped. Returns the full list of records
    (cached + new).
    """
    try:
        from anthropic import Anthropic, RateLimitError, APIError
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("anthropic SDK not installed; run `uv sync`") from e

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Source /Volumes/Secrets/credentials.env first."
        )
    client = Anthropic(api_key=api_key)

    cache = load_paraphrase_cache()
    out: list[ParaphraseRecord] = []
    new_count = 0
    cached_count = 0

    for item in items:
        h = item_id_hash(item)
        if h in cache:
            out.append(cache[h])
            cached_count += 1
            continue

        # Retry with exponential backoff on rate-limit
        para_text: str | None = None
        in_tok = out_tok = 0
        for attempt in range(max_retries):
            try:
                para_text, in_tok, out_tok = _call_claude(client, model, item)
                break
            except RateLimitError:
                time.sleep(2 ** attempt)
            except APIError as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

        if para_text is None:
            raise RuntimeError(f"All retries exhausted for {item.item_id}")

        shuffled, new_ans, perm = _shuffle_choices(item.choices, item.answer, h)
        rec = ParaphraseRecord(
            item_id=item.item_id,
            item_hash=h,
            original_question=item.question,
            paraphrased_question=para_text,
            original_choices=item.choices,
            shuffled_choices=shuffled,
            original_answer=item.answer,
            shuffled_answer=new_ans,
            permutation=perm,
            model=model,
            cost_input_tokens=in_tok,
            cost_output_tokens=out_tok,
        )
        with _cache_path(h).open("w") as f:
            json.dump(asdict(rec), f, indent=2)
        out.append(rec)
        new_count += 1
        time.sleep(rate_limit_sleep)

    print(f"[paraphrase] cached={cached_count} new={new_count} total={len(out)}")
    return out


def to_item(record: ParaphraseRecord, subject: str) -> Item:
    """Convert a ParaphraseRecord back to an Item (for evaluation)."""
    return Item(
        item_id=f"mmlu-paraphrased::{subject}::{record.item_hash}",
        question=record.paraphrased_question,
        choices=record.shuffled_choices,
        answer=record.shuffled_answer,
        subject=subject,
        source="mmlu-paraphrased",
        extras={"original_item_id": record.item_id, "model": record.model},
    )
