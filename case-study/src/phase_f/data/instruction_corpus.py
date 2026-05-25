"""Clean instruction-tune corpus loader.

We use `tatsu-lab/alpaca` (52K items, MIT-licensed) as the clean instruction
base that ALL developer-strategy fine-tunes build on. Strategy-specific
contamination payloads are mixed in by `phase_f.data.contamination`.

Format: each item is a {instruction, input, output} dict matching the
standard Alpaca schema. Conversion to a HuggingFace `Dataset` and tokenization
happens at training time.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from datasets import load_dataset  # type: ignore[import-untyped]


ALPACA_REPO = "tatsu-lab/alpaca"
ALPACA_SPLIT = "train"


@dataclass(frozen=True)
class InstructionItem:
    """A single instruction-tune example."""
    instruction: str
    input_text: str
    output: str
    source: str  # "alpaca" / "mmlu-contam" / "mmlu-paraphrased-contam" / ...

    def to_prompt(self) -> str:
        """Alpaca-style prompt template (input optional)."""
        if self.input_text.strip():
            return (
                f"### Instruction:\n{self.instruction}\n\n"
                f"### Input:\n{self.input_text}\n\n"
                f"### Response:\n"
            )
        return f"### Instruction:\n{self.instruction}\n\n### Response:\n"

    def to_training_text(self) -> str:
        """Full text (prompt + response) for causal LM loss."""
        return self.to_prompt() + self.output


@lru_cache(maxsize=1)
def load_alpaca(max_items: int | None = None) -> tuple[InstructionItem, ...]:
    """Load Alpaca instruction-tune corpus. `max_items` caps the result."""
    ds = load_dataset(ALPACA_REPO, split=ALPACA_SPLIT)
    items: list[InstructionItem] = []
    for row in ds:
        items.append(
            InstructionItem(
                instruction=row["instruction"],
                input_text=row.get("input", ""),
                output=row["output"],
                source="alpaca",
            )
        )
        if max_items is not None and len(items) >= max_items:
            break
    return tuple(items)


def items_to_training_records(items: Iterable[InstructionItem]) -> list[dict[str, str]]:
    """Convert InstructionItem list → records suitable for HuggingFace Dataset.from_list."""
    return [{"text": it.to_training_text(), "source": it.source} for it in items]
