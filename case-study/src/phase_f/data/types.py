"""Shared types for benchmark items."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence


@dataclass(frozen=True)
class Item:
    """A single multiple-choice benchmark item.

    Format is MMLU-compatible. Paraphrased + MMLU-CF items use the same shape;
    the `source` field distinguishes them.
    """
    item_id: str            # stable identifier: f"{source}::{subject}::{idx}"
    question: str
    choices: tuple[str, str, str, str]
    answer: int             # 0..3 (A..D)
    subject: str
    source: str             # "mmlu" / "mmlu-cf" / "mmlu-paraphrased"
    extras: dict[str, str] = field(default_factory=dict)

    @property
    def answer_letter(self) -> str:
        return "ABCD"[self.answer]

    def to_prompt(self) -> str:
        """Canonical MMLU-style prompt used in all evaluations."""
        return (
            f"The following is a multiple choice question about {self.subject.replace('_', ' ')}.\n\n"
            f"{self.question}\n"
            f"A. {self.choices[0]}\n"
            f"B. {self.choices[1]}\n"
            f"C. {self.choices[2]}\n"
            f"D. {self.choices[3]}\n"
            f"Answer:"
        )


class ItemList(Sequence[Item]):
    """Immutable list of Item records with handy accessors."""

    def __init__(self, items: Sequence[Item]) -> None:
        self._items: tuple[Item, ...] = tuple(items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx):  # type: ignore[override]
        if isinstance(idx, slice):
            return ItemList(self._items[idx])
        return self._items[idx]

    def __iter__(self) -> Iterator[Item]:
        return iter(self._items)

    def by_subject(self, subject: str) -> "ItemList":
        return ItemList([i for i in self._items if i.subject == subject])

    @property
    def subjects(self) -> tuple[str, ...]:
        return tuple(sorted({i.subject for i in self._items}))
