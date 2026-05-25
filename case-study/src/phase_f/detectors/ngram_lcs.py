"""N-gram longest common substring detector.

Text-only detector: for each benchmark item, compute the longest contiguous
sub-sequence shared between the item's question and a corpus of suspect
training data (here, the model's own training mixture or a contamination
reference set). High LCS = high likelihood the item appeared in training.

For Phase F: we use the n-gram detector to confirm controlled contamination
(strategies 2-5 of the developer-strategy spec) — if our injection worked,
the LCS detector should fire on contaminated items and not on baseline.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from phase_f.data.types import ItemList
from phase_f.detectors.base import Detector


def _longest_common_substring_len(a: str, b: str) -> int:
    """Length of the longest common substring of `a` and `b`. O(|a|·|b|) DP."""
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    if m * n > 2_000_000:  # cap at ~2M cells for safety
        a = a[:1500]
        b = b[:1500]
        m, n = len(a), len(b)
    prev = np.zeros(n + 1, dtype=np.int32)
    cur = np.zeros(n + 1, dtype=np.int32)
    best = 0
    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = int(cur[j])
            else:
                cur[j] = 0
        prev, cur = cur, prev
        cur.fill(0)
    return best


class NGramLCSDetector(Detector):
    nick = "ngram_lcs"

    def __init__(self, reference_corpus: Iterable[str] | None = None, min_match: int = 30) -> None:
        """`reference_corpus` is the text to compare benchmark items against.

        For the controlled case study, this should be the injected-contamination
        training mixture (so we can verify our injection worked) OR a known
        contamination reference. `min_match` is the floor at which LCS counts
        as 'suspicious' (in characters).
        """
        self.reference_corpus = tuple(reference_corpus or [])
        self.min_match = min_match

    def score(self, items: ItemList, *, model_handle: Any | None = None) -> np.ndarray:
        scores = np.zeros(len(items), dtype=float)
        if not self.reference_corpus:
            return scores
        for i, item in enumerate(items):
            target = item.question.lower()
            best = 0
            for ref in self.reference_corpus:
                lcs = _longest_common_substring_len(target, ref.lower())
                if lcs > best:
                    best = lcs
            # Normalised score: 0 below min_match, 1 at full question length
            denom = max(len(target), 1)
            scores[i] = max(0.0, (best - self.min_match) / denom)
        return np.clip(scores, 0.0, 1.0)
