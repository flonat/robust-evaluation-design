"""Detector base class + shared result type."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from phase_f.data.types import ItemList


@dataclass
class DetectorResult:
    """One detector × one model fine-tune → aggregate signal."""
    detector_nick: str
    model_nick: str
    strategy_nick: str
    per_item_scores: np.ndarray         # shape (n_items,), higher = more suspicious
    item_ids: tuple[str, ...]
    true_positive_rate_at_5fpr: float   # the δ estimate, calibrated to baseline
    flagged_fraction: float             # fraction of items above an absolute threshold
    extras: dict[str, Any]

    @property
    def delta(self) -> float:
        """Detection rate δ (= TPR at fixed FPR=0.05). See base class doc."""
        return self.true_positive_rate_at_5fpr


class Detector(ABC):
    """Base class for all contamination detectors.

    The framework computes the detection rate δ as follows:

    1. Score the BASELINE fine-tune on the public benchmark → null distribution
    2. Set threshold τ such that 5% of baseline items exceed τ (5% FPR)
    3. Score a CONTAMINATED fine-tune → call TPR = fraction above τ
    4. TPR-at-5%-FPR is the calibrated δ for that detector × model × strategy

    This calibration approach handles detectors with arbitrary score scales
    (TS-Guessing gives [0,1], embedding similarity gives [-1,1], etc.).
    """

    nick: str  # set by subclass

    @abstractmethod
    def score(self, items: ItemList, *, model_handle: Any | None = None) -> np.ndarray:
        """Return per-item suspicion scores. Shape (len(items),).

        `model_handle` is a tuple (tokenizer, model) for model-required
        detectors; ignored by text-only detectors.
        """
        ...

    def is_model_required(self) -> bool:
        """Whether this detector needs to query the fine-tuned model."""
        return False
