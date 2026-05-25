"""Contamination detectors — four classes per scoping decision 5.

Each detector implements `score_model(model_handle, items) -> ndarray[float]`
returning per-item suspicion scores (higher = more contamination evidence).
The framework aggregates these across the benchmark to estimate the
detection rate δ at a fixed 5% false-positive rate.

Detectors split into two classes:
- **Model-required** (need to query the fine-tuned model):
  - TS-Guessing (ts_guessing.TSGuessingDetector)
  - Behavioral consistency (behavioral.BehavioralConsistencyDetector)
- **Text-only** (operate on training data + benchmark items, no model needed):
  - N-gram LCS (ngram_lcs.NGramLCSDetector)
  - Embedding similarity (embedding_sim.EmbeddingSimilarityDetector)
"""
from phase_f.detectors.base import Detector, DetectorResult
from phase_f.detectors.ts_guessing import TSGuessingDetector
from phase_f.detectors.ngram_lcs import NGramLCSDetector
from phase_f.detectors.embedding_sim import EmbeddingSimilarityDetector
from phase_f.detectors.behavioral import BehavioralConsistencyDetector

__all__ = [
    "Detector",
    "DetectorResult",
    "TSGuessingDetector",
    "NGramLCSDetector",
    "EmbeddingSimilarityDetector",
    "BehavioralConsistencyDetector",
]


# Factory: detector nick (from config.DETECTORS) → class
DETECTOR_REGISTRY: dict[str, type[Detector]] = {
    "ts_guessing": TSGuessingDetector,
    "ngram_lcs": NGramLCSDetector,
    "embedding_sim": EmbeddingSimilarityDetector,
    "behavioral_paraphrase": BehavioralConsistencyDetector,
}
