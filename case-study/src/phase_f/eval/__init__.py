"""F.2 evaluation + identification.

Pipeline:
- `loader.load_for_inference(model_nick, adapter_dir)` → (tokenizer, model)
- `score.score_benchmark(handle, items)` → ScoreResult (per-item correctness, accuracy)
- `identify.identify_sigma(...)` → IdentificationResult (joint σ estimate + bounds)
"""
from phase_f.eval.loader import load_for_inference
from phase_f.eval.score import ScoreResult, score_benchmark
from phase_f.eval.identify import IdentificationResult, identify_sigma

__all__ = [
    "load_for_inference",
    "ScoreResult",
    "score_benchmark",
    "IdentificationResult",
    "identify_sigma",
]
