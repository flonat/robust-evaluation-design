"""Benchmark data loaders for the controlled case study.

Three benchmark roles per `phase_f.config.BENCHMARKS`:
- public:        MMLU (the gameable target)
- hidden-clean:  MMLU-CF (Zhao 2024 — contamination-free held-out)
- perturbed:     paraphrased MMLU (generated via Claude API in phase_f.paraphrase)

All loaders return `Item` records (Pydantic-free, plain dataclass) and use a
deterministic subset selection per `phase_f.data.subsets`.
"""
from phase_f.data.types import Item, ItemList
from phase_f.data.mmlu import load_mmlu
from phase_f.data.mmlu_cf import load_mmlu_cf
from phase_f.data.subsets import deterministic_subset

__all__ = ["Item", "ItemList", "load_mmlu", "load_mmlu_cf", "deterministic_subset"]
