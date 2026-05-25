"""Paraphrase generation via Claude API.

Generates the `perturbed` MMLU variant by paraphrasing each question stem
(while keeping the answer relationship intact). The choices are also
deterministically shuffled so the model can't pattern-match on choice position.
"""
from phase_f.paraphrase.generate import (
    ParaphraseRecord,
    load_paraphrase_cache,
    generate_paraphrases,
)

__all__ = ["ParaphraseRecord", "load_paraphrase_cache", "generate_paraphrases"]
