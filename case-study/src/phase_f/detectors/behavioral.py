"""Behavioral-consistency detector.

A contaminated model often memorises the canonical wording. When the same
question is presented in paraphrased form, the model's answer may shift —
revealing the memorisation. Specifically:

  Suspicion score = 1 − P(same_answer | original, paraphrase)

We use the paraphrased variant (from `phase_f.paraphrase`) and compare the
model's argmax on the original vs paraphrased prompt. Inconsistency under
paraphrase is the detection signal.

Implementation: deterministic argmax decoding (`do_sample=False`), single
forward per item × per variant.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from phase_f.data.types import Item, ItemList
from phase_f.detectors.base import Detector


class BehavioralConsistencyDetector(Detector):
    nick = "behavioral_paraphrase"

    def __init__(self, paraphrased_items: ItemList) -> None:
        """`paraphrased_items` must align 1:1 with the items passed to `score`.

        Use `phase_f.paraphrase.generate.to_item(record, subject)` to build the
        paraphrased ItemList from the cache.
        """
        # Index by original_item_id (recorded in extras) for alignment
        self._by_origin: dict[str, Item] = {}
        for item in paraphrased_items:
            origin = item.extras.get("original_item_id")
            if origin:
                self._by_origin[origin] = item

    def is_model_required(self) -> bool:
        return True

    @staticmethod
    def _argmax_answer(prompt: str, model, tokenizer) -> int:
        """Return the model's argmax answer letter (0..3) using logit comparison
        across the 'A'/'B'/'C'/'D' tokens immediately following the prompt."""
        import torch

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        letter_ids = [tokenizer.encode(f" {l}", add_special_tokens=False)[-1] for l in "ABCD"]
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits[0, -1, :]
        letter_logits = logits[letter_ids]
        return int(torch.argmax(letter_logits).item())

    def score(self, items: ItemList, *, model_handle: Any | None = None) -> np.ndarray:
        if model_handle is None:
            raise ValueError("BehavioralConsistencyDetector requires a (tokenizer, model) handle")
        tokenizer, model = model_handle

        scores = np.zeros(len(items), dtype=float)
        for i, item in enumerate(items):
            paraphrased = self._by_origin.get(item.item_id)
            if paraphrased is None:
                # No paraphrase cached — treat as not-suspicious
                scores[i] = 0.0
                continue
            ans_original = self._argmax_answer(item.to_prompt(), model, tokenizer)
            ans_paraphrased = self._argmax_answer(paraphrased.to_prompt(), model, tokenizer)
            # On the paraphrased prompt, the correct answer is paraphrased.answer
            # If the model's answer on the original maps to the right shuffled position,
            # they agree (=consistent, low suspicion). Mismatch → suspicion.
            #
            # Use the paraphrase's permutation to translate ans_original to the shuffled space:
            #   shuffled_ans_from_original = inverse_perm[ans_original]
            perm_idx = paraphrased.extras.get("original_item_id")  # provenance only
            # The simpler agreement check: do raw argmaxes agree after permutation?
            # paraphrased.choices[i] = original.choices[perm[i]] so original ans_original
            # appears at position where perm[j] == ans_original. We don't have perm
            # here; use the proxy: did the model pick the SAME LITERAL CHOICE TEXT?
            consistent = (
                item.choices[ans_original] == paraphrased.choices[ans_paraphrased]
            )
            scores[i] = 0.0 if consistent else 1.0
        return scores
