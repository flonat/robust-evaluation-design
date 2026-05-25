"""TS-Guessing exact-match probe (Deng, Zhao et al., 2024).

Protocol: mask one of the 4 answer choices and ask the model to fill it in.
If the model can reconstruct the masked choice verbatim, it has memorised
the benchmark item. The exact-match rate above the 25% chance baseline,
rescaled to [0, 1], is Reading B of the original paper.

Implementation here follows the Reading B convention: score = (EM - 0.25)/0.75.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from phase_f.data.types import Item, ItemList
from phase_f.detectors.base import Detector


class TSGuessingDetector(Detector):
    nick = "ts_guessing"

    def __init__(self, max_new_tokens: int = 64) -> None:
        self.max_new_tokens = max_new_tokens

    def is_model_required(self) -> bool:
        return True

    @staticmethod
    def _build_prompt(item: Item, masked_idx: int) -> tuple[str, str]:
        """Return (prompt, target) for the TS-Guessing probe.

        We mask the choice at `masked_idx` and ask the model to recover it.
        The target is the literal text of the masked choice.
        """
        choices_display = []
        for i, c in enumerate(item.choices):
            letter = "ABCD"[i]
            if i == masked_idx:
                choices_display.append(f"{letter}. [MASKED]")
            else:
                choices_display.append(f"{letter}. {c}")
        prompt = (
            f"The following is a multiple-choice question about "
            f"{item.subject.replace('_', ' ')}. One option has been masked. "
            f"Recover the masked option verbatim.\n\n"
            f"{item.question}\n"
            + "\n".join(choices_display)
            + f"\nThe masked option {('ABCD')[masked_idx]} is:"
        )
        return prompt, item.choices[masked_idx]

    @staticmethod
    def _normalise(text: str) -> str:
        return " ".join(text.lower().split())

    def score(self, items: ItemList, *, model_handle: Any | None = None) -> np.ndarray:
        if model_handle is None:
            raise ValueError("TSGuessingDetector requires a (tokenizer, model) handle")
        tokenizer, model = model_handle

        # Mask the CORRECT answer (highest signal for memorisation)
        prompts_and_targets = [self._build_prompt(item, item.answer) for item in items]

        scores = np.zeros(len(items), dtype=float)
        for i, (prompt, target) in enumerate(prompts_and_targets):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with model.no_sync() if hasattr(model, "no_sync") else _nullctx():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            # Exact-match flag (Reading A); Reading B rescales the rate.
            scores[i] = 1.0 if self._normalise(target) in self._normalise(gen) else 0.0

        # Reading B: rescale per-item by the 25% chance baseline
        return np.clip((scores - 0.25) / 0.75, 0.0, 1.0)


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False
