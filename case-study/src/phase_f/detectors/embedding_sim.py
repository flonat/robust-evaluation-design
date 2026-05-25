"""Embedding-similarity detector (sentence-transformers).

Text-only: embed benchmark items and a reference corpus into a shared dense
vector space, then for each benchmark item return the max cosine similarity
to any reference document. High similarity to the training mixture is a
contamination signal that's robust to surface paraphrase (unlike n-gram LCS).
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from phase_f.data.types import ItemList
from phase_f.detectors.base import Detector


class EmbeddingSimilarityDetector(Detector):
    nick = "embedding_sim"

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        reference_corpus: Iterable[str] | None = None,
        encoder_model: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.reference_corpus = tuple(reference_corpus or [])
        self.encoder_model_name = encoder_model or self.DEFAULT_MODEL
        self.batch_size = batch_size
        self._encoder = None
        self._reference_emb: np.ndarray | None = None

    def _ensure_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.encoder_model_name)
            if self.reference_corpus:
                self._reference_emb = self._encoder.encode(
                    list(self.reference_corpus),
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )

    def score(self, items: ItemList, *, model_handle: Any | None = None) -> np.ndarray:
        if not self.reference_corpus:
            return np.zeros(len(items), dtype=float)
        self._ensure_encoder()
        assert self._encoder is not None and self._reference_emb is not None

        texts = [item.question for item in items]
        item_emb = self._encoder.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # Cosine similarity since both sides are L2-normalised
        sim_matrix = item_emb @ self._reference_emb.T  # (n_items, n_refs)
        # Max over reference corpus per item
        max_sim = sim_matrix.max(axis=1)
        # Map [-1, 1] → [0, 1]
        return np.clip((max_sim + 1.0) / 2.0, 0.0, 1.0)
