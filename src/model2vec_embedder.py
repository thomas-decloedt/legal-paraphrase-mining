from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from model2vec import StaticModel

from src.config import DenseEmbeddingConfig
from src.embedder import EmbeddingCache
from src.logging_utils import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Model2VecEmbedder:
    def __init__(
        self,
        config: DenseEmbeddingConfig,
        cache_dir: Path,
    ) -> None:
        self.config = config
        cache_subdir = (
            cache_dir / config.model_name.replace("/", "_") / f"dim{config.dimension}"
        )
        self.cache = EmbeddingCache(
            cache_dir=cache_subdir,
            model_name=config.model_name,
        )
        self._model: StaticModel | None = None

    @property
    def model(self) -> StaticModel:
        if self._model is None:
            log = get_logger()
            t0 = time.perf_counter()
            log.info(f"Loading Model2Vec: {self.config.model_name}")

            if self.config.dimension < 256:
                log.info(f"Reducing dimensions to {self.config.dimension}")
                self._model = StaticModel.from_pretrained(
                    self.config.model_name,
                    dimensionality=self.config.dimension,
                )
            else:
                self._model = StaticModel.from_pretrained(self.config.model_name)

            log.info(f"Model loaded in {time.perf_counter() - t0:.2f}s")

        return self._model

    def _prepare_text(self, text: str) -> str:
        if self.config.instruction:
            return f"Instruct: {self.config.instruction}\nQuery: {text}"
        return text

    def embed_batch(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> list[NDArray[np.float32]]:
        log = get_logger()
        t0 = time.perf_counter()

        prepared_texts = [self._prepare_text(t) for t in texts]
        cached_results, missing_indices = self.cache.get_many(prepared_texts)

        if not missing_indices:
            log.info(
                f"All {len(texts)} embeddings from cache ({time.perf_counter() - t0:.2f}s)"
            )
            return [r for r in cached_results if r is not None]

        log.info(
            f"Embedding {len(missing_indices)}/{len(texts)} texts (cache hit: {len(texts) - len(missing_indices)})"
        )

        missing_texts = [prepared_texts[i] for i in missing_indices]
        t_encode = time.perf_counter()

        new_embeddings = self.model.encode(missing_texts)

        if self.config.normalize:
            norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
            new_embeddings = new_embeddings / (norms + 1e-8)

        log.info(f"Encoding took {time.perf_counter() - t_encode:.2f}s")

        for i, idx in enumerate(missing_indices):
            emb = np.asarray(new_embeddings[i], dtype=np.float32)
            self.cache.put(prepared_texts[idx], emb)
            cached_results[idx] = emb

        log.info(f"Total embedding time: {time.perf_counter() - t0:.2f}s")
        return [r for r in cached_results if r is not None]
