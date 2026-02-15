from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class EmbeddingCache(BaseModel):
    cache_dir: Path
    model_name: str

    model_config = {"arbitrary_types_allowed": True}

    def _get_cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _get_cache_path(self, text: str) -> Path:
        key = self._get_cache_key(text)
        return self.cache_dir / f"{key}.npy"

    def get(self, text: str) -> NDArray[np.float32] | None:
        path = self._get_cache_path(text)
        if path.exists():
            return np.load(path)
        return None

    def put(self, text: str, embedding: NDArray[np.float32]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._get_cache_path(text)
        np.save(path, embedding)

    def get_many(
        self, texts: list[str]
    ) -> tuple[list[NDArray[np.float32] | None], list[int]]:
        results: list[NDArray[np.float32] | None] = []
        missing: list[int] = []

        for i, text in enumerate(texts):
            cached = self.get(text)
            results.append(cached)
            if cached is None:
                missing.append(i)

        return results, missing


class Embedder:
    def __init__(
        self,
        cache_dir: Path,
        model_name: str = "jinaai/jina-embeddings-v3",
        default_batch_size: int = 128,
    ):
        self.model_name = model_name
        self.default_batch_size = default_batch_size
        self.cache = EmbeddingCache(cache_dir=cache_dir, model_name=model_name)
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
        return self._model

    def embed(self, text: str) -> NDArray[np.float32]:
        cached = self.cache.get(text)
        if cached is not None:
            return cached

        embedding = self.model.encode(text, convert_to_numpy=True)
        embedding = np.asarray(embedding, dtype=np.float32)
        self.cache.put(text, embedding)
        return embedding

    def embed_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[NDArray[np.float32]]:
        if batch_size is None:
            batch_size = self.default_batch_size

        cached_results, missing_indices = self.cache.get_many(texts)

        if not missing_indices:
            return [r for r in cached_results if r is not None]

        missing_texts = [texts[i] for i in missing_indices]
        new_embeddings = self.model.encode(
            missing_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        for i, idx in enumerate(missing_indices):
            emb = np.asarray(new_embeddings[i], dtype=np.float32)
            self.cache.put(texts[idx], emb)
            cached_results[idx] = emb

        return [r for r in cached_results if r is not None]


def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
