from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    QueryRequest,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)

from src.config import DenseRetrievalConfig
from src.logging_utils import get_logger
from src.models import ScoredPair, Sentence

from .pairwise import compute_jaccard_similarity, tokenize_for_jaccard

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DenseEmbedderProtocol(Protocol):
    def embed_batch(
        self, texts: list[str], show_progress: bool = True
    ) -> list[NDArray[np.float32]]: ...


class DenseANNRetriever:
    def __init__(
        self,
        config: DenseRetrievalConfig,
        embedder: DenseEmbedderProtocol,
        host: str = "localhost",
        port: int = 6333,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.client = QdrantClient(host=host, port=port)
        self.sentence_map: dict[int, Sentence] = {}

    def index_sentences(self, sentences: list[Sentence]) -> None:
        log = get_logger()
        t0 = time.perf_counter()

        self.sentence_map = {s.id: s for s in sentences}

        if self.client.collection_exists(self.config.collection_name):
            self.client.delete_collection(self.config.collection_name)

        quantization_config = None
        if self.config.use_quantization:
            quantization_config = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            )

        self.client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=VectorParams(
                size=self.config.embedding.dimension,
                distance=Distance.COSINE,
                on_disk=self.config.on_disk,
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    on_disk=self.config.on_disk,
                ),
            ),
            quantization_config=quantization_config,
        )

        log.info(
            f"Created collection '{self.config.collection_name}' (dim={self.config.embedding.dimension})"
        )

        texts = [s.text for s in sentences]
        t_embed = time.perf_counter()
        embeddings = self.embedder.embed_batch(texts)
        log.info(f"Embedding phase: {time.perf_counter() - t_embed:.2f}s")

        t_index = time.perf_counter()
        batch_size = 1000
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            points = [
                PointStruct(
                    id=s.id,
                    vector=emb.tolist(),
                    payload={"language": s.language},
                )
                for s, emb in zip(batch_sentences, batch_embeddings)
            ]

            self.client.upsert(
                collection_name=self.config.collection_name, points=points
            )

            if (i + batch_size) % 10000 == 0:
                log.info(f"  Indexed {i + batch_size:,} / {len(sentences):,} sentences")

        log.info(f"Indexing phase: {time.perf_counter() - t_index:.2f}s")
        log.info(
            f"Total index time: {time.perf_counter() - t0:.2f}s for {len(sentences)} sentences"
        )

    def retrieve_candidates(
        self,
        sentences: list[Sentence],
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        cross_lingual_only: bool | None = None,
        query_batch_size: int = 256,
    ) -> list[ScoredPair]:
        log = get_logger()
        t0 = time.perf_counter()

        top_k = top_k or self.config.top_k
        similarity_threshold = similarity_threshold or self.config.similarity_threshold
        cross_lingual_only = (
            cross_lingual_only
            if cross_lingual_only is not None
            else self.config.cross_lingual_only
        )

        texts = [s.text for s in sentences]
        t_embed = time.perf_counter()
        embeddings = self.embedder.embed_batch(texts, show_progress=False)
        log.info(f"Query embedding: {time.perf_counter() - t_embed:.2f}s")

        seen_pairs: set[tuple[int, int]] = set()
        results: list[ScoredPair] = []

        t_search = time.perf_counter()
        num_batches = (len(sentences) + query_batch_size - 1) // query_batch_size

        for batch_idx in range(num_batches):
            start = batch_idx * query_batch_size
            end = min(start + query_batch_size, len(sentences))
            batch_sentences = sentences[start:end]
            batch_embeddings = embeddings[start:end]

            requests = [
                QueryRequest(
                    query=emb.tolist(),
                    limit=top_k + 1,
                    score_threshold=similarity_threshold,
                    with_payload=True,
                )
                for emb in batch_embeddings
            ]

            batch_results = self.client.query_batch_points(
                collection_name=self.config.collection_name,
                requests=requests,
            )

            for sentence, result in zip(batch_sentences, batch_results):
                tokens_a = tokenize_for_jaccard(sentence.text)

                for point in result.points:
                    other_id = int(point.id)
                    if other_id == sentence.id:
                        continue

                    pair_key = (min(sentence.id, other_id), max(sentence.id, other_id))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    other = self.sentence_map.get(other_id)
                    if other is None:
                        continue

                    is_cross_lingual = sentence.language != other.language
                    if cross_lingual_only and not is_cross_lingual:
                        continue

                    tokens_b = tokenize_for_jaccard(other.text)
                    jaccard = compute_jaccard_similarity(tokens_a, tokens_b)
                    shared = list(set(sentence.keywords) & set(other.keywords))

                    results.append(
                        ScoredPair(
                            sentence_a=sentence,
                            sentence_b=other,
                            semantic_similarity=float(point.score),
                            jaccard_similarity=jaccard,
                            shared_keywords=shared,
                        )
                    )

            if (batch_idx + 1) % 50 == 0 or batch_idx == num_batches - 1:
                log.info(f"  Queried {end:,} / {len(sentences):,} sentences")

        log.info(f"Search phase: {time.perf_counter() - t_search:.2f}s")
        log.info(
            f"Dense ANN retrieval: {len(results)} pairs in {time.perf_counter() - t0:.2f}s"
        )

        if cross_lingual_only:
            cross_count = len(results)
            log.info(f"Cross-lingual pairs: {cross_count}")
        else:
            en_en = sum(
                1
                for r in results
                if r.sentence_a.language == "en" and r.sentence_b.language == "en"
            )
            de_de = sum(
                1
                for r in results
                if r.sentence_a.language == "de" and r.sentence_b.language == "de"
            )
            cross = sum(
                1 for r in results if r.sentence_a.language != r.sentence_b.language
            )
            log.info(f"Pairs by type: EN-EN={en_en}, DE-DE={de_de}, Cross={cross}")

        return results
