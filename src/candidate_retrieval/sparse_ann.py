from __future__ import annotations

import time

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
)

from src.legal_entities import (
    LegalEntities,
    extract_legal_entities_batch,
    extract_legal_entities_batch_fast,
)
from src.logging_utils import get_logger
from src.models import Candidate, Sentence
from src.synonym_expander import build_synonym_map

from .pairwise import compute_jaccard_similarity, tokenize_for_jaccard

COLLECTION_NAME = "sentences"


class SparseANNRetriever:
    def __init__(
        self, host: str = "localhost", port: int = 6333, use_bm25: bool = True
    ):
        self.client = QdrantClient(host=host, port=port)
        self.keyword_to_idx: dict[str, int] = {}
        self.sentences: list[Sentence] = []
        self.sentence_map: dict[int, Sentence] = {}
        self.use_bm25 = use_bm25
        self.bm25_model: SparseTextEmbedding | None = None

        if use_bm25:
            self.bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def _build_vocabulary(self, sentences: list[Sentence]) -> None:
        all_keywords: set[str] = set()
        for s in sentences:
            all_keywords.update(s.keywords)
        self.keyword_to_idx = {kw: idx for idx, kw in enumerate(sorted(all_keywords))}

    def _keywords_to_sparse_vector(self, keywords: list[str]) -> SparseVector:
        seen_indices: set[int] = set()
        indices = []
        values = []
        for kw in keywords:
            if kw in self.keyword_to_idx:
                idx = self.keyword_to_idx[kw]
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    indices.append(idx)
                    values.append(1.0)
        return SparseVector(indices=indices, values=values)

    def index_sentences(self, sentences: list[Sentence]) -> None:
        log = get_logger()
        self.sentences = sentences
        self.sentence_map = {s.id: s for s in sentences}

        t0 = time.perf_counter()
        if not self.use_bm25:
            self._build_vocabulary(sentences)
            log.info(f"Vocabulary: {len(self.keyword_to_idx)} unique keywords")

        if self.client.collection_exists(COLLECTION_NAME):
            self.client.delete_collection(COLLECTION_NAME)

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={},
            sparse_vectors_config={
                "keywords": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                )
            },
        )

        batch_size = 1000
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            points = []

            if self.use_bm25 and self.bm25_model is not None:
                texts = [" ".join(s.keywords) for s in batch]
                embeddings_list = list(self.bm25_model.embed(texts))

                for s, embedding in zip(batch, embeddings_list):
                    sparse_vec = SparseVector(
                        indices=embedding.indices.tolist(),
                        values=embedding.values.tolist(),
                    )
                    points.append(
                        PointStruct(
                            id=s.id,
                            vector={"keywords": sparse_vec},
                            payload={"language": s.language},
                        )
                    )
            else:
                for s in batch:
                    sparse_vec = self._keywords_to_sparse_vector(s.keywords)
                    if sparse_vec.indices:
                        points.append(
                            PointStruct(
                                id=s.id,
                                vector={"keywords": sparse_vec},
                                payload={"language": s.language},
                            )
                        )

            if points:
                self.client.upsert(collection_name=COLLECTION_NAME, points=points)

        log.info(
            f"Indexed {len(sentences)} sentences in {time.perf_counter() - t0:.2f}s"
        )

    def retrieve_candidates(
        self,
        sentences: list[Sentence],
        top_k: int = 100,
        filter_conflicting_entities: bool = True,
        fast: bool = False,
        min_shared_keywords: int = 2,
        max_jaccard_similarity: float = 1.0,
        query_batch_size: int = 256,
        enable_synonyms: bool = True,
        min_similarity_score: float = 0.0,
        min_candidates_per_sentence: int = 1,
    ) -> list[Candidate]:
        log = get_logger()

        entities_map: dict[int, LegalEntities] = {}
        if filter_conflicting_entities:
            t0 = time.perf_counter()
            texts = [s.text for s in sentences]
            if fast:
                entities_list = extract_legal_entities_batch_fast(texts)
            else:
                entities_list = extract_legal_entities_batch(texts)
            for s, entities in zip(sentences, entities_list):
                entities_map[s.id] = entities
            log.info(f"Entity extraction: {time.perf_counter() - t0:.2f}s")

        syn_map: dict[str, list[str]] = {}
        if enable_synonyms:
            language = sentences[0].language if sentences else "en"

            if self.use_bm25 and not self.keyword_to_idx:
                all_keywords: set[str] = set()
                for s in self.sentences:
                    all_keywords.update(s.keywords)
                keywords_list = list(all_keywords)
            else:
                keywords_list = list(self.keyword_to_idx.keys())

            syn_map = build_synonym_map(keywords_list, language)
            log.info(f"Built synonym map for {len(syn_map)} keywords ({language})")

        t0 = time.perf_counter()
        sentence_vectors: list[tuple[Sentence, SparseVector]] = []

        if self.use_bm25 and self.bm25_model is not None:
            query_texts: list[str] = []
            sentence_order: list[Sentence] = []

            for sentence in sentences:
                if enable_synonyms and syn_map:
                    keywords_with_syns: list[str] = []
                    for kw in sentence.keywords:
                        keywords_with_syns.extend(syn_map.get(kw, [kw]))
                    query_text = " ".join(keywords_with_syns)
                else:
                    query_text = " ".join(sentence.keywords)

                if query_text.strip():
                    query_texts.append(query_text)
                    sentence_order.append(sentence)

            embeddings_list = list(self.bm25_model.embed(query_texts))
            for sentence, embedding in zip(sentence_order, embeddings_list):
                sparse_vec = SparseVector(
                    indices=embedding.indices.tolist(),
                    values=embedding.values.tolist(),
                )
                sentence_vectors.append((sentence, sparse_vec))

            log.info(f"Built {len(sentence_vectors)} BM25 query vectors")
        else:
            for sentence in sentences:
                if enable_synonyms and syn_map:
                    keywords_with_syns = []
                    for kw in sentence.keywords:
                        keywords_with_syns.extend(syn_map.get(kw, [kw]))
                    sparse_vec = self._keywords_to_sparse_vector(keywords_with_syns)
                else:
                    sparse_vec = self._keywords_to_sparse_vector(sentence.keywords)

                if sparse_vec.indices:
                    sentence_vectors.append((sentence, sparse_vec))

            log.info(f"Built {len(sentence_vectors)} query vectors")

        seen_pairs: set[tuple[int, int]] = set()
        candidates: list[Candidate] = []
        entity_conflicts_filtered = 0
        jaccard_filtered = 0

        if self.use_bm25:
            score_threshold = (
                min_similarity_score if min_similarity_score > 0.0 else None
            )
        else:
            score_threshold = (
                float(min_shared_keywords) if min_shared_keywords > 1 else None
            )

        num_batches = (len(sentence_vectors) + query_batch_size - 1) // query_batch_size

        for batch_idx in range(num_batches):
            start = batch_idx * query_batch_size
            end = min(start + query_batch_size, len(sentence_vectors))
            batch = sentence_vectors[start:end]

            from qdrant_client.models import QueryRequest

            requests = [
                QueryRequest(
                    query=sparse_vec,
                    using="keywords",
                    limit=top_k + 1,
                    score_threshold=score_threshold,
                    with_payload=False,
                )
                for _, sparse_vec in batch
            ]

            batch_results = self.client.query_batch_points(
                collection_name=COLLECTION_NAME,
                requests=requests,
            )

            for (sentence, _), result in zip(batch, batch_results):
                tokens_a = tokenize_for_jaccard(sentence.text)
                entities_a = entities_map.get(sentence.id)

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

                    if filter_conflicting_entities and entities_a is not None:
                        entities_b = entities_map.get(other_id)
                        if entities_b is not None and entities_a.conflicts_with(
                            entities_b
                        ):
                            entity_conflicts_filtered += 1
                            continue

                    tokens_b = tokenize_for_jaccard(other.text)
                    jaccard = compute_jaccard_similarity(tokens_a, tokens_b)

                    if jaccard > max_jaccard_similarity:
                        jaccard_filtered += 1
                        continue

                    shared = list(set(sentence.keywords) & set(other.keywords))

                    candidates.append(
                        Candidate(
                            sentence_a=sentence,
                            sentence_b=other,
                            shared_keywords=shared,
                            jaccard_similarity=jaccard,
                        )
                    )

            if (batch_idx + 1) % 100 == 0 or batch_idx == num_batches - 1:
                log.info(f"  Queried {end:,} / {len(sentence_vectors):,} sentences")

        log.info(f"Sparse ANN retrieval: {time.perf_counter() - t0:.2f}s")
        if filter_conflicting_entities and entity_conflicts_filtered > 0:
            log.info(
                f"Filtered {entity_conflicts_filtered} pairs with conflicting entities"
            )
        if jaccard_filtered > 0:
            log.info(
                f"Filtered {jaccard_filtered} pairs with Jaccard > {max_jaccard_similarity} (low lexical diversity)"
            )

        return candidates
