from __future__ import annotations

import time

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    QueryRequest,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
)

from src.keyword_translator import KeywordTranslator
from src.logging_utils import get_logger
from src.models import Candidate, Sentence
from src.synonym_expander import build_synonym_map

from .pairwise import compute_jaccard_similarity, tokenize_for_jaccard


class CrossLingualRetriever:
    def __init__(
        self,
        translator: KeywordTranslator,
        host: str = "localhost",
        port: int = 6333,
    ):
        self.translator = translator
        self.client = QdrantClient(host=host, port=port)
        self.keyword_to_idx_en: dict[str, int] = {}
        self.keyword_to_idx_de: dict[str, int] = {}
        self.sentence_map: dict[int, Sentence] = {}

    def _build_vocabulary(
        self, sentences: list[Sentence], language: str
    ) -> dict[str, int]:
        all_keywords: set[str] = set()
        for s in sentences:
            if s.language == language:
                all_keywords.update(s.keywords)
        return {kw: idx for idx, kw in enumerate(sorted(all_keywords))}

    def _keywords_to_sparse_vector(
        self, keywords: list[str], vocab: dict[str, int]
    ) -> SparseVector:
        seen_indices: set[int] = set()
        indices = []
        values = []
        for kw in keywords:
            if kw in vocab:
                idx = vocab[kw]
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    indices.append(idx)
                    values.append(1.0)
        return SparseVector(indices=indices, values=values)

    def _create_collection(self, name: str) -> None:
        if self.client.collection_exists(name):
            self.client.delete_collection(name)

        self.client.create_collection(
            collection_name=name,
            vectors_config={},
            sparse_vectors_config={
                "keywords": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                )
            },
        )

    def index_by_language(self, sentences: list[Sentence]) -> None:
        log = get_logger()
        t0 = time.perf_counter()

        en_sentences = [s for s in sentences if s.language == "en"]
        de_sentences = [s for s in sentences if s.language == "de"]

        self.keyword_to_idx_en = self._build_vocabulary(sentences, "en")
        self.keyword_to_idx_de = self._build_vocabulary(sentences, "de")

        log.info(
            f"Vocabularies: EN={len(self.keyword_to_idx_en)}, DE={len(self.keyword_to_idx_de)}"
        )

        self.sentence_map = {s.id: s for s in sentences}

        self._create_collection("sentences_en")
        self._index_sentences("sentences_en", en_sentences, self.keyword_to_idx_en)

        self._create_collection("sentences_de")
        self._index_sentences("sentences_de", de_sentences, self.keyword_to_idx_de)

        log.info(
            f"Indexed {len(en_sentences)} EN + {len(de_sentences)} DE sentences "
            f"in {time.perf_counter() - t0:.2f}s"
        )

    def _index_sentences(
        self,
        collection: str,
        sentences: list[Sentence],
        vocab: dict[str, int],
    ) -> None:
        batch_size = 1000
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            points = []
            for s in batch:
                sparse_vec = self._keywords_to_sparse_vector(s.keywords, vocab)
                if sparse_vec.indices:
                    points.append(
                        PointStruct(
                            id=s.id,
                            vector={"keywords": sparse_vec},
                            payload={"language": s.language},
                        )
                    )
            if points:
                self.client.upsert(collection_name=collection, points=points)

    def retrieve_cross_lingual(
        self,
        sentences: list[Sentence],
        top_k: int = 50,
        min_shared_keywords: int = 2,
        query_batch_size: int = 256,
        cross_lingual_only: bool = False,
        enable_synonyms: bool = True,
        enable_bigram_permutations: bool = False,
    ) -> list[Candidate]:
        log = get_logger()
        t0 = time.perf_counter()

        en_keywords: set[str] = set()
        de_keywords: set[str] = set()
        for s in sentences:
            if s.language == "en":
                en_keywords.update(s.keywords)
            else:
                de_keywords.update(s.keywords)

        log.info(f"Original keywords: {len(en_keywords)} EN + {len(de_keywords)} DE")

        log.info(f"Translating {len(en_keywords)} EN + {len(de_keywords)} DE keywords")
        en_to_de = self.translator.translate_batch(
            list(en_keywords), "en-de", enable_bigram_permutations
        )
        de_to_en = self.translator.translate_batch(
            list(de_keywords), "de-en", enable_bigram_permutations
        )

        if enable_synonyms:
            log.info("Building synonym maps for source keywords...")
            en_syn_map = build_synonym_map(list(en_keywords), "en")
            de_syn_map = build_synonym_map(list(de_keywords), "de")
            log.info(
                f"Synonym maps: EN {len(en_syn_map)} keywords, DE {len(de_syn_map)} keywords"
            )
        else:
            en_syn_map = {kw: [kw] for kw in en_keywords}
            de_syn_map = {kw: [kw] for kw in de_keywords}

        query_vectors: list[tuple[Sentence, SparseVector, str]] = []
        for s in sentences:
            if s.language == "en":
                translated_de: list[str] = []
                for kw in s.keywords:
                    translated_de.extend(en_to_de.get(kw, [kw]))
                sparse_vec_de = self._keywords_to_sparse_vector(
                    translated_de, self.keyword_to_idx_de
                )
                if sparse_vec_de.indices:
                    query_vectors.append((s, sparse_vec_de, "sentences_de"))

                en_with_syns: list[str] = []
                for kw in s.keywords:
                    en_with_syns.extend(en_syn_map.get(kw, [kw]))
                sparse_vec_en = self._keywords_to_sparse_vector(
                    en_with_syns, self.keyword_to_idx_en
                )
                if sparse_vec_en.indices:
                    query_vectors.append((s, sparse_vec_en, "sentences_en"))
            else:
                translated_en: list[str] = []
                for kw in s.keywords:
                    translated_en.extend(de_to_en.get(kw, [kw]))
                sparse_vec_en = self._keywords_to_sparse_vector(
                    translated_en, self.keyword_to_idx_en
                )
                if sparse_vec_en.indices:
                    query_vectors.append((s, sparse_vec_en, "sentences_en"))

                de_with_syns: list[str] = []
                for kw in s.keywords:
                    de_with_syns.extend(de_syn_map.get(kw, [kw]))
                sparse_vec_de = self._keywords_to_sparse_vector(
                    de_with_syns, self.keyword_to_idx_de
                )
                if sparse_vec_de.indices:
                    query_vectors.append((s, sparse_vec_de, "sentences_de"))

        log.info(
            f"Built {len(query_vectors)} query vectors (both same-language and cross-lingual)"
        )

        seen_pairs: set[tuple[int, int]] = set()
        candidates: list[Candidate] = []
        score_threshold = (
            float(min_shared_keywords) if min_shared_keywords > 1 else None
        )

        en_queries = [(s, v) for s, v, c in query_vectors if c == "sentences_en"]
        de_queries = [(s, v) for s, v, c in query_vectors if c == "sentences_de"]

        for target_collection, queries in [
            ("sentences_en", en_queries),
            ("sentences_de", de_queries),
        ]:
            if not queries:
                continue

            num_batches = (len(queries) + query_batch_size - 1) // query_batch_size
            for batch_idx in range(num_batches):
                start = batch_idx * query_batch_size
                end = min(start + query_batch_size, len(queries))
                batch = queries[start:end]

                requests = [
                    QueryRequest(
                        query=sparse_vec,
                        using="keywords",
                        limit=top_k,
                        score_threshold=score_threshold,
                        with_payload=False,
                    )
                    for _, sparse_vec in batch
                ]

                batch_results = self.client.query_batch_points(
                    collection_name=target_collection,
                    requests=requests,
                )

                for (sentence, _), result in zip(batch, batch_results):
                    tokens_a = tokenize_for_jaccard(sentence.text)

                    for point in result.points:
                        other_id = int(point.id)
                        if other_id == sentence.id:
                            continue

                        pair_key = (
                            min(sentence.id, other_id),
                            max(sentence.id, other_id),
                        )
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

                        if is_cross_lingual:
                            if sentence.language == "en":
                                translated_kw: set[str] = set()
                                for kw in sentence.keywords:
                                    translated_kw.update(en_to_de.get(kw, [kw]))
                                shared = list(translated_kw & set(other.keywords))
                            else:
                                translated_kw = set()
                                for kw in sentence.keywords:
                                    translated_kw.update(de_to_en.get(kw, [kw]))
                                shared = list(translated_kw & set(other.keywords))
                        else:
                            shared = list(set(sentence.keywords) & set(other.keywords))

                        candidates.append(
                            Candidate(
                                sentence_a=sentence,
                                sentence_b=other,
                                shared_keywords=shared,
                                jaccard_similarity=jaccard,
                            )
                        )

        log.info(
            f"Cross-lingual retrieval: {len(candidates)} pairs in {time.perf_counter() - t0:.2f}s"
        )
        return candidates
