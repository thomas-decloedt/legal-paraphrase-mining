#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

from src.candidate_retrieval import (
    CrossLingualRetriever,
    DenseANNRetriever,
    SparseANNRetriever,
    retrieve_candidates,
    retrieve_candidates_clustered,
)
from src.config import DEFAULT_CONFIG, PipelineConfig
from src.data_loader import LoadConfig, load_sentences
from src.embedder import Embedder
from src.inverted_index import InvertedIndex
from src.keyword_extractor import extract_keywords_batch, extract_keywords_batch_fast
from src.keyword_translator import KeywordTranslator
from src.logging_utils import setup_logging, timed_section
from src.models import Candidate, ScoredPair, Sentence
from src.scorer import export_to_csv, export_to_json, rank_pairs, score_candidates

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"


def run_pipeline(config: PipelineConfig = DEFAULT_CONFIG) -> None:
    log = setup_logging(RESULTS_DIR / "run.log")

    approach: str = config.retrieval_method
    if config.cross_lingual.enabled:
        approach = "cross-lingual+" + approach
    if config.fast:
        approach += "+fast"

    log.info("=" * 60)
    log.info(f"Paraphrase Finding Pipeline ({approach})")
    log.info("=" * 60)
    log.info(
        f"Config: EN={config.english.num_sentences}, DE={config.german.num_sentences}"
    )
    log.info(
        f"Thresholds: EN kw={config.english.min_shared_keywords} sem={config.english.semantic_threshold}, "
        f"DE kw={config.german.min_shared_keywords} sem={config.german.semantic_threshold}"
    )
    if config.cross_lingual.enabled:
        log.info(
            f"Cross-lingual: kw={config.cross_lingual.min_shared_keywords} "
            f"sem={config.cross_lingual.semantic_threshold}"
        )

    with timed_section("Loading sentences"):
        load_both = config.german.num_sentences > 0 and config.english.num_sentences > 0

        if load_both:
            en_config = LoadConfig(
                num_sentences_per_language=config.english.num_sentences,
                languages=["en"],
                min_sentence_length=config.min_sentence_length,
            )
            de_config = LoadConfig(
                num_sentences_per_language=config.german.num_sentences,
                languages=["de"],
                min_sentence_length=config.min_sentence_length,
            )
            en_cache = (
                CACHE_DIR / f"sentences_en_{config.english.num_sentences}.parquet"
            )
            de_cache = CACHE_DIR / f"sentences_de_{config.german.num_sentences}.parquet"

            df_en = load_sentences(en_config, cache_path=en_cache)
            df_de = load_sentences(de_config, cache_path=de_cache)

            import polars as pl

            id_offset = int(df_en["id"].max()) + 1  # type: ignore[arg-type]
            df_de = df_de.with_columns(pl.col("id") + id_offset)

            df = pl.concat([df_en, df_de])
            log.info(f"Loaded {len(df_en)} EN + {len(df_de)} DE = {len(df)} sentences")
        else:
            languages = ["de"] if config.german.num_sentences > 0 else ["en"]
            num_sentences = (
                config.german.num_sentences
                if config.german.num_sentences > 0
                else config.english.num_sentences
            )
            lang_code = languages[0]
            cache_name = f"sentences_{lang_code}_{num_sentences}.parquet"
            load_config = LoadConfig(
                num_sentences_per_language=num_sentences,
                languages=languages,
                min_sentence_length=config.min_sentence_length,
            )
            df = load_sentences(load_config, cache_path=CACHE_DIR / cache_name)
            log.info(f"Loaded {len(df)} sentences ({lang_code.upper()})")

    with timed_section("Extracting keywords"):
        texts = df["text"].to_list()
        languages = df["language"].to_list()
        if config.fast:
            keywords_list = extract_keywords_batch_fast(texts, languages)
        else:
            keywords_list = extract_keywords_batch(texts, languages)

        sentences: list[Sentence] = []
        for i, row in enumerate(df.iter_rows(named=True)):
            sentences.append(
                Sentence(
                    id=row["id"],
                    text=row["text"],
                    language=row["language"],
                    keywords=keywords_list[i],
                )
            )
        avg_kw = sum(len(s.keywords) for s in sentences) / len(sentences)
        log.info(f"Average keywords per sentence: {avg_kw:.1f}")

    all_candidates: list[Candidate] = []
    dense_scored: list[ScoredPair] = []

    if config.retrieval_method == "two-stage":
        with timed_section("Retrieving candidates (two-stage)"):
            from src.candidate_retrieval.two_stage import TwoStageRetriever
            from src.neural_translator import NeuralTranslator

            neural_translator = NeuralTranslator(cache_dir=CACHE_DIR / "opus_mt")
            sparse_retriever = SparseANNRetriever(
                use_bm25=config.sparse_retrieval.use_bm25
            )
            two_stage = TwoStageRetriever(neural_translator, sparse_retriever)

            candidates = two_stage.retrieve_cross_lingual(sentences, top_k=config.top_k)
            all_candidates.extend(candidates)
            log.info(f"Found {len(candidates)} cross-lingual pairs")

    elif config.retrieval_method == "dense-ann":
        with timed_section("Retrieving candidates (dense-ann)"):
            from src.candidate_retrieval.dense_ann import DenseEmbedderProtocol
            from src.dense_embedder import DenseEmbedder
            from src.model2vec_embedder import Model2VecEmbedder

            dense_embedder: DenseEmbedderProtocol
            if config.dense.embedding.embedding_backend == "model2vec":
                dense_embedder = Model2VecEmbedder(
                    config=config.dense.embedding,
                    cache_dir=CACHE_DIR / "dense_embeddings",
                )
            else:
                dense_embedder = DenseEmbedder(
                    config=config.dense.embedding,
                    cache_dir=CACHE_DIR / "dense_embeddings",
                )
            dense_retriever = DenseANNRetriever(
                config=config.dense,
                embedder=dense_embedder,
            )
            dense_retriever.index_sentences(sentences)
            dense_scored = dense_retriever.retrieve_candidates(
                sentences,
                top_k=config.top_k,
                cross_lingual_only=config.dense.cross_lingual_only,
            )
            log.info(f"Found {len(dense_scored)} scored pairs (no re-scoring needed)")

    elif config.cross_lingual.enabled:
        with timed_section("Retrieving candidates (cross-lingual)"):
            translator = KeywordTranslator()
            retriever = CrossLingualRetriever(translator)
            retriever.index_by_language(sentences)

            min_kw = min(
                config.english.min_shared_keywords,
                config.german.min_shared_keywords,
                config.cross_lingual.min_shared_keywords,
            )
            candidates = retriever.retrieve_cross_lingual(
                sentences,
                top_k=config.top_k,
                min_shared_keywords=min_kw,
                enable_synonyms=config.synonyms.enabled,
                enable_bigram_permutations=config.cross_lingual.enable_bigram_permutations,
            )
            all_candidates.extend(candidates)
            log.info(f"Found {len(candidates)} candidate pairs")

    elif config.retrieval_method == "sparse-ann":
        with timed_section("Retrieving candidates (sparse-ann)"):
            sparse_retriever = SparseANNRetriever(
                use_bm25=config.sparse_retrieval.use_bm25
            )
            sparse_retriever.index_sentences(sentences)
            candidates = sparse_retriever.retrieve_candidates(
                sentences,
                top_k=config.top_k,
                fast=config.fast,
                min_shared_keywords=config.english.min_shared_keywords,
                max_jaccard_similarity=config.english.max_jaccard_similarity,
                enable_synonyms=config.synonyms.enabled,
                min_similarity_score=config.sparse_retrieval.min_similarity_score,
            )
            all_candidates.extend(candidates)
            log.info(f"Found {len(candidates)} candidate pairs")

    elif config.retrieval_method == "clustering":
        with timed_section("Retrieving candidates (clustering)"):
            candidates = retrieve_candidates_clustered(
                sentences, threshold=0.1, num_perm=128
            )
            all_candidates.extend(candidates)
            log.info(f"Found {len(candidates)} candidate pairs")

    else:
        with timed_section("Building inverted index"):
            index = InvertedIndex.build(sentences)
            log.info(f"Index size: {len(index.index)} unique keywords")

        with timed_section("Retrieving candidates (pairwise)"):
            candidates = retrieve_candidates(
                sentences, index, min_shared_keywords=config.english.min_shared_keywords
            )
            all_candidates.extend(candidates)
            log.info(f"Found {len(candidates)} candidate pairs")

    if dense_scored:
        scored = dense_scored
        log.info(f"Using {len(scored)} pre-scored pairs from dense retrieval")
    elif not all_candidates:
        log.warning("No candidates found! Try lowering min_shared_keywords.")
        return
    else:
        with timed_section("Scoring candidates with embeddings"):
            embedder = Embedder(
                cache_dir=CACHE_DIR / "embeddings",
                model_name=config.embedding_model,
                default_batch_size=config.embedding_batch_size,
            )
            scored = score_candidates(all_candidates, embedder)
            log.info(f"Scored {len(scored)} pairs")

    with timed_section("Filtering and exporting"):
        filtered: list = []
        for p in scored:
            lang_a, lang_b = p.sentence_a.language, p.sentence_b.language
            is_cross = lang_a != lang_b

            if is_cross:
                threshold = config.cross_lingual.semantic_threshold
                min_kw = config.cross_lingual.min_shared_keywords
            elif lang_a == "en":
                threshold = config.english.semantic_threshold
                min_kw = config.english.min_shared_keywords
            else:
                threshold = config.german.semantic_threshold
                min_kw = config.german.min_shared_keywords

            if p.semantic_similarity >= threshold and len(p.shared_keywords) >= min_kw:
                filtered.append(p)

        log.info(f"Pairs passing thresholds: {len(filtered)}")
        top_pairs = rank_pairs(filtered, top_k=len(filtered))
        export_to_csv(top_pairs, RESULTS_DIR / "top_pairs.csv")
        export_to_json(top_pairs, RESULTS_DIR / "paraphrase_clusters.json")

    log.info("=" * 60)
    log.info("Results Summary")
    log.info("=" * 60)

    high_quality = [
        p
        for p in top_pairs
        if p.semantic_similarity >= 0.9 and p.jaccard_similarity < 0.5
    ]
    log.info(f"High quality (sem >= 0.9 AND jaccard < 0.5): {len(high_quality)}")

    en_en = [
        p
        for p in top_pairs
        if p.sentence_a.language == "en" and p.sentence_b.language == "en"
    ]
    de_de = [
        p
        for p in top_pairs
        if p.sentence_a.language == "de" and p.sentence_b.language == "de"
    ]
    cross = [p for p in top_pairs if p.sentence_a.language != p.sentence_b.language]

    log.info(f"EN-EN pairs: {len(en_en)}")
    log.info(f"DE-DE pairs: {len(de_de)}")
    log.info(f"Cross-lingual (EN-DE): {len(cross)}")

    log.info("--- Top 5 Pairs ---")
    for i, pair in enumerate(top_pairs[:5]):
        log.info(
            f"[{i + 1}] Quality: {pair.quality_score:.3f} | "
            f"Semantic: {pair.semantic_similarity:.3f} | "
            f"Jaccard: {pair.jaccard_similarity:.3f}"
        )
        log.info(f"    Lang: {pair.sentence_a.language} / {pair.sentence_b.language}")
        log.info(f"    A: {pair.sentence_a.text[:100]}...")
        log.info(f"    B: {pair.sentence_b.text[:100]}...")
        log.info(f"    Keywords: {', '.join(pair.shared_keywords[:5])}")


if __name__ == "__main__":
    run_pipeline(DEFAULT_CONFIG)
