from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LanguageConfig(BaseModel):
    num_sentences: int = Field(default=10000, ge=1)
    min_shared_keywords: int = Field(default=4, ge=1)
    max_jaccard_similarity: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Maximum Jaccard similarity for lexical diversity filtering (lower = more diverse)",
    )
    semantic_threshold: float = Field(default=0.95, ge=0.0, le=1.0)


class CrossLingualConfig(BaseModel):
    enabled: bool = True
    min_shared_keywords: int = Field(default=3, ge=0)
    semantic_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    enable_bigram_permutations: bool = Field(
        default=False,
        description="Enable all permutations for bigram translations (expensive, causes 84s slowdown)",
    )


class SynonymConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description="Enable synonym expansion for same-language matching",
    )
    max_synonyms_per_word: int = Field(
        default=5,
        ge=1,
        description="Maximum synonyms per keyword (WordNet limit)",
    )


class SparseRetrievalConfig(BaseModel):
    use_bm25: bool = Field(
        default=True,
        description="Use Qdrant's BM25 text index (requires fastembed) instead of binary sparse vectors",
    )
    bm25_k1: float | None = Field(
        default=None,
        description="BM25 k1 parameter (if supported by Qdrant API)",
    )
    bm25_b: float | None = Field(
        default=None,
        description="BM25 b parameter (if supported by Qdrant API)",
    )
    min_similarity_score: float = Field(
        default=24.0,
        ge=0.0,
        description="Minimum BM25 score threshold (0.0 = no filter, 1.0 = light, 2.0 = moderate, 3.0+ = aggressive)",
    )


class DenseEmbeddingConfig(BaseModel):
    model_name: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        description="HuggingFace model name for multilingual embeddings",
    )
    embedding_backend: Literal["sentence-transformer", "model2vec"] = Field(
        default="sentence-transformer",
        description="Backend: sentence-transformer (Qwen3) or model2vec (static, 500x faster)",
    )
    dimension: int = Field(
        default=256,
        ge=32,
        le=1024,
        description="Embedding dimension (Matryoshka: 32-1024, lower = faster)",
    )
    instruction: str | None = Field(
        default="Find semantically similar legal sentences across languages",
        description="Instruction prefix for instruction-tuned models",
    )
    batch_size: int = Field(default=64, ge=1)
    normalize: bool = Field(default=True, description="L2 normalize embeddings")


class DenseRetrievalConfig(BaseModel):
    enabled: bool = Field(default=False, description="Enable dense retrieval")
    embedding: DenseEmbeddingConfig = Field(default_factory=DenseEmbeddingConfig)
    collection_name: str = Field(default="sentences_dense")
    top_k: int = Field(default=100, ge=1)
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for candidates",
    )
    cross_lingual_only: bool = Field(
        default=False,
        description="If True, only return EN↔DE pairs",
    )
    use_quantization: bool = Field(
        default=False,
        description="Enable INT8 scalar quantization for memory savings",
    )
    on_disk: bool = Field(
        default=False,
        description="Store vectors on disk for large datasets",
    )


class PipelineConfig(BaseModel):
    min_sentence_length: int = Field(
        default=50,
        ge=1,
        description="Minimum sentence length in characters",
    )

    fast: bool = Field(
        default=True, description="Use fast regex extractors instead of spaCy"
    )

    retrieval_method: Literal[
        "pairwise", "clustering", "sparse-ann", "dense-ann", "two-stage"
    ] = Field(
        default="sparse-ann",
        description="Candidate retrieval method",
    )
    top_k: int = Field(default=100, ge=1, description="Number of neighbors for ANN")

    embedding_model: str = Field(
        default="jinaai/jina-embeddings-v3",
        description="HuggingFace model for semantic similarity scoring",
    )
    embedding_batch_size: int = Field(
        default=128,
        ge=1,
        description="Batch size for embedding computation",
    )

    sparse_retrieval: SparseRetrievalConfig = Field(
        default_factory=SparseRetrievalConfig
    )

    dense: DenseRetrievalConfig = Field(default_factory=DenseRetrievalConfig)

    english: LanguageConfig = Field(default_factory=LanguageConfig)
    german: LanguageConfig = Field(default_factory=LanguageConfig)

    cross_lingual: CrossLingualConfig = Field(default_factory=CrossLingualConfig)

    synonyms: SynonymConfig = Field(default_factory=SynonymConfig)


DEFAULT_CONFIG = PipelineConfig(
    fast=True,
    retrieval_method="sparse-ann",
    top_k=1,
    english=LanguageConfig(
        num_sentences=10_000_000,
        min_shared_keywords=8,
        max_jaccard_similarity=1,
        semantic_threshold=0.95,
    ),
    german=LanguageConfig(
        num_sentences=10_000_000,
        min_shared_keywords=8,
        max_jaccard_similarity=1,
        semantic_threshold=0.95,
    ),
    cross_lingual=CrossLingualConfig(
        enabled=False,
        min_shared_keywords=0,
        semantic_threshold=0.85,
    ),
    dense=DenseRetrievalConfig(
        enabled=False,
        embedding=DenseEmbeddingConfig(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            embedding_backend="sentence-transformer",
            dimension=256,
            instruction="Find semantically similar legal sentences across languages",
        ),
        similarity_threshold=0.85,
        cross_lingual_only=True,
    ),
)
