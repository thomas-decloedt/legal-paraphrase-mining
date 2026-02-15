"""Candidate retrieval approaches for paraphrase mining."""

from .clustering import retrieve_candidates_clustered
from .cross_lingual import CrossLingualRetriever
from .dense_ann import DenseANNRetriever
from .pairwise import retrieve_candidates
from .sparse_ann import SparseANNRetriever

__all__ = [
    "retrieve_candidates",
    "retrieve_candidates_clustered",
    "SparseANNRetriever",
    "CrossLingualRetriever",
    "DenseANNRetriever",
]
