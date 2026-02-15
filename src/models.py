from __future__ import annotations

from pydantic import BaseModel


class Sentence(BaseModel):
    id: int
    text: str
    language: str
    keywords: list[str] = []

    def __hash__(self) -> int:
        return hash(self.id)


class Candidate(BaseModel):
    sentence_a: Sentence
    sentence_b: Sentence
    shared_keywords: list[str]
    jaccard_similarity: float


class ScoredPair(BaseModel):
    sentence_a: Sentence
    sentence_b: Sentence
    semantic_similarity: float
    jaccard_similarity: float
    shared_keywords: list[str]

    @property
    def quality_score(self) -> float:
        return self.semantic_similarity * (1 - self.jaccard_similarity)
