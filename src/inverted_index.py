from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel

from src.models import Sentence


class InvertedIndex(BaseModel):
    index: dict[str, list[int]] = {}

    @classmethod
    def build(cls, sentences: list[Sentence]) -> InvertedIndex:
        index: dict[str, list[int]] = defaultdict(list)

        for sentence in sentences:
            for keyword in sentence.keywords:
                index[keyword].append(sentence.id)

        return cls(index=dict(index))

    def get_sentence_ids(self, keyword: str) -> list[int]:
        return self.index.get(keyword, [])

    def find_candidates(
        self, sentence: Sentence, min_shared_keywords: int = 2
    ) -> dict[int, list[str]]:
        candidate_keywords: dict[int, list[str]] = defaultdict(list)

        for keyword in sentence.keywords:
            for other_id in self.get_sentence_ids(keyword):
                if other_id != sentence.id:
                    candidate_keywords[other_id].append(keyword)

        return {
            sid: kws
            for sid, kws in candidate_keywords.items()
            if len(kws) >= min_shared_keywords
        }
