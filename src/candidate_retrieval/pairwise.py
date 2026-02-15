from __future__ import annotations

import time

from src.inverted_index import InvertedIndex
from src.legal_entities import LegalEntities, extract_legal_entities_batch
from src.logging_utils import get_logger
from src.models import Candidate, Sentence


def compute_jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def tokenize_for_jaccard(text: str) -> set[str]:
    import re

    tokens = re.findall(r"\b\w+\b", text.lower())
    return set(tokens)


def retrieve_candidates(
    sentences: list[Sentence],
    index: InvertedIndex,
    min_shared_keywords: int = 2,
    filter_conflicting_entities: bool = True,
) -> list[Candidate]:
    sentence_map = {s.id: s for s in sentences}
    seen_pairs: set[tuple[int, int]] = set()
    candidates: list[Candidate] = []

    log = get_logger()
    entities_map: dict[int, LegalEntities] = {}
    if filter_conflicting_entities:
        t0 = time.perf_counter()
        texts = [s.text for s in sentences]
        entities_list = extract_legal_entities_batch(texts)
        for s, entities in zip(sentences, entities_list):
            entities_map[s.id] = entities
        log.info(f"Entity extraction: {time.perf_counter() - t0:.2f}s")

    entity_conflicts_filtered = 0

    t0 = time.perf_counter()
    for sentence in sentences:
        candidate_ids = index.find_candidates(sentence, min_shared_keywords)

        tokens_a = tokenize_for_jaccard(sentence.text)
        entities_a = entities_map.get(sentence.id)

        for other_id, shared_kws in candidate_ids.items():
            pair_key = (min(sentence.id, other_id), max(sentence.id, other_id))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            other = sentence_map.get(other_id)
            if other is None:
                continue

            if filter_conflicting_entities and entities_a is not None:
                entities_b = entities_map.get(other_id)
                if entities_b is not None and entities_a.conflicts_with(entities_b):
                    entity_conflicts_filtered += 1
                    continue

            tokens_b = tokenize_for_jaccard(other.text)
            jaccard = compute_jaccard_similarity(tokens_a, tokens_b)

            candidates.append(
                Candidate(
                    sentence_a=sentence,
                    sentence_b=other,
                    shared_keywords=shared_kws,
                    jaccard_similarity=jaccard,
                )
            )

    log.info(f"Candidate pair iteration: {time.perf_counter() - t0:.2f}s")

    if filter_conflicting_entities and entity_conflicts_filtered > 0:
        log.info(
            f"Filtered {entity_conflicts_filtered} pairs with conflicting entities"
        )

    return candidates
