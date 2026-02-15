from __future__ import annotations

import time

from datasketch import MinHash, MinHashLSH  # type: ignore[import-untyped]

from src.legal_entities import LegalEntities, extract_legal_entities_batch
from src.logging_utils import get_logger
from src.models import Candidate, Sentence

from .pairwise import compute_jaccard_similarity, tokenize_for_jaccard


def create_minhash(keywords: list[str], num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for kw in keywords:
        m.update(kw.encode("utf8"))
    return m


def cluster_by_keywords(
    sentences: list[Sentence],
    threshold: float = 0.3,
    num_perm: int = 128,
) -> list[list[Sentence]]:
    log = get_logger()

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    minhashes: dict[int, MinHash] = {}
    for sentence in sentences:
        if not sentence.keywords:
            continue
        mh = create_minhash(sentence.keywords, num_perm)
        minhashes[sentence.id] = mh
        lsh.insert(str(sentence.id), mh)

    sentence_map = {s.id: s for s in sentences}
    visited: set[int] = set()
    clusters: list[list[Sentence]] = []

    for sentence in sentences:
        if sentence.id in visited or sentence.id not in minhashes:
            continue

        mh = minhashes[sentence.id]
        similar_ids = lsh.query(mh)

        cluster: list[Sentence] = []
        for sid_str in similar_ids:
            sid = int(sid_str)
            if sid not in visited:
                visited.add(sid)
                cluster.append(sentence_map[sid])

        if len(cluster) > 1:
            clusters.append(cluster)

    log.info(f"Created {len(clusters)} clusters from {len(sentences)} sentences")
    cluster_sizes = [len(c) for c in clusters]
    if cluster_sizes:
        log.info(
            f"Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
            f"avg={sum(cluster_sizes) / len(cluster_sizes):.1f}"
        )

    return clusters


def retrieve_candidates_clustered(
    sentences: list[Sentence],
    threshold: float = 0.3,
    num_perm: int = 128,
    filter_conflicting_entities: bool = True,
) -> list[Candidate]:
    log = get_logger()

    t0 = time.perf_counter()
    clusters = cluster_by_keywords(sentences, threshold, num_perm)
    log.info(f"Clustering: {time.perf_counter() - t0:.2f}s")

    entities_map: dict[int, LegalEntities] = {}
    if filter_conflicting_entities:
        t0 = time.perf_counter()
        texts = [s.text for s in sentences]
        entities_list = extract_legal_entities_batch(texts)
        for s, entities in zip(sentences, entities_list):
            entities_map[s.id] = entities
        log.info(f"Entity extraction: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    candidates: list[Candidate] = []
    entity_conflicts_filtered = 0
    seen_pairs: set[tuple[int, int]] = set()

    for cluster in clusters:
        for i, sent_a in enumerate(cluster):
            tokens_a = tokenize_for_jaccard(sent_a.text)
            entities_a = entities_map.get(sent_a.id)

            for sent_b in cluster[i + 1 :]:
                pair_key = (min(sent_a.id, sent_b.id), max(sent_a.id, sent_b.id))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                if filter_conflicting_entities and entities_a is not None:
                    entities_b = entities_map.get(sent_b.id)
                    if entities_b is not None and entities_a.conflicts_with(entities_b):
                        entity_conflicts_filtered += 1
                        continue

                tokens_b = tokenize_for_jaccard(sent_b.text)
                jaccard = compute_jaccard_similarity(tokens_a, tokens_b)

                shared = set(sent_a.keywords) & set(sent_b.keywords)

                candidates.append(
                    Candidate(
                        sentence_a=sent_a,
                        sentence_b=sent_b,
                        shared_keywords=list(shared),
                        jaccard_similarity=jaccard,
                    )
                )

    log.info(f"Pairwise within clusters: {time.perf_counter() - t0:.2f}s")

    if filter_conflicting_entities and entity_conflicts_filtered > 0:
        log.info(
            f"Filtered {entity_conflicts_filtered} pairs with conflicting entities"
        )

    return candidates
