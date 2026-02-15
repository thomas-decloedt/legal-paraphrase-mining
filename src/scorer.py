from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import polars as pl
from pydantic import BaseModel

from src.embedder import Embedder
from src.logging_utils import get_logger
from src.models import Candidate, ScoredPair


class ParaphraseCluster(BaseModel):
    seed: str
    paraphrases: list[str]
    avg_semantic_similarity: float
    avg_jaccard_similarity: float


def _get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def score_candidates(
    candidates: list[Candidate],
    embedder: Embedder,
) -> list[ScoredPair]:
    if not candidates:
        return []

    import torch

    texts_to_embed: list[str] = []
    text_to_idx: dict[str, int] = {}

    for candidate in candidates:
        for text in [candidate.sentence_a.text, candidate.sentence_b.text]:
            if text not in text_to_idx:
                text_to_idx[text] = len(texts_to_embed)
                texts_to_embed.append(text)

    device = _get_device()
    log = get_logger()
    log.info(
        f"Embedding {len(texts_to_embed):,} unique sentences (model: {embedder.model_name})..."
    )

    embeddings = embedder.embed_batch(texts_to_embed)
    log.info(
        f"Completed embedding. Now scoring {len(candidates):,} pairs on device: {device}"
    )

    idx_a_list = [text_to_idx[c.sentence_a.text] for c in candidates]
    idx_b_list = [text_to_idx[c.sentence_b.text] for c in candidates]

    import numpy as np
    from sklearn.preprocessing import normalize  # type: ignore[import-untyped]

    emb_matrix_cpu = np.vstack(embeddings)
    emb_normalized_cpu = normalize(emb_matrix_cpu, norm="l2", axis=1).astype(np.float32)

    gpu_batch_size = 50_000
    n_pairs = len(candidates)
    similarities = np.empty(n_pairs, dtype=np.float32)

    for batch_start in range(0, n_pairs, gpu_batch_size):
        batch_end = min(batch_start + gpu_batch_size, n_pairs)

        batch_idx_a = idx_a_list[batch_start:batch_end]
        batch_idx_b = idx_b_list[batch_start:batch_end]
        unique_indices = sorted(set(batch_idx_a) | set(batch_idx_b))

        emb_subset = torch.tensor(
            emb_normalized_cpu[unique_indices], device=device, dtype=torch.float32
        )

        global_to_local = {
            global_idx: local_idx for local_idx, global_idx in enumerate(unique_indices)
        }

        local_idx_a = torch.tensor(
            [global_to_local[idx] for idx in batch_idx_a], device=device
        )
        local_idx_b = torch.tensor(
            [global_to_local[idx] for idx in batch_idx_b], device=device
        )

        emb_a = emb_subset[local_idx_a]
        emb_b = emb_subset[local_idx_b]

        batch_sims = torch.sum(emb_a * emb_b, dim=1).cpu().numpy()
        similarities[batch_start:batch_end] = batch_sims

        if batch_start > 0 and batch_start % 100_000 == 0:
            log.info(f"  Scored {batch_start:,} / {n_pairs:,} pairs")

    scored_pairs: list[ScoredPair] = []
    for i, candidate in enumerate(candidates):
        scored_pairs.append(
            ScoredPair(
                sentence_a=candidate.sentence_a,
                sentence_b=candidate.sentence_b,
                semantic_similarity=float(similarities[i]),
                jaccard_similarity=candidate.jaccard_similarity,
                shared_keywords=candidate.shared_keywords,
            )
        )

    return scored_pairs


def rank_pairs(pairs: list[ScoredPair], top_k: int = 50) -> list[ScoredPair]:
    sorted_pairs = sorted(pairs, key=lambda p: p.quality_score, reverse=True)
    return sorted_pairs[:top_k]


def export_to_csv(pairs: list[ScoredPair], output_path: Path) -> None:
    rows = []
    for i, pair in enumerate(pairs):
        rows.append(
            {
                "rank": i + 1,
                "sentence_a": pair.sentence_a.text,
                "language_a": pair.sentence_a.language,
                "sentence_b": pair.sentence_b.text,
                "language_b": pair.sentence_b.language,
                "semantic_similarity": round(pair.semantic_similarity, 4),
                "jaccard_similarity": round(pair.jaccard_similarity, 4),
                "quality_score": round(pair.quality_score, 4),
                "shared_keywords": ", ".join(pair.shared_keywords),
            }
        )

    df = pl.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)
    print(f"Exported {len(pairs)} pairs to {output_path}")


def build_clusters(pairs: list[ScoredPair]) -> list[ParaphraseCluster]:
    sentence_counts: dict[str, int] = defaultdict(int)
    sentence_pairs: dict[str, list[ScoredPair]] = defaultdict(list)

    for pair in pairs:
        sentence_counts[pair.sentence_a.text] += 1
        sentence_counts[pair.sentence_b.text] += 1
        sentence_pairs[pair.sentence_a.text].append(pair)
        sentence_pairs[pair.sentence_b.text].append(pair)

    used_sentences: set[str] = set()
    clusters: list[ParaphraseCluster] = []

    for sentence in sorted(sentence_counts.keys(), key=lambda s: -sentence_counts[s]):
        if sentence in used_sentences:
            continue

        paraphrases: list[str] = []
        semantic_sims: list[float] = []
        jaccard_sims: list[float] = []

        for pair in sentence_pairs[sentence]:
            other = (
                pair.sentence_b.text
                if pair.sentence_a.text == sentence
                else pair.sentence_a.text
            )
            if other not in used_sentences:
                paraphrases.append(other)
                semantic_sims.append(pair.semantic_similarity)
                jaccard_sims.append(pair.jaccard_similarity)

        if paraphrases:
            used_sentences.add(sentence)
            for p in paraphrases:
                used_sentences.add(p)

            clusters.append(
                ParaphraseCluster(
                    seed=sentence,
                    paraphrases=paraphrases,
                    avg_semantic_similarity=sum(semantic_sims) / len(semantic_sims),
                    avg_jaccard_similarity=sum(jaccard_sims) / len(jaccard_sims),
                )
            )

    return clusters


def export_to_json(pairs: list[ScoredPair], output_path: Path) -> None:
    clusters = build_clusters(pairs)

    output = [{"seed": c.seed, "paraphrases": c.paraphrases} for c in clusters]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Exported {len(clusters)} clusters to {output_path}")
