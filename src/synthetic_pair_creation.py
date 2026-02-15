from __future__ import annotations

from pathlib import Path

import polars as pl
from pydantic import BaseModel
from tqdm import tqdm

from src.config import DEFAULT_CONFIG
from src.embedder import Embedder
from src.logging_utils import get_logger
from src.models import Candidate, Sentence, ScoredPair
from src.neural_translator import NeuralTranslator
from src.scorer import score_candidates


class SyntheticPair(BaseModel):
    sentence_a: Sentence
    sentence_b: Sentence
    synthetic_a: bool
    synthetic_b: bool
    semantic_similarity: float
    jaccard_similarity: float
    shared_keywords: list[str]

    @property
    def quality_score(self) -> float:
        return self.semantic_similarity * (1 - self.jaccard_similarity)


def load_existing_pairs(csv_path: Path) -> list[ScoredPair]:
    log = get_logger()
    log.info(f"Loading existing pairs from {csv_path}")

    df = pl.read_csv(csv_path)

    pairs: list[ScoredPair] = []
    for row in df.iter_rows(named=True):
        sent_a = Sentence(
            id=hash(row["sentence_a"]),
            text=row["sentence_a"],
            language=row["language_a"],
            keywords=[],
        )
        sent_b = Sentence(
            id=hash(row["sentence_b"]),
            text=row["sentence_b"],
            language=row["language_b"],
            keywords=[],
        )

        keywords = []
        if row["shared_keywords"] and isinstance(row["shared_keywords"], str):
            keywords = [k.strip() for k in row["shared_keywords"].split(",")]

        pairs.append(
            ScoredPair(
                sentence_a=sent_a,
                sentence_b=sent_b,
                semantic_similarity=row["semantic_similarity"],
                jaccard_similarity=row["jaccard_similarity"],
                shared_keywords=keywords,
            )
        )

    log.info(f"Loaded {len(pairs):,} pairs")
    return pairs


def extract_unique_sentences(
    pairs: list[ScoredPair],
) -> tuple[set[str], set[str]]:
    log = get_logger()

    en_sents: set[str] = set()
    de_sents: set[str] = set()

    for pair in pairs:
        if pair.sentence_a.language == "en" and pair.sentence_b.language == "en":
            en_sents.add(pair.sentence_a.text)
            en_sents.add(pair.sentence_b.text)
        elif pair.sentence_a.language == "de" and pair.sentence_b.language == "de":
            de_sents.add(pair.sentence_a.text)
            de_sents.add(pair.sentence_b.text)

    log.info(f"Extracted {len(en_sents):,} unique EN sentences")
    log.info(f"Extracted {len(de_sents):,} unique DE sentences")

    return en_sents, de_sents


def translate_all(
    en_sents: set[str],
    de_sents: set[str],
    translator: NeuralTranslator,
    batch_size: int = 1024,
) -> tuple[dict[str, str], dict[str, str]]:
    log = get_logger()

    en_texts = list(en_sents)
    de_texts = list(de_sents)

    log.info(f"Translating {len(en_texts):,} EN → DE...")
    en_translations = []
    for i in tqdm(range(0, len(en_texts), batch_size), desc="EN→DE", unit="batch"):
        batch = en_texts[i : i + batch_size]
        batch_trans = translator.translate_batch(batch, "en", "de", batch_size)
        en_translations.extend(batch_trans)
    en_to_de = dict(zip(en_texts, en_translations))

    log.info(f"Translating {len(de_texts):,} DE → EN...")
    de_translations = []
    for i in tqdm(range(0, len(de_texts), batch_size), desc="DE→EN", unit="batch"):
        batch = de_texts[i : i + batch_size]
        batch_trans = translator.translate_batch(batch, "de", "en", batch_size)
        de_translations.extend(batch_trans)
    de_to_en = dict(zip(de_texts, de_translations))

    return en_to_de, de_to_en


def generate_synthetic_pairs(
    original_pairs: list[ScoredPair],
    en_to_de: dict[str, str],
    de_to_en: dict[str, str],
) -> list[tuple[Candidate, bool, bool]]:
    """Generate ALL synthetic pair variations from monolingual pairs.

    For each monolingual pair (A, B), creates 12 variations:
    1. (A, B) - original
    2. (B, A) - reversed original
    3. (translate(A), B) - cross-lingual
    4. (translate(B), A) - cross-lingual reversed
    5. (A, translate(B)) - cross-lingual
    6. (B, translate(A)) - cross-lingual reversed
    7. (translate(A), translate(B)) - synthetic monolingual
    8. (translate(B), translate(A)) - synthetic monolingual reversed
    9. (A, translate(A)) - self-translation
    10. (translate(A), A) - self-translation reversed
    11. (B, translate(B)) - self-translation
    12. (translate(B), B) - self-translation reversed

    Returns:
        List of (Candidate, synthetic_a, synthetic_b) tuples
    """
    log = get_logger()

    synthetic_candidates: list[tuple[Candidate, bool, bool]] = []

    for pair in tqdm(original_pairs, desc="Generating synthetic pairs", unit="pair"):
        if pair.sentence_a.language == "en" and pair.sentence_b.language == "en":
            a_text = pair.sentence_a.text
            b_text = pair.sentence_b.text
            a_trans = en_to_de[a_text]
            b_trans = en_to_de[b_text]

            a_en = pair.sentence_a
            b_en = pair.sentence_b
            a_de = Sentence(id=hash(a_trans), text=a_trans, language="de", keywords=[])
            b_de = Sentence(id=hash(b_trans), text=b_trans, language="de", keywords=[])

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=a_en,
                        sentence_b=b_en,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    False,
                    False,
                )
            )

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=b_en,
                        sentence_b=a_en,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    False,
                    False,
                )
            )

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=a_de,
                        sentence_b=b_en,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    True,
                    False,
                )
            )

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=b_de,
                        sentence_b=a_en,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    True,
                    False,
                )
            )

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=a_en,
                        sentence_b=b_de,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    False,
                    True,
                )
            )

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=b_en,
                        sentence_b=a_de,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    False,
                    True,
                )
            )

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=a_de,
                        sentence_b=b_de,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    True,
                    True,
                )
            )

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=b_de,
                        sentence_b=a_de,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    True,
                    True,
                )
            )

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=a_en,
                        sentence_b=a_de,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    False,
                    True,
                )
            )

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=a_de,
                        sentence_b=a_en,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    True,
                    False,
                )
            )

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=b_en,
                        sentence_b=b_de,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    False,
                    True,
                )
            )

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=b_de,
                        sentence_b=b_en,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    True,
                    False,
                )
            )

        elif pair.sentence_a.language == "de" and pair.sentence_b.language == "de":
            a_text = pair.sentence_a.text
            b_text = pair.sentence_b.text
            a_trans = de_to_en[a_text]
            b_trans = de_to_en[b_text]

            a_de = pair.sentence_a
            b_de = pair.sentence_b
            a_en = Sentence(id=hash(a_trans), text=a_trans, language="en", keywords=[])
            b_en = Sentence(id=hash(b_trans), text=b_trans, language="en", keywords=[])

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=a_de,
                        sentence_b=b_de,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    False,
                    False,
                )
            )

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=b_de,
                        sentence_b=a_de,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    False,
                    False,
                )
            )

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=a_en,
                        sentence_b=b_de,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    True,
                    False,
                )
            )

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=b_en,
                        sentence_b=a_de,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    True,
                    False,
                )
            )

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=a_de,
                        sentence_b=b_en,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    False,
                    True,
                )
            )

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=b_de,
                        sentence_b=a_en,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    False,
                    True,
                )
            )

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=a_en,
                        sentence_b=b_en,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    True,
                    True,
                )
            )

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=b_en,
                        sentence_b=a_en,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    True,
                    True,
                )
            )

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=a_de,
                        sentence_b=a_en,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    False,
                    True,
                )
            )

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=a_en,
                        sentence_b=a_de,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    True,
                    False,
                )
            )

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=b_de,
                        sentence_b=b_en,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    False,
                    True,
                )
            )

            synthetic_candidates.append(
                (
                    Candidate(
                        sentence_a=b_en,
                        sentence_b=b_de,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    ),
                    True,
                    False,
                )
            )

    log.info(f"Generated {len(synthetic_candidates):,} synthetic candidate pairs")
    log.info(f"  From {len(original_pairs):,} monolingual pairs (12 variations each)")
    return synthetic_candidates


def filter_and_score(
    candidates: list[tuple[Candidate, bool, bool]],
    embedder: Embedder,
    threshold: float,
) -> list[SyntheticPair]:
    log = get_logger()

    candidate_list = [c[0] for c in candidates]
    synthetic_flags = [(c[1], c[2]) for c in candidates]

    log.info(f"Scoring {len(candidate_list):,} candidates...")
    scored_pairs = score_candidates(candidate_list, embedder)

    log.info(f"Filtering by semantic threshold ≥ {threshold}")
    filtered: list[SyntheticPair] = []

    for scored, (synth_a, synth_b) in zip(scored_pairs, synthetic_flags):
        if scored.semantic_similarity >= threshold:
            filtered.append(
                SyntheticPair(
                    sentence_a=scored.sentence_a,
                    sentence_b=scored.sentence_b,
                    synthetic_a=synth_a,
                    synthetic_b=synth_b,
                    semantic_similarity=scored.semantic_similarity,
                    jaccard_similarity=scored.jaccard_similarity,
                    shared_keywords=scored.shared_keywords,
                )
            )

    log.info(
        f"Filtered to {len(filtered):,} pairs ({len(filtered) / len(scored_pairs) * 100:.1f}%)"
    )
    return filtered


def export_synthetic_pairs_csv(pairs: list[SyntheticPair], output_path: Path) -> None:
    log = get_logger()

    sorted_pairs = sorted(pairs, key=lambda p: p.quality_score, reverse=True)

    rows = []
    for i, pair in enumerate(sorted_pairs):
        rows.append(
            {
                "rank": i + 1,
                "sentence_a": pair.sentence_a.text,
                "language_a": pair.sentence_a.language,
                "synthetic_a": pair.synthetic_a,
                "sentence_b": pair.sentence_b.text,
                "language_b": pair.sentence_b.language,
                "synthetic_b": pair.synthetic_b,
                "semantic_similarity": round(pair.semantic_similarity, 4),
                "jaccard_similarity": round(pair.jaccard_similarity, 4),
                "quality_score": round(pair.quality_score, 4),
                "shared_keywords": ", ".join(pair.shared_keywords),
            }
        )

    df = pl.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)
    log.info(f"Exported {len(pairs):,} pairs to {output_path}")


def main() -> None:
    log = get_logger()

    PROJECT_ROOT = Path(__file__).parent.parent
    RESULTS_DIR = PROJECT_ROOT / "results"
    CACHE_DIR = PROJECT_ROOT / ".cache"

    input_path = RESULTS_DIR / "top_pairs.csv"
    output_path = RESULTS_DIR / "synthetic_cross_lingual_pairs.csv"

    log.info("=== Synthetic Cross-Lingual Pair Generation ===")

    pairs = load_existing_pairs(input_path)

    en_sents, de_sents = extract_unique_sentences(pairs)

    log.info("Initializing NeuralTranslator...")
    translator = NeuralTranslator(cache_dir=CACHE_DIR / "ct2_models")

    en_to_de, de_to_en = translate_all(en_sents, de_sents, translator)

    synthetic_candidates = generate_synthetic_pairs(pairs, en_to_de, de_to_en)

    log.info("Initializing Embedder for scoring...")
    embedder = Embedder(
        cache_dir=CACHE_DIR / "embeddings",
        model_name=DEFAULT_CONFIG.embedding_model,
        default_batch_size=DEFAULT_CONFIG.embedding_batch_size,
    )

    threshold = DEFAULT_CONFIG.cross_lingual.semantic_threshold
    filtered_pairs = filter_and_score(synthetic_candidates, embedder, threshold)

    export_synthetic_pairs_csv(filtered_pairs, output_path)

    log.info("=== Synthetic Pair Generation Complete ===")
    log.info(f"Input: {len(pairs):,} monolingual pairs")
    log.info(
        f"Generated: {len(synthetic_candidates):,} candidate pairs (12 variations each)"
    )
    log.info(
        f"Filtered output: {len(filtered_pairs):,} pairs (semantic threshold ≥ {threshold})"
    )

    original = sum(1 for p in filtered_pairs if not p.synthetic_a and not p.synthetic_b)
    cross_lingual = sum(
        1 for p in filtered_pairs if p.sentence_a.language != p.sentence_b.language
    )
    synthetic_mono = sum(
        1
        for p in filtered_pairs
        if p.sentence_a.language == p.sentence_b.language
        and p.synthetic_a
        and p.synthetic_b
    )
    self_translation = sum(
        1
        for p in filtered_pairs
        if p.sentence_a.text == p.sentence_b.text
        or (
            p.synthetic_a != p.synthetic_b
            and p.sentence_a.language != p.sentence_b.language
        )
    )

    log.info(f"  Original pairs: {original:,}")
    log.info(f"  Cross-lingual: {cross_lingual:,}")
    log.info(f"  Synthetic monolingual: {synthetic_mono:,}")
    log.info(f"  Self-translations: {self_translation:,}")


if __name__ == "__main__":
    main()
