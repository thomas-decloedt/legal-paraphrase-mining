from __future__ import annotations

from src.candidate_retrieval.sparse_ann import SparseANNRetriever
from src.logging_utils import get_logger
from src.models import Candidate, Sentence
from src.neural_translator import NeuralTranslator


class TwoStageRetriever:
    def __init__(
        self, translator: NeuralTranslator, sparse_retriever: SparseANNRetriever
    ):
        self.translator = translator
        self.sparse = sparse_retriever

    def retrieve_cross_lingual(
        self, sentences: list[Sentence], top_k: int = 100
    ) -> list[Candidate]:
        log = get_logger()

        en_sents = [s for s in sentences if s.language == "en"]
        de_sents = [s for s in sentences if s.language == "de"]

        log.info(
            f"Stage 1: Finding monolingual paraphrases ({len(en_sents)} EN, {len(de_sents)} DE)"
        )

        self.sparse.index_sentences(en_sents)
        en_pairs = self.sparse.retrieve_candidates(en_sents, top_k=top_k, fast=True)
        log.info(f"Found {len(en_pairs)} EN-EN paraphrase pairs")

        self.sparse.index_sentences(de_sents)
        de_pairs = self.sparse.retrieve_candidates(de_sents, top_k=top_k, fast=True)
        log.info(f"Found {len(de_pairs)} DE-DE paraphrase pairs")

        log.info("Stage 2: Translating paraphrase candidates...")
        cross_lingual = []

        if en_pairs:
            text_to_indices: dict[str, list[int]] = {}
            unique_texts: list[str] = []
            for idx, pair in enumerate(en_pairs):
                text = pair.sentence_a.text
                if text not in text_to_indices:
                    text_to_indices[text] = []
                    unique_texts.append(text)
                text_to_indices[text].append(idx)

            log.info(
                f"Deduped {len(en_pairs)} EN pairs → {len(unique_texts)} unique texts"
            )
            log.info(f"Translating {len(unique_texts)} EN → DE...")

            translations = self.translator.translate_batch(unique_texts, "en", "de")
            translation_map: dict[str, str] = dict(zip(unique_texts, translations))

            for pair in en_pairs:
                translation = translation_map[pair.sentence_a.text]
                translated_sent = Sentence(
                    id=pair.sentence_a.id,
                    text=translation,
                    language="de",
                    keywords=[],
                )
                cross_lingual.append(
                    Candidate(
                        sentence_a=pair.sentence_b,
                        sentence_b=translated_sent,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    )
                )

        if de_pairs:
            text_to_indices = {}
            unique_texts = []
            for idx, pair in enumerate(de_pairs):
                text = pair.sentence_a.text
                if text not in text_to_indices:
                    text_to_indices[text] = []
                    unique_texts.append(text)
                text_to_indices[text].append(idx)

            log.info(
                f"Deduped {len(de_pairs)} DE pairs → {len(unique_texts)} unique texts"
            )
            log.info(f"Translating {len(unique_texts)} DE → EN...")

            translations = self.translator.translate_batch(unique_texts, "de", "en")
            translation_map = dict(zip(unique_texts, translations))

            for pair in de_pairs:
                translation = translation_map[pair.sentence_a.text]
                translated_sent = Sentence(
                    id=pair.sentence_a.id,
                    text=translation,
                    language="en",
                    keywords=[],
                )
                cross_lingual.append(
                    Candidate(
                        sentence_a=pair.sentence_b,
                        sentence_b=translated_sent,
                        shared_keywords=[],
                        jaccard_similarity=0.0,
                    )
                )

        log.info(f"Created {len(cross_lingual)} cross-lingual candidate pairs")
        return cross_lingual
