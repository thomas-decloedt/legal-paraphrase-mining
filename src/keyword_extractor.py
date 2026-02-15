from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import spacy
from spacy.lang.de.stop_words import STOP_WORDS as SPACY_STOPWORDS_DE
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOPWORDS_EN

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc

_models: dict[str, Language] = {}

_STOPWORDS_DIR = Path(__file__).parent / "stopwords"


@lru_cache(maxsize=1)
def _load_legal_stopwords_en() -> frozenset[str]:
    csv_path = _STOPWORDS_DIR / "EU_legal_EN.csv"
    if not csv_path.exists():
        return frozenset()
    words: set[str] = set()
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(
            (row for row in f if not row.startswith("#")),
        )
        for row in reader:
            if row.get("word"):
                words.add(row["word"].lower().strip())
    return frozenset(words)


@lru_cache(maxsize=1)
def _load_legal_stopwords_de() -> frozenset[str]:
    """Load German legal stopwords from SW-DE-RS CSV file.

    Source: https://zenodo.org/records/3995593 (CC0 license)
    Based on high-frequency words from German Federal Courts (1998-2020)
    """
    csv_path = _STOPWORDS_DIR / "SW-DE-RS_v1-0-0.csv"
    if not csv_path.exists():
        return frozenset()
    words: set[str] = set()
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Allgemein"):
                words.add(row["Allgemein"].lower().strip())
    return frozenset(words)


STOPWORDS_EN = frozenset(SPACY_STOPWORDS_EN) | _load_legal_stopwords_en()
STOPWORDS_DE = frozenset(SPACY_STOPWORDS_DE) | _load_legal_stopwords_de()


def get_model(language: str) -> Language:
    if language not in _models:
        model_name = "en_core_web_sm" if language == "en" else "de_core_news_sm"
        _models[language] = spacy.load(model_name)
    return _models[language]


def extract_keywords(text: str, language: str) -> list[str]:
    nlp = get_model(language)
    doc: Doc = nlp(text)

    keywords: set[str] = set()

    for token in doc:
        if token.pos_ in ("NOUN", "PROPN", "VERB") and not token.is_stop:
            lemma = token.lemma_.lower()
            if len(lemma) > 2:
                keywords.add(lemma)

    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower().strip()
        if len(chunk_text) > 3 and not all(t.is_stop for t in chunk):
            keywords.add(chunk_text)

    return sorted(keywords)


def extract_keywords_batch(texts: list[str], languages: list[str]) -> list[list[str]]:
    if not texts:
        return []

    en_indices: list[int] = []
    de_indices: list[int] = []
    en_texts: list[str] = []
    de_texts: list[str] = []

    for i, (text, lang) in enumerate(zip(texts, languages, strict=True)):
        if lang == "en":
            en_indices.append(i)
            en_texts.append(text)
        else:
            de_indices.append(i)
            de_texts.append(text)

    results: list[list[str]] = [[] for _ in texts]

    if en_texts:
        nlp_en = get_model("en")
        for idx, doc in zip(
            en_indices, nlp_en.pipe(en_texts, batch_size=50), strict=True
        ):
            results[idx] = _extract_from_doc(doc)

    if de_texts:
        nlp_de = get_model("de")
        for idx, doc in zip(
            de_indices, nlp_de.pipe(de_texts, batch_size=50), strict=True
        ):
            results[idx] = _extract_from_doc(doc)

    return results


def _extract_from_doc(doc: Doc) -> list[str]:
    keywords: set[str] = set()

    for token in doc:
        if token.pos_ in ("NOUN", "PROPN", "VERB") and not token.is_stop:
            lemma = token.lemma_.lower()
            if len(lemma) > 2:
                keywords.add(lemma)

    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower().strip()
        if len(chunk_text) > 3 and not all(t.is_stop for t in chunk):
            keywords.add(chunk_text)

    return sorted(keywords)


_WORD_PATTERN = re.compile(r"\b[a-zA-Z]{3,}\b")

_STEMMERS: dict = {}


def _get_stemmer(language: str):
    if language not in _STEMMERS:
        from nltk.stem import SnowballStemmer  # type: ignore[import-untyped]

        lang_name = "english" if language == "en" else "german"
        _STEMMERS[language] = SnowballStemmer(lang_name)
    return _STEMMERS[language]


def stem_word(word: str, language: str) -> str:
    return _get_stemmer(language).stem(word)


def extract_keywords_fast(
    text: str, language: str = "en", expand_synonyms: bool = False
) -> list[str]:
    """Fast keyword extraction using regex + stopword filtering + stemming.

    ~100x faster than spaCy-based extraction. Includes BOTH original words
    and stemmed versions for better recall in cross-lingual matching.

    Args:
        text: Input text
        language: Language code ("en" or "de")
        expand_synonyms: Whether to expand keywords with synonyms (default False).
            For scalable cross-lingual retrieval, leave False and let the
            retriever expand synonyms at query time after translation.

    Returns:
        Sorted list of keywords (original + stemmed, optionally + synonyms)
    """
    stopwords = STOPWORDS_EN if language == "en" else STOPWORDS_DE
    stemmer = _get_stemmer(language)

    tokens = _WORD_PATTERN.findall(text.lower())

    keywords: set[str] = set()
    content_tokens = []
    stemmed_tokens = []

    for t in tokens:
        if t not in stopwords:
            keywords.add(t)
            stem = stemmer.stem(t)
            keywords.add(stem)
            content_tokens.append(t)
            stemmed_tokens.append(stem)

    for i in range(len(content_tokens) - 1):
        keywords.add(f"{content_tokens[i]} {content_tokens[i + 1]}")
        keywords.add(f"{stemmed_tokens[i]} {stemmed_tokens[i + 1]}")

    if expand_synonyms:
        from src.synonym_expander import expand_keywords

        expanded = expand_keywords(list(keywords), language)
        for syn in expanded:
            if " " not in syn:
                keywords.add(syn)
                keywords.add(stemmer.stem(syn))
            else:
                keywords.add(syn)

    return sorted(keywords)


def extract_keywords_batch_fast(
    texts: list[str], languages: list[str], expand_synonyms: bool = False
) -> list[list[str]]:
    return [
        extract_keywords_fast(text, lang, expand_synonyms=expand_synonyms)
        for text, lang in zip(texts, languages, strict=True)
    ]
