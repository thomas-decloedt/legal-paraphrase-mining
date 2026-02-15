from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_ODENET: object | None = None
_IATE_SYNONYMS: dict[str, list[str]] | None = None

_IATE_PATH = Path(__file__).parent.parent / "data" / "iate" / "synonyms.json"


def _get_odenet():
    global _ODENET
    if _ODENET is None:
        import wn

        _ODENET = wn.Wordnet("odenet")
    return _ODENET


@lru_cache(maxsize=50000)
def get_synonyms_en(word: str) -> tuple[str, ...]:
    from nltk.corpus import wordnet as wn  # type: ignore[import-untyped]

    synonyms: set[str] = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            syn = lemma.name().replace("_", " ").lower()
            if syn != word and len(syn) > 2:
                synonyms.add(syn)
                if len(synonyms) >= 5:
                    return tuple(synonyms)
    return tuple(synonyms)


@lru_cache(maxsize=50000)
def get_synonyms_de(word: str) -> tuple[str, ...]:
    de = _get_odenet()
    synonyms: set[str] = set()

    for synset in de.synsets(word):
        for w in synset.words():
            syn = w.lemma().lower()
            if syn != word.lower() and len(syn) > 2:
                synonyms.add(syn)
                if len(synonyms) >= 5:
                    return tuple(synonyms)
    return tuple(synonyms)


def _load_iate() -> dict[str, list[str]]:
    global _IATE_SYNONYMS
    if _IATE_SYNONYMS is None:
        if _IATE_PATH.exists():
            _IATE_SYNONYMS = json.loads(_IATE_PATH.read_text())
        else:
            _IATE_SYNONYMS = {}
    return _IATE_SYNONYMS


def expand_keywords(keywords: list[str], language: str) -> list[str]:
    expanded: set[str] = set(keywords)
    iate = _load_iate()

    for kw in keywords:
        if " " in kw:
            continue

        kw_lower = kw.lower()

        if language == "en":
            expanded.update(get_synonyms_en(kw_lower))
        else:
            expanded.update(get_synonyms_de(kw_lower))

        if kw_lower in iate:
            expanded.update(iate[kw_lower])

    return list(expanded)


def build_synonym_map(keywords: list[str], language: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    iate = _load_iate()

    for kw in keywords:
        if " " in kw:
            result[kw] = [kw]
            continue

        kw_lower = kw.lower()
        synonyms: set[str] = {kw_lower}

        if language == "en":
            synonyms.update(get_synonyms_en(kw_lower))
        else:
            synonyms.update(get_synonyms_de(kw_lower))

        if kw_lower in iate:
            synonyms.update(iate[kw_lower])

        result[kw] = list(synonyms)

    return result


def get_cache_stats() -> dict[str, int]:
    return {
        "en_cache_size": get_synonyms_en.cache_info().currsize,
        "de_cache_size": get_synonyms_de.cache_info().currsize,
        "iate_loaded": len(_load_iate()),
    }
