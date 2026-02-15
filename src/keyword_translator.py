from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

from compound_split.char_split import split_compound  # type: ignore[import-untyped]

_DICT_DIR = Path(__file__).parent.parent / "data" / "dictionaries"

_COMPOUND_SPLIT_THRESHOLD = 0.5


def _split_german_compound(word: str) -> list[str]:
    if len(word) < 6:
        return [word]

    splits = split_compound(word)
    if not splits or splits[0][0] < _COMPOUND_SPLIT_THRESHOLD:
        return [word]

    _, part1, part2 = splits[0]

    # Clean up parts: lowercase and remove linking 's' (Fugen-s)
    parts = []
    for part in [part1, part2]:
        part = part.lower()
        if part.endswith("s") and len(part) > 3:
            parts.append(part[:-1])
        parts.append(part)

    return list(set(parts))


@lru_cache(maxsize=2)
def _load_muse_dict(direction: str) -> dict[str, list[str]]:
    """Load MUSE bilingual dictionary.
    Source: https://github.com/facebookresearch/MUSE
    """
    path = _DICT_DIR / f"{direction}.txt"
    if not path.exists():
        return {}

    translations: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                src, tgt = parts[0].lower(), parts[1].lower()
                if src not in translations:
                    translations[src] = []
                if tgt not in translations[src]:
                    translations[src].append(tgt)
    return translations


def _load_word2word(src_lang: str, tgt_lang: str) -> dict[str, list[str]]:
    """Load word2word bilingual dictionary.
    Source: https://github.com/kakaobrain/word2word
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from word2word import Word2word  # type: ignore[import-untyped]

            w2w = Word2word(src_lang, tgt_lang)
            word2x = w2w.word2x
            if word2x is None:
                return {}
            return {
                word.lower(): [t.lower() for t in w2w(word)] for word in word2x.keys()
            }
    except Exception:
        return {}


def _merge_dicts(
    *dicts: dict[str, list[str]],
) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for d in dicts:
        for word, translations in d.items():
            if word not in merged:
                merged[word] = []
            for t in translations:
                if t not in merged[word]:
                    merged[word].append(t)
    return merged


def _add_stemmed_entries(
    dictionary: dict[str, list[str]],
    src_lang: str,
    tgt_lang: str,
) -> dict[str, list[str]]:
    from src.keyword_extractor import stem_word

    expanded: dict[str, list[str]] = {}

    for src_word, translations in dictionary.items():
        src_stem = stem_word(src_word, src_lang)

        all_translations: set[str] = set(translations)
        for t in translations:
            all_translations.add(stem_word(t, tgt_lang))

        trans_list = list(all_translations)

        if src_word not in expanded:
            expanded[src_word] = []
        for t in trans_list:
            if t not in expanded[src_word]:
                expanded[src_word].append(t)

        if src_stem != src_word:
            if src_stem not in expanded:
                expanded[src_stem] = []
            for t in trans_list:
                if t not in expanded[src_stem]:
                    expanded[src_stem].append(t)

    return expanded


class KeywordTranslator:
    def __init__(self, cache_dir: Path | None = None):
        muse_en_de = _load_muse_dict("en-de")
        muse_de_en = _load_muse_dict("de-en")
        w2w_en_de = _load_word2word("en", "de")
        w2w_de_en = _load_word2word("de", "en")

        merged_en_de = _merge_dicts(muse_en_de, w2w_en_de)
        merged_de_en = _merge_dicts(muse_de_en, w2w_de_en)

        self._en_de_dict = _add_stemmed_entries(merged_en_de, "en", "de")
        self._de_en_dict = _add_stemmed_entries(merged_de_en, "de", "en")

    def _translate_word(
        self, word: str, dictionary: dict[str, list[str]], is_german: bool
    ) -> list[str]:
        if word in dictionary:
            return dictionary[word]

        if is_german:
            parts = _split_german_compound(word)
            if len(parts) > 1:
                all_translations: list[str] = []
                for part in parts:
                    all_translations.extend(dictionary.get(part, [part]))
                return list(set(all_translations))

        return [word]

    def translate_batch(
        self,
        keywords: list[str],
        direction: str = "en-de",
        enable_bigram_permutations: bool = False,
    ) -> dict[str, list[str]]:
        from itertools import permutations, product

        dictionary = self._en_de_dict if direction == "en-de" else self._de_en_dict
        is_german = direction == "de-en"

        results: dict[str, list[str]] = {}
        for kw in keywords:
            kw_lower = kw.lower()
            if " " in kw_lower:
                words = kw_lower.split()
                word_translations = [
                    self._translate_word(w, dictionary, is_german) for w in words
                ]
                all_translations: set[str] = set()
                for combo in product(*word_translations):
                    all_translations.add(" ".join(combo))
                    if enable_bigram_permutations:
                        for perm in permutations(combo):
                            all_translations.add(" ".join(perm))
                    all_translations.update(combo)
                results[kw] = list(all_translations)
            else:
                results[kw] = self._translate_word(kw_lower, dictionary, is_german)

        return results

    def get_cache_stats(self) -> dict[str, int]:
        return {
            "en_de_entries": len(self._en_de_dict),
            "de_en_entries": len(self._de_en_dict),
        }
