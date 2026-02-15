from __future__ import annotations

import hashlib
from pathlib import Path

import ctranslate2  # type: ignore[import-untyped]
from transformers import AutoTokenizer

from src.logging_utils import get_logger


class TranslationCache:
    def __init__(self, cache_dir: Path, direction: str):
        self.cache_dir = cache_dir / "translations" / direction
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _get_cache_path(self, text: str) -> Path:
        key = self._get_cache_key(text)
        return self.cache_dir / f"{key}.txt"

    def get(self, text: str) -> str | None:
        path = self._get_cache_path(text)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def put(self, text: str, translation: str) -> None:
        path = self._get_cache_path(text)
        path.write_text(translation, encoding="utf-8")

    def get_many(self, texts: list[str]) -> tuple[list[str | None], list[int]]:
        results: list[str | None] = []
        missing: list[int] = []

        for i, text in enumerate(texts):
            cached = self.get(text)
            results.append(cached)
            if cached is None:
                missing.append(i)

        return results, missing


class NeuralTranslator:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_en_de = TranslationCache(cache_dir, "en-de")
        self.cache_de_en = TranslationCache(cache_dir, "de-en")

        log = get_logger()
        log.info("Loading Opus-MT tokenizers...")

        self.tokenizer_en_de = AutoTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-en-de"
        )
        self.tokenizer_de_en = AutoTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-de-en"
        )

        log.info("Loading CTranslate2 models...")
        self.model_en_de = self._load_ctranslate_model(
            "Helsinki-NLP/opus-mt-en-de", "en-de"
        )
        self.model_de_en = self._load_ctranslate_model(
            "Helsinki-NLP/opus-mt-de-en", "de-en"
        )
        log.info("Translation models loaded successfully")

    def _load_ctranslate_model(
        self, hf_name: str, direction: str
    ) -> ctranslate2.Translator:
        ct2_path = self.cache_dir / f"ct2_{direction}"

        if not ct2_path.exists():
            from ctranslate2 import converters

            log = get_logger()
            log.info(f"Converting {hf_name} to CTranslate2 format (one-time setup)...")
            converter = converters.TransformersConverter(hf_name)
            converter.convert(str(ct2_path), quantization="int8")
            log.info(f"Model converted and saved to {ct2_path}")

        return ctranslate2.Translator(str(ct2_path))

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
        batch_size: int = 32,
    ) -> list[str]:
        if not texts:
            return []

        if source_lang == "en":
            tokenizer = self.tokenizer_en_de
            model = self.model_en_de
            cache = self.cache_en_de
        else:
            tokenizer = self.tokenizer_de_en
            model = self.model_de_en
            cache = self.cache_de_en

        cached_results, missing_indices = cache.get_many(texts)

        if not missing_indices:
            return [r for r in cached_results if r is not None]

        missing_texts = [texts[i] for i in missing_indices]

        inputs = [
            tokenizer.convert_ids_to_tokens(tokenizer.encode(t)) for t in missing_texts
        ]

        results = model.translate_batch(
            inputs, beam_size=1, batch_type="examples", max_batch_size=batch_size
        )

        translations = [
            tokenizer.decode(tokenizer.convert_tokens_to_ids(r.hypotheses[0]))
            for r in results
        ]

        for i, idx in enumerate(missing_indices):
            cache.put(texts[idx], translations[i])
            cached_results[idx] = translations[i]

        return [r for r in cached_results if r is not None]
