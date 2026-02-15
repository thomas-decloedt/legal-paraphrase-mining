from __future__ import annotations

from pathlib import Path

import polars as pl
from datasets import load_dataset  # type: ignore[import-untyped]
from pydantic import BaseModel


class LoadConfig(BaseModel):
    num_sentences_per_language: int = 500
    min_sentence_length: int = 30
    max_sentence_length: int = 300
    subset: str = "caselaw"
    languages: list[str] = ["en", "de"]


def load_sentences(config: LoadConfig, cache_path: Path | None = None) -> pl.DataFrame:
    if cache_path and cache_path.exists():
        return pl.read_parquet(cache_path)

    sentences: list[dict[str, str | int]] = []
    seen_texts: set[str] = set()
    sentence_id = 0

    for lang in config.languages:
        config_name = f"{lang}_{config.subset}"
        dataset = load_dataset(
            "joelniklaus/Multi_Legal_Pile",
            name=config_name,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        count = 0
        for item in dataset:
            text = item["text"]
            if not isinstance(text, str):
                continue

            for sent in _split_sentences(text):
                if not (
                    config.min_sentence_length
                    <= len(sent)
                    <= config.max_sentence_length
                ):
                    continue
                if not _is_valid_sentence(sent):
                    continue
                if sent in seen_texts:
                    continue
                seen_texts.add(sent)

                sentences.append({"id": sentence_id, "text": sent, "language": lang})
                sentence_id += 1
                count += 1

                if count >= config.num_sentences_per_language:
                    break

            if count >= config.num_sentences_per_language:
                break

    df = pl.DataFrame(sentences)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path)

    return df


def _split_sentences(text: str) -> list[str]:
    from syntok import segmenter  # type: ignore[import-untyped]

    sentences = []
    try:
        for paragraph in segmenter.process(text):
            for sentence in paragraph:
                sent_text = " ".join(token.value for token in sentence)
                cleaned = _clean_sentence(sent_text)
                if cleaned:
                    sentences.append(cleaned)
    except Exception:
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [_clean_sentence(s) for s in sentences if s.strip()]

    return sentences


def _clean_sentence(text: str) -> str:
    import re

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_valid_sentence(text: str) -> bool:
    import re

    if len(text) < 50 or len(text) > 600:
        return False

    words = text.split()
    if len(words) < 5:
        return False

    if not text[0].isupper():
        return False

    if text.rstrip()[-1] not in ".!?":
        return False

    location_pattern = re.compile(r"\([A-Z][a-z]+(?:,\s*\w+)*\)")
    location_matches = location_pattern.findall(text)
    if location_matches:
        location_chars = sum(len(m) for m in location_matches)
        location_ratio = location_chars / len(text)
        location_count = len(location_matches)
        if location_ratio > 0.35 or location_count >= 5:
            return False

    court_metadata_pattern = re.compile(
        r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s+(Agent|Judges?|Rapporteur|President)"
    )
    if court_metadata_pattern.match(text):
        return False

    procedural_patterns = [
        r"^The appellant'?s? grounds? of appeal",
        r"^Pleas? in law and main arguments",
        r"^Dismisses? the (remainder|action)",
        r"^Orders? .+ to bear .+ costs",
        r"^Grounds? of appeal and main arguments",
        r"^In support of (the|their|its) (appeal|action)",
        r"^(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth) plea in law",
        r"^Re:",
        r"(trägt|bear).*cost",
        r"Streitwert|value.*dispute",
        r"vorläufig vollstreckbar|provisionally enforceable",
        r"Sicherheitsleistung.*vollstreckbar",
    ]
    procedural_pattern = re.compile("|".join(procedural_patterns), re.IGNORECASE)
    if procedural_pattern.match(text):
        return False

    ends_with_number = re.compile(r"\d+\.?\s*$")
    citation_artifact = re.compile(r"\(\d+\)\s+\d+\.?\s*$")
    ends_with_ellipsis = re.compile(r"\.\.\.\s*$")
    ends_with_capital = re.compile(r"\s[A-Z]\.\s*$")
    incomplete_citation = re.compile(r"(Abs\.|para\.|Art\.|§)\s*$")
    truncated_ref = re.compile(r"See\s+p\.$|:\s*\(['\"]")
    if (
        ends_with_number.search(text)
        or citation_artifact.search(text)
        or ends_with_ellipsis.search(text)
        or ends_with_capital.search(text)
        or incomplete_citation.search(text)
        or truncated_ref.search(text)
    ):
        return False

    list_pattern = re.compile(r"^(\d+\.|—|•)\s+")
    if list_pattern.match(text):
        return False

    legal_citation_pattern = re.compile(
        r"(Article|Regulation|Directive|TFEU|GDPR|Council|Parliament)"
    )
    citation_tokens = [
        t for t in words if legal_citation_pattern.search(t) or re.match(r"\d+", t)
    ]
    citation_ratio = len(citation_tokens) / len(words) if words else 0
    if citation_ratio > 0.6:
        return False

    lowered = text.lower()
    skip_patterns = [
        "official journal",
        "oj c ",
        "oj l ",
        "(presidents of chambers)",
        "president of the chamber",
        "rechtsanwält",
        "case c-",
        "case t-",
        "(reference for",
        "lawyer)",
        "lawyers)",
        "avvocat",
        "defendant:",
        "defendants:",
        "applicant:",
        "form of order sought",
        "represented by",
        "(agents:",
        "commission (represented",
        "council (represented",
        "wagner centre",
        "kirchberg",
        "legal service",
        "office of",
        "zustellungsanschrift",
        "bevollmächtigte",
    ]
    if any(pattern in lowered for pattern in skip_patterns):
        return False

    date_pattern = re.compile(r"\d{1,2}[./]\s*\w+\s*\d{4}")
    if len(date_pattern.findall(text)) >= 3:
        return False

    if text.count("\n") > 2:
        return False

    has_verb_indicator = any(
        word.endswith(("ed", "ing", "es", "s", "ly")) for word in words if len(word) > 3
    )
    if not has_verb_indicator:
        return False

    return True
