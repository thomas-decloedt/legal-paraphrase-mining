from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import spacy


@dataclass
class LegalEntities:
    regulations: frozenset[str]
    articles: frozenset[str]
    cases: frozenset[str]
    dates: frozenset[str]
    quantities: frozenset[str]
    parties: frozenset[str]

    def is_empty(self) -> bool:
        return (
            not self.regulations
            and not self.articles
            and not self.cases
            and not self.dates
            and not self.quantities
            and not self.parties
        )

    def conflicts_with(self, other: LegalEntities) -> bool:
        if self.regulations and other.regulations:
            if not self.regulations & other.regulations:
                return True

        if self.articles and other.articles:
            if not self.articles & other.articles:
                return True

        if self.cases and other.cases:
            if not self.cases & other.cases:
                return True

        if self.dates and other.dates:
            if not self.dates & other.dates:
                return True

        if self.quantities and other.quantities:
            if not self.quantities & other.quantities:
                return True

        if self.parties and other.parties:
            if not self.parties & other.parties:
                return True

        return False


_REGULATION_PATTERN = re.compile(
    r"""
    (?:
        (?:Regulation|Directive|Decision)\s*
        (?:\((?:EC|EU|EEC)\)\s*)?
        (?:No\.?\s*)?
        (\d+/\d+(?:/(?:EC|EU|EEC))?)
    )
    |
    (?:
        (?:Regulation|Directive|Decision)\s+
        (\d{4}/\d+(?:/(?:EC|EU|EEC))?)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ARTICLE_PATTERN = re.compile(
    r"""
    Articles?\s+
    (\d+
        (?:\(\d+\))*
        (?:\([a-z]\))*
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CASE_PATTERN = re.compile(
    r"""
    Case\s+
    ([CcTt]-\d+/\d+)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_DATE_PATTERN = re.compile(
    r"""
    (\d{1,2}\s+
    (?:January|February|March|April|May|June|July|August|September|October|November|December)
    \s+\d{4})
    """,
    re.IGNORECASE | re.VERBOSE,
)

_QUANTITY_PATTERN = re.compile(
    r"""
    \b(
        (?:a\s+single|single|one|two|three|four|five|six|seven|eight|nine|ten|\d+)
        \s+
        (?:plea|pleas|ground|grounds|part|parts|limb|limbs|claim|claims|argument|arguments)
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


@lru_cache(maxsize=1)
def _get_spacy_nlp() -> spacy.Language:
    return spacy.load("en_core_web_sm")


def extract_legal_entities(text: str) -> LegalEntities:
    regulations: set[str] = set()
    for match in _REGULATION_PATTERN.finditer(text):
        reg = match.group(1) or match.group(2)
        if reg:
            reg = re.sub(r"/(?:EC|EU|EEC)$", "", reg, flags=re.IGNORECASE)
            regulations.add(reg.lower())

    articles: set[str] = set()
    for match in _ARTICLE_PATTERN.finditer(text):
        art = match.group(1)
        if art:
            articles.add(art.lower())

    cases: set[str] = set()
    for match in _CASE_PATTERN.finditer(text):
        case = match.group(1)
        if case:
            cases.add(case.lower())

    dates: set[str] = set()
    for match in _DATE_PATTERN.finditer(text):
        date = match.group(1)
        if date:
            normalized = " ".join(date.lower().split())
            dates.add(normalized)

    quantities: set[str] = set()
    for match in _QUANTITY_PATTERN.finditer(text):
        qty = match.group(1)
        if qty:
            normalized = " ".join(qty.lower().split())
            quantities.add(normalized)

    parties: set[str] = set()
    nlp = _get_spacy_nlp()
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ("GPE", "ORG"):
            normalized = ent.text.lower()
            if normalized.startswith("the "):
                normalized = normalized[4:]
            parties.add(normalized)

    return LegalEntities(
        regulations=frozenset(regulations),
        articles=frozenset(articles),
        cases=frozenset(cases),
        dates=frozenset(dates),
        quantities=frozenset(quantities),
        parties=frozenset(parties),
    )


def extract_legal_entities_batch(texts: list[str]) -> list[LegalEntities]:
    nlp = _get_spacy_nlp()

    all_regulations: list[set[str]] = []
    all_articles: list[set[str]] = []
    all_cases: list[set[str]] = []
    all_dates: list[set[str]] = []
    all_quantities: list[set[str]] = []

    for text in texts:
        regulations: set[str] = set()
        for match in _REGULATION_PATTERN.finditer(text):
            reg = match.group(1) or match.group(2)
            if reg:
                reg = re.sub(r"/(?:EC|EU|EEC)$", "", reg, flags=re.IGNORECASE)
                regulations.add(reg.lower())
        all_regulations.append(regulations)

        articles: set[str] = set()
        for match in _ARTICLE_PATTERN.finditer(text):
            art = match.group(1)
            if art:
                articles.add(art.lower())
        all_articles.append(articles)

        cases: set[str] = set()
        for match in _CASE_PATTERN.finditer(text):
            case = match.group(1)
            if case:
                cases.add(case.lower())
        all_cases.append(cases)

        dates: set[str] = set()
        for match in _DATE_PATTERN.finditer(text):
            date = match.group(1)
            if date:
                normalized = " ".join(date.lower().split())
                dates.add(normalized)
        all_dates.append(dates)

        quantities: set[str] = set()
        for match in _QUANTITY_PATTERN.finditer(text):
            qty = match.group(1)
            if qty:
                normalized = " ".join(qty.lower().split())
                quantities.add(normalized)
        all_quantities.append(quantities)

    # Batch process with spaCy for NER (parties)
    all_parties: list[set[str]] = []
    for doc in nlp.pipe(texts, batch_size=256):
        parties: set[str] = set()
        for ent in doc.ents:
            if ent.label_ in ("GPE", "ORG"):
                normalized = ent.text.lower()
                if normalized.startswith("the "):
                    normalized = normalized[4:]
                parties.add(normalized)
        all_parties.append(parties)

    return [
        LegalEntities(
            regulations=frozenset(all_regulations[i]),
            articles=frozenset(all_articles[i]),
            cases=frozenset(all_cases[i]),
            dates=frozenset(all_dates[i]),
            quantities=frozenset(all_quantities[i]),
            parties=frozenset(all_parties[i]),
        )
        for i in range(len(texts))
    ]


# =============================================================================
# Fast extraction (regex-based)
# =============================================================================

_PARTY_PATTERN = re.compile(
    r"""
    (?:
        # "Kingdom of X", "Republic of X", etc.
        (?:Kingdom|Republic|State|Commonwealth|Principality|Duchy)\s+of\s+
        ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)
    )
    |
    (?:
        # "European Commission", "European Parliament", etc.
        (European\s+(?:Commission|Parliament|Council|Union|Court|Community))
    )
    |
    (?:
        # Common EU institutions
        ((?:European\s+)?Court\s+of\s+(?:Justice|Auditors|First\s+Instance))
    )
    |
    (?:
        # OHIM, EUIPO, etc.
        \b(OHIM|EUIPO|WIPO|EPO)\b
    )
    """,
    re.VERBOSE,
)


def extract_legal_entities_fast(text: str) -> LegalEntities:
    regulations: set[str] = set()
    for match in _REGULATION_PATTERN.finditer(text):
        reg = match.group(1) or match.group(2)
        if reg:
            reg = re.sub(r"/(?:EC|EU|EEC)$", "", reg, flags=re.IGNORECASE)
            regulations.add(reg.lower())

    articles: set[str] = set()
    for match in _ARTICLE_PATTERN.finditer(text):
        art = match.group(1)
        if art:
            articles.add(art.lower())

    cases: set[str] = set()
    for match in _CASE_PATTERN.finditer(text):
        case = match.group(1)
        if case:
            cases.add(case.lower())

    dates: set[str] = set()
    for match in _DATE_PATTERN.finditer(text):
        date = match.group(1)
        if date:
            normalized = " ".join(date.lower().split())
            dates.add(normalized)

    quantities: set[str] = set()
    for match in _QUANTITY_PATTERN.finditer(text):
        qty = match.group(1)
        if qty:
            normalized = " ".join(qty.lower().split())
            quantities.add(normalized)

    parties: set[str] = set()
    for match in _PARTY_PATTERN.finditer(text):
        for group in match.groups():
            if group:
                normalized = group.lower().strip()
                if normalized.startswith("the "):
                    normalized = normalized[4:]
                parties.add(normalized)

    return LegalEntities(
        regulations=frozenset(regulations),
        articles=frozenset(articles),
        cases=frozenset(cases),
        dates=frozenset(dates),
        quantities=frozenset(quantities),
        parties=frozenset(parties),
    )


def extract_legal_entities_batch_fast(texts: list[str]) -> list[LegalEntities]:
    return [extract_legal_entities_fast(text) for text in texts]
