"""Tests for legal entity extraction and conflict detection."""

from src.legal_entities import extract_legal_entities


class TestExtractLegalEntities:
    """Test legal entity extraction patterns."""

    def test_extracts_regulation_with_number(self) -> None:
        text = "infringement of Regulation 207/2009"
        entities = extract_legal_entities(text)
        assert "207/2009" in entities.regulations

    def test_extracts_regulation_with_ec_suffix(self) -> None:
        text = "Regulation (EC) No 874/2004"
        entities = extract_legal_entities(text)
        assert "874/2004" in entities.regulations

    def test_extracts_directive(self) -> None:
        text = "Directive 79/409/EEC on wild birds"
        entities = extract_legal_entities(text)
        assert "79/409" in entities.regulations

    def test_extracts_article_singular(self) -> None:
        text = "Article 215(2) TFEU"
        entities = extract_legal_entities(text)
        assert "215(2)" in entities.articles

    def test_extracts_article_plural(self) -> None:
        text = "Articles 8(1)(b) and 8(5) of Council Regulation"
        entities = extract_legal_entities(text)
        assert "8(1)(b)" in entities.articles

    def test_extracts_case_reference(self) -> None:
        text = "Case C-123/45 concerning"
        entities = extract_legal_entities(text)
        assert "c-123/45" in entities.cases

    def test_extracts_date(self) -> None:
        text = "expired on 28 October 2001"
        entities = extract_legal_entities(text)
        assert "28 october 2001" in entities.dates

    def test_extracts_quantity_two_pleas(self) -> None:
        text = "the applicant relies on two pleas in law"
        entities = extract_legal_entities(text)
        assert "two pleas" in entities.quantities

    def test_extracts_quantity_single_ground(self) -> None:
        text = "the appellant puts forward a single ground of appeal"
        entities = extract_legal_entities(text)
        assert "a single ground" in entities.quantities

    def test_extracts_quantity_three_parts(self) -> None:
        text = "a single plea divided into three parts"
        entities = extract_legal_entities(text)
        assert "three parts" in entities.quantities

    def test_extracts_kingdom_of_spain(self) -> None:
        text = "Orders the Kingdom of Spain to pay the costs"
        entities = extract_legal_entities(text)
        assert "kingdom of spain" in entities.parties

    def test_extracts_european_commission(self) -> None:
        text = "Orders the European Commission to bear its own costs"
        entities = extract_legal_entities(text)
        assert "european commission" in entities.parties

    def test_extracts_federal_republic(self) -> None:
        text = "Orders the Federal Republic of Germany to pay the costs"
        entities = extract_legal_entities(text)
        assert "federal republic of germany" in entities.parties


class TestConflictsWithRegulations:
    """Test conflict detection for different regulations - previously false positives."""

    def test_different_regulations_conflict(self) -> None:
        """Regulation 207/2009 vs 1049/2001 should conflict."""
        a = extract_legal_entities("Plea alleging infringement of Regulation 207/2009")
        b = extract_legal_entities("Plea alleging infringement of Regulation 1049/2001")
        assert a.conflicts_with(b)

    def test_same_regulation_no_conflict(self) -> None:
        """Same regulation should not conflict."""
        a = extract_legal_entities("infringement of Regulation 40/94")
        b = extract_legal_entities("Breach of Article 7(1)(b) of Regulation No 40/94")
        assert not a.conflicts_with(b)

    def test_no_regulation_no_conflict(self) -> None:
        """If one side has no regulation, no conflict."""
        a = extract_legal_entities("The applicant submits")
        b = extract_legal_entities("infringement of Regulation 207/2009")
        assert not a.conflicts_with(b)


class TestConflictsWithArticles:
    """Test conflict detection for different articles - previously false positives."""

    def test_different_articles_conflict(self) -> None:
        """Article 215(2) vs Article 9 should conflict."""
        a = extract_legal_entities("Article 215(2) TFEU")
        b = extract_legal_entities("infringement of Article 9")
        assert a.conflicts_with(b)

    def test_shared_article_no_conflict(self) -> None:
        """Shared Article 8(1)(b) should not conflict."""
        a = extract_legal_entities("Articles 8(1)(b) and 8(5) of Council Regulation")
        b = extract_legal_entities("Infringement of Article 43(2) and Article 8(1)(b)")
        assert not a.conflicts_with(b)

    def test_no_articles_no_conflict(self) -> None:
        """If one side has no articles, no conflict."""
        a = extract_legal_entities("The Council failed")
        b = extract_legal_entities("Article 4(1)(a)")
        assert not a.conflicts_with(b)


class TestConflictsWithDates:
    """Test conflict detection for different dates - previously false positives."""

    def test_different_dates_conflict(self) -> None:
        """28 October 2001 vs 2 December 2002 should conflict."""
        a = extract_legal_entities(
            "The period within which the directive had to be transposed expired on 28 October 2001."
        )
        b = extract_legal_entities(
            "The period within which the directive had to be transposed expired on 2 December 2002."
        )
        assert a.conflicts_with(b)

    def test_same_date_no_conflict(self) -> None:
        """Same date should not conflict."""
        a = extract_legal_entities("expired on 28 October 2001")
        b = extract_legal_entities("deadline of 28 October 2001")
        assert not a.conflicts_with(b)

    def test_no_date_no_conflict(self) -> None:
        """If one side has no date, no conflict."""
        a = extract_legal_entities("The directive was not transposed")
        b = extract_legal_entities("expired on 28 October 2001")
        assert not a.conflicts_with(b)


class TestPreviousFalsePositives:
    """Regression tests for previously identified false positive clusters."""

    def test_regulation_template_cluster(self) -> None:
        """
        Previously matched as paraphrases but are about different regulations:
        - Regulation 207/2009
        - Regulation 1049/2001
        - Directive 2006/112/EC
        """
        texts = [
            "Plea alleging infringement of Regulation 207/2009",
            "Plea alleging infringement of Regulation 1049/2001",
            "Plea alleging infringement of Directive 2006/112/EC",
        ]
        entities = [extract_legal_entities(t) for t in texts]

        # All pairs should conflict
        assert entities[0].conflicts_with(entities[1])
        assert entities[0].conflicts_with(entities[2])
        assert entities[1].conflicts_with(entities[2])

    def test_transposition_date_cluster(self) -> None:
        """
        Previously matched as paraphrases but have different transposition dates:
        - 28 October 2001
        - 2 December 2002
        - 20 January 2007
        - 21 December 2007
        """
        texts = [
            "The period within which the directive had to be transposed expired on 28 October 2001.",
            "The period within which the directive had to be transposed expired on 2 December 2002.",
            "The period within which the directive had to be transposed expired on 20 January 2007.",
            "The period within which the directive had to be transposed expired on 21 December 2007.",
        ]
        entities = [extract_legal_entities(t) for t in texts]

        # All pairs should conflict (different dates)
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                assert entities[i].conflicts_with(entities[j]), (
                    f"Pair {i},{j} should conflict"
                )

    def test_article_template_cluster(self) -> None:
        """
        Previously matched but reference different articles:
        - Articles 8(1)(b) and 8(5)
        - Article 215(2)
        - Article 9
        """
        seed = extract_legal_entities(
            "The present application is aimed at showing that the General Court erred "
            "in concluding that the grounds of invalidity set forth in Articles 8(1)(b) "
            "and 8(5) of Council Regulation (EC) No."
        )
        para1 = extract_legal_entities(
            "The applicant submits that as the contested decision was invalid, "
            "the Council could not rely on Article 215(2) TFEU to enact the contested regulation."
        )
        para2 = extract_legal_entities(
            "In support of its application the applicant alleges infringement "
            "of Article 9 of Regulation (EC) No 874/2004 (1)."
        )

        # These should conflict (different articles)
        assert seed.conflicts_with(para1)
        assert seed.conflicts_with(para2)


class TestConflictsWithQuantities:
    """Test conflict detection for different quantities - previously false positives."""

    def test_different_plea_counts_conflict(self) -> None:
        """'two pleas' vs 'four pleas' should conflict."""
        a = extract_legal_entities("the applicant relies on two pleas in law")
        b = extract_legal_entities("the applicant relies on four pleas in law")
        assert a.conflicts_with(b)

    def test_single_vs_multiple_conflict(self) -> None:
        """'single ground' vs 'two grounds' should conflict."""
        a = extract_legal_entities(
            "the appellant puts forward a single ground of appeal"
        )
        b = extract_legal_entities("The appellant relies on two grounds of appeal")
        assert a.conflicts_with(b)

    def test_same_quantity_no_conflict(self) -> None:
        """Same quantity should not conflict."""
        a = extract_legal_entities("relies on two pleas in law")
        b = extract_legal_entities("puts forward two pleas")
        assert not a.conflicts_with(b)

    def test_no_quantity_no_conflict(self) -> None:
        """If one side has no quantity, no conflict."""
        a = extract_legal_entities("The applicant submits")
        b = extract_legal_entities("relies on two pleas")
        assert not a.conflicts_with(b)

    def test_plea_cluster_regression(self) -> None:
        """
        Previously matched as paraphrases but have different plea counts:
        - single plea divided into three parts
        - two pleas
        - four pleas
        - six pleas
        """
        texts = [
            "the applicant relies on a single plea divided into three parts",
            "the applicant relies on two pleas in law",
            "the applicant relies on four pleas in law",
            "the applicant relies on six pleas in law",
        ]
        entities = [extract_legal_entities(t) for t in texts]

        # All pairs should conflict (different quantities)
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                assert entities[i].conflicts_with(entities[j]), (
                    f"Pair {i},{j} should conflict"
                )


class TestConflictsWithParties:
    """Test conflict detection for different parties - previously false positives."""

    def test_different_countries_conflict(self) -> None:
        """Kingdom of Spain vs Kingdom of Belgium should conflict."""
        a = extract_legal_entities("Orders the Kingdom of Spain to pay the costs")
        b = extract_legal_entities("Orders the Kingdom of Belgium to pay the costs")
        assert a.conflicts_with(b)

    def test_spain_vs_netherlands_conflict(self) -> None:
        """Kingdom of Spain vs Kingdom of the Netherlands should conflict."""
        a = extract_legal_entities("Orders the Kingdom of Spain to pay")
        b = extract_legal_entities("Order the Kingdom of the Netherlands to pay")
        assert a.conflicts_with(b)

    def test_same_country_no_conflict(self) -> None:
        """Same country should not conflict."""
        a = extract_legal_entities("Orders the Kingdom of Spain to pay the costs")
        b = extract_legal_entities("The Kingdom of Spain is ordered to bear its costs")
        assert not a.conflicts_with(b)

    def test_spain_and_commission_vs_spain_only_no_conflict(self) -> None:
        """Spain + Commission vs just Spain should NOT conflict (they share Spain)."""
        a = extract_legal_entities(
            "Orders the Kingdom of Spain and the European Commission to bear their own costs"
        )
        b = extract_legal_entities("Orders the Kingdom of Spain to pay the costs")
        assert not a.conflicts_with(b)

    def test_no_party_no_conflict(self) -> None:
        """If one side has no party, no conflict."""
        a = extract_legal_entities("The applicant submits")
        b = extract_legal_entities("Orders the Kingdom of Spain to pay")
        assert not a.conflicts_with(b)

    def test_country_cluster_regression(self) -> None:
        """
        Previously matched as paraphrases but have different countries:
        - Kingdom of Belgium
        - Kingdom of the Netherlands
        """
        a = extract_legal_entities("Orders the Kingdom of Belgium to pay the costs")
        b = extract_legal_entities(
            "Order the Kingdom of the Netherlands to pay the costs"
        )
        assert a.conflicts_with(b)
