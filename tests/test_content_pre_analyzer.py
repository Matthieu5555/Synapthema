"""Tests for regex-based content signal detection and document type detection."""

from src.extraction.types import Book, Chapter, Section
from src.transformation.content_pre_analyzer import (
    DOCUMENT_TYPE_TEMPLATE_WEIGHTS,
    ChapterSignals,
    SectionSignals,
    analyze_chapter_sections,
    analyze_section,
    detect_document_type,
    format_document_type_guidance,
)


class TestFormulaDetection:
    """Formula pattern detection in section text."""

    def test_latex_inline_math(self) -> None:
        text = "The formula $E = mc^2$ relates energy to mass."
        signals = analyze_section("Physics", text)
        assert signals.formula_count >= 1

    def test_latex_display_math(self) -> None:
        text = "The equation is:\n$$PV = \\frac{FV}{(1+r)^n}$$\nwhere r is the rate."
        signals = analyze_section("Finance", text)
        assert signals.formula_count >= 1

    def test_latex_commands(self) -> None:
        text = r"We use \frac{a}{b} and \sum_{i=1}^{n} x_i to compute."
        signals = analyze_section("Math", text)
        assert signals.formula_count >= 2

    def test_equation_patterns(self) -> None:
        text = "The Sharpe ratio is SR = (Rp - Rf) / sigma. Also CAPM = Rf + beta * (Rm - Rf)."
        signals = analyze_section("Finance", text)
        assert signals.formula_count >= 1

    def test_no_currency_false_positive(self) -> None:
        """Dollar signs used as currency should not count as formulas."""
        text = "The cost is $100 and the revenue is $200."
        signals = analyze_section("Business", text)
        assert signals.formula_count == 0

    def test_no_formulas(self) -> None:
        text = "This is a plain text section about history and culture."
        signals = analyze_section("History", text)
        assert signals.formula_count == 0


class TestProceduralDetection:
    """Procedural language detection."""

    def test_step_numbers(self) -> None:
        text = "Step 1: Calculate the mean. Step 2: Find the variance."
        signals = analyze_section("Stats", text)
        assert signals.procedural_count >= 2

    def test_ordinals(self) -> None:
        text = "First, identify the risk. Second, assess the impact."
        signals = analyze_section("Risk", text)
        assert signals.procedural_count >= 2

    def test_procedure_keyword(self) -> None:
        text = "The procedure involves several stages. Follow the algorithm below."
        signals = analyze_section("Method", text)
        assert signals.procedural_count >= 2

    def test_numbered_instructions(self) -> None:
        text = "1) Calculate the NPV. 2) Determine the IRR. 3) Compare results."
        signals = analyze_section("Finance", text)
        assert signals.procedural_count >= 2

    def test_no_procedural(self) -> None:
        text = "Markets are efficient when prices reflect all available information."
        signals = analyze_section("EMH", text)
        assert signals.procedural_count == 0


class TestComparisonDetection:
    """Comparison language detection."""

    def test_whereas(self) -> None:
        text = "Stocks provide growth, whereas bonds provide income."
        signals = analyze_section("Assets", text)
        assert signals.comparison_count >= 1

    def test_in_contrast(self) -> None:
        text = "In contrast to active management, passive strategies track indices."
        signals = analyze_section("Investing", text)
        assert signals.comparison_count >= 1

    def test_versus(self) -> None:
        text = "Value versus growth is a classic factor debate."
        signals = analyze_section("Factors", text)
        assert signals.comparison_count >= 1

    def test_differs_from(self) -> None:
        text = "This approach differs from the traditional method."
        signals = analyze_section("Methods", text)
        assert signals.comparison_count >= 1

    def test_pros_and_cons(self) -> None:
        text = "Let's examine the pros and cons of this strategy."
        signals = analyze_section("Strategy", text)
        assert signals.comparison_count >= 1

    def test_no_comparisons(self) -> None:
        text = "The concept of present value is fundamental to finance."
        signals = analyze_section("PV", text)
        assert signals.comparison_count == 0


class TestDefinitionDetection:
    """Definition pattern detection."""

    def test_is_defined_as(self) -> None:
        text = "Duration is defined as the weighted average time to receive cash flows."
        signals = analyze_section("Bonds", text)
        assert signals.definition_count >= 1

    def test_refers_to(self) -> None:
        text = "Liquidity refers to how easily an asset can be converted to cash."
        signals = analyze_section("Markets", text)
        assert signals.definition_count >= 1

    def test_known_as(self) -> None:
        text = "This is also known as the time value of money."
        signals = analyze_section("TVM", text)
        assert signals.definition_count >= 1

    def test_is_a_measure(self) -> None:
        text = "Volatility is a measure of the dispersion of returns."
        signals = analyze_section("Risk", text)
        assert signals.definition_count >= 1


class TestExampleDetection:
    """Example pattern detection."""

    def test_for_example(self) -> None:
        text = "For example, if the interest rate rises, bond prices fall."
        signals = analyze_section("Bonds", text)
        assert signals.example_count >= 1

    def test_consider(self) -> None:
        text = "Consider a portfolio with 60% stocks and 40% bonds."
        signals = analyze_section("Portfolio", text)
        assert signals.example_count >= 1

    def test_suppose(self) -> None:
        text = "Suppose an investor has $10,000 to invest for 5 years."
        signals = analyze_section("Investing", text)
        assert signals.example_count >= 1

    def test_case_study(self) -> None:
        text = "This case study examines the 2008 financial crisis."
        signals = analyze_section("Crisis", text)
        assert signals.example_count >= 1

    def test_real_world(self) -> None:
        text = "In a real-world scenario, transaction costs matter."
        signals = analyze_section("Practice", text)
        assert signals.example_count >= 1


class TestKeyTermExtraction:
    """Key term extraction from text."""

    def test_bold_terms(self) -> None:
        text = "The **Sharpe ratio** is calculated using **standard deviation**."
        signals = analyze_section("Risk", text)
        assert "Sharpe ratio" in signals.key_terms
        assert "standard deviation" in signals.key_terms

    def test_capitalized_phrases(self) -> None:
        text = "Modern Portfolio Theory builds on Capital Asset Pricing Model."
        signals = analyze_section("Finance", text)
        lower_terms = [t.lower() for t in signals.key_terms]
        assert "modern portfolio theory" in lower_terms or "capital asset pricing" in lower_terms

    def test_deduplication(self) -> None:
        text = "The **Sharpe Ratio** and the Sharpe Ratio are the same thing."
        signals = analyze_section("Risk", text)
        lower_terms = [t.lower() for t in signals.key_terms]
        assert lower_terms.count("sharpe ratio") == 1

    def test_max_terms_cap(self) -> None:
        """Should not return more than 20 terms."""
        # Generate text with many capitalized phrases
        phrases = [f"Important Concept {chr(65+i)}" for i in range(25)]
        text = ". ".join(phrases) + "."
        signals = analyze_section("Many", text)
        assert len(signals.key_terms) <= 20


class TestSectionSignalsDominantType:
    """SectionSignals.dominant_type property."""

    def test_procedural_dominant(self) -> None:
        signals = SectionSignals(
            section_title="Steps",
            procedural_count=10,
            definition_count=2,
        )
        assert signals.dominant_type == "procedural"

    def test_conceptual_dominant(self) -> None:
        signals = SectionSignals(
            section_title="Defs",
            definition_count=8,
            example_count=1,
        )
        assert signals.dominant_type == "conceptual"

    def test_mixed_when_no_clear_winner(self) -> None:
        signals = SectionSignals(
            section_title="Mixed",
            formula_count=3,
            procedural_count=3,
            comparison_count=3,
            definition_count=3,
            example_count=3,
        )
        assert signals.dominant_type == "mixed"

    def test_mixed_when_empty(self) -> None:
        signals = SectionSignals(section_title="Empty")
        assert signals.dominant_type == "mixed"


class TestChapterSignals:
    """ChapterSignals aggregation."""

    def test_aggregates_totals(self) -> None:
        s1 = SectionSignals(
            section_title="S1",
            formula_count=3,
            definition_count=2,
            key_terms=("term1", "term2"),
        )
        s2 = SectionSignals(
            section_title="S2",
            formula_count=1,
            example_count=4,
            key_terms=("term2", "term3"),  # term2 overlaps
        )
        chapter = ChapterSignals(
            chapter_title="Ch1",
            sections=(s1, s2),
        )
        assert chapter.total_formulas == 4
        assert chapter.total_definitions == 2
        assert chapter.total_examples == 4
        # Key terms deduplicated
        assert len(chapter.all_key_terms) == 3

    def test_empty_chapter(self) -> None:
        chapter = ChapterSignals(chapter_title="Empty")
        assert chapter.total_formulas == 0
        assert chapter.all_key_terms == ()


class TestAnalyzeChapterSections:
    """Integration: analyze_chapter_sections function."""

    def test_end_to_end(self) -> None:
        sections = [
            ("Definitions", "Duration is defined as the weighted average time. Convexity refers to the curvature."),
            ("Calculation", "Step 1: Calculate duration. Step 2: Adjust for convexity. The formula $D = \\sum w_i t_i$."),
        ]
        result = analyze_chapter_sections("Bond Math", sections)
        assert result.chapter_title == "Bond Math"
        assert len(result.sections) == 2
        assert result.total_definitions >= 2
        assert result.total_formulas >= 1
        assert result.total_procedures >= 2


# ── Document type detection tests ──────────────────────────────────────────


def _make_book(text: str, title: str = "Test Book") -> Book:
    """Create a minimal Book with a single section containing the given text."""
    return Book(
        title=title,
        author="Test",
        total_pages=10,
        chapters=(
            Chapter(
                chapter_number=1,
                title="Chapter 1",
                start_page=1,
                end_page=10,
                sections=(
                    Section(
                        title="Section 1",
                        level=2,
                        start_page=1,
                        end_page=10,
                        text=text,
                    ),
                ),
            ),
        ),
    )


class TestDetectDocumentType:
    """Document-level type classification."""

    def test_quantitative_from_formulas_and_keywords(self) -> None:
        text = (
            "The theorem states that $E = mc^2$. We can calculate the derivative "
            "using the formula \\frac{d}{dx}. The proof follows from the equation "
            "for variance $\\sigma^2 = E[X^2] - (E[X])^2$. "
            "The coefficient matrix determines the solution."
        )
        book = _make_book(text)
        assert detect_document_type(book) == "quantitative"

    def test_narrative_from_historical_prose(self) -> None:
        text = (
            "In the 19th century, the Industrial Revolution transformed European society. "
            "The movement was founded on principles of mechanization and mass production. "
            "This era saw the development of railways and the discovery of new energy sources. "
            "The story of this period begins with the invention of the steam engine. "
            "Throughout this century, society was fundamentally changed by these developments. "
            "The revolution in manufacturing developed rapidly across the continent."
        )
        book = _make_book(text)
        assert detect_document_type(book) == "narrative"

    def test_procedural_from_steps(self) -> None:
        text = (
            "Step 1: Install the software. Step 2: Configure the settings. "
            "Step 3: Deploy the application. The procedure requires following these "
            "steps in order. The algorithm processes data in a specific workflow. "
            "First, execute the initialization process. Second, implement the main loop."
        )
        book = _make_book(text)
        assert detect_document_type(book) == "procedural"

    def test_analytical_from_comparison_language(self) -> None:
        text = (
            "This analysis compares three frameworks for evaluation. "
            "The criteria for assessment include cost and efficiency. "
            "We evaluate each methodology against established criteria. "
            "Our findings suggest that the analysis reveals significant differences. "
            "The hypothesis is tested by comparing the contrast between approaches."
        )
        book = _make_book(text)
        assert detect_document_type(book) == "analytical"

    def test_regulatory_from_compliance_language(self) -> None:
        text = (
            "This regulation establishes the compliance requirements for all entities. "
            "The statute mandates adherence to the standard set forth in the provision. "
            "Enforcement of this legislation requires meeting specific requirements. "
            "The mandate covers all aspects of regulatory compliance. "
            "Each provision of the ordinance must be followed strictly."
        )
        book = _make_book(text)
        assert detect_document_type(book) == "regulatory"

    def test_mixed_for_ambiguous_content(self) -> None:
        text = "This is a short text with no clear signals."
        book = _make_book(text)
        assert detect_document_type(book) == "mixed"

    def test_empty_book_returns_mixed(self) -> None:
        book = Book(
            title="Empty", author="Nobody", total_pages=0, chapters=(),
        )
        assert detect_document_type(book) == "mixed"


class TestDocumentTypeTemplateWeights:
    """Template weight maps for each document type."""

    def test_all_types_have_weights(self) -> None:
        expected_types = {"quantitative", "narrative", "procedural", "analytical", "regulatory", "mixed"}
        assert set(DOCUMENT_TYPE_TEMPLATE_WEIGHTS.keys()) == expected_types

    def test_weights_sum_to_approximately_one(self) -> None:
        for doc_type, weights in DOCUMENT_TYPE_TEMPLATE_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, f"{doc_type} weights sum to {total}, expected ~1.0"


class TestFormatDocumentTypeGuidance:
    """format_document_type_guidance() output."""

    def test_returns_formatted_string(self) -> None:
        result = format_document_type_guidance("quantitative")
        assert "worked_example" in result
        assert "30%" in result

    def test_mixed_fallback(self) -> None:
        result = format_document_type_guidance("mixed")
        assert "analogy_first" in result
