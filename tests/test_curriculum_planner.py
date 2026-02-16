"""Tests for the curriculum planner — Stage 1.5.

Tests plan_curriculum(), passthrough fallback, content summary builder,
and chapter/section matching helpers.
"""

from __future__ import annotations

from typing import TypeVar

from src.extraction.types import Book, Chapter, Section
import pytest

from src.transformation.curriculum_planner import (
    _build_content_summary,
    _build_rich_content_summary,
    _extract_chapter_number_from_title,
    _split_overloaded_sections,
    find_matching_chapter,
    find_matching_section,
    plan_curriculum,
    validate_progression,
)
from src.transformation.types import (
    CurriculumBlueprint,
    ModuleBlueprint,
    SectionBlueprint,
)

T = TypeVar("T")


# ── Test fixtures ─────────────────────────────────────────────────────────────


def _make_section(title: str = "Test Section", text: str = "A" * 200) -> Section:
    return Section(
        title=title, level=2, start_page=1, end_page=5, text=text,
    )


def _make_chapter(
    number: int = 1,
    title: str = "Test Chapter",
    sections: tuple[Section, ...] | None = None,
) -> Chapter:
    if sections is None:
        sections = (_make_section(),)
    return Chapter(
        chapter_number=number,
        title=title,
        start_page=1,
        end_page=10,
        sections=sections,
    )


def _make_book(chapters: tuple[Chapter, ...] | None = None) -> Book:
    if chapters is None:
        chapters = (
            _make_chapter(
                number=1,
                title="Introduction to Testing",
                sections=(
                    _make_section("What is Testing"),
                    _make_section("Why Test"),
                ),
            ),
            _make_chapter(
                number=2,
                title="Advanced Testing",
                sections=(
                    _make_section("Unit Tests"),
                    _make_section("Integration Tests"),
                ),
            ),
        )
    return Book(
        title="The Testing Handbook",
        author="Test Author",
        total_pages=100,
        chapters=chapters,
    )


class MockPlannerClient:
    """Returns a fixed CurriculumBlueprint from complete_structured()."""

    def __init__(self, blueprint: CurriculumBlueprint | None = None) -> None:
        self._blueprint = blueprint or CurriculumBlueprint(
            course_title="Mock Course",
            course_summary="A mock course.",
            learner_journey="Module 1 → Module 2",
            modules=[
                ModuleBlueprint(
                    title="Introduction to Testing",
                    source_chapter_number=1,
                    summary="Intro module",
                    sections=[
                        SectionBlueprint(
                            title="What is Testing",
                            source_section_title="What is Testing",
                            learning_objectives=["Define testing"],
                            template="narrative",
                            bloom_target="remember",
                        ),
                        SectionBlueprint(
                            title="Why Test",
                            source_section_title="Why Test",
                            learning_objectives=["Explain importance of testing"],
                            template="analogy_first",
                            bloom_target="understand",
                        ),
                    ],
                ),
                ModuleBlueprint(
                    title="Advanced Testing",
                    source_chapter_number=2,
                    summary="Advanced module",
                    sections=[
                        SectionBlueprint(
                            title="Unit Tests",
                            source_section_title="Unit Tests",
                            learning_objectives=["Write a unit test"],
                            template="worked_example",
                            bloom_target="apply",
                        ),
                    ],
                ),
            ],
        )
        self.call_count = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return "mock"

    def complete_light(self, system_prompt: str, user_prompt: str) -> str:
        return self.complete(system_prompt, user_prompt)

    def complete_structured(
        self, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> T:
        self.call_count += 1
        return self._blueprint  # type: ignore[return-value]

    def complete_structured_light(
        self, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> T:
        return self.complete_structured(system_prompt, user_prompt, response_model)


class FailingPlannerClient:
    """Always raises an error."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("fail")

    def complete_light(self, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("fail")

    def complete_structured(
        self, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> T:
        from src.transformation.llm_client import LLMError
        raise LLMError("Planning failed")

    def complete_structured_light(
        self, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> T:
        from src.transformation.llm_client import LLMError
        raise LLMError("Planning failed")


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestPlanCurriculum:
    """Tests for the plan_curriculum() entry point."""

    def test_returns_blueprint_from_llm(self) -> None:
        book = _make_book()
        client = MockPlannerClient()

        blueprint = plan_curriculum(book, client)

        assert blueprint.course_title == "Mock Course"
        assert len(blueprint.modules) == 2
        assert client.call_count == 1

    def test_raises_on_llm_failure(self) -> None:
        from src.transformation.llm_client import LLMError

        book = _make_book()
        client = FailingPlannerClient()

        with pytest.raises(LLMError):
            plan_curriculum(book, client)


class TestBuildContentSummary:
    """Tests for the content summary builder."""

    def test_includes_chapter_and_section_titles(self) -> None:
        book = _make_book()

        summary = _build_content_summary(book)

        assert "Introduction to Testing" in summary
        assert "Advanced Testing" in summary
        assert "What is Testing" in summary
        assert "Unit Tests" in summary

    def test_includes_page_ranges(self) -> None:
        book = _make_book()

        summary = _build_content_summary(book)

        assert "pp." in summary

    def test_includes_document_metadata(self) -> None:
        book = _make_book()

        summary = _build_content_summary(book)

        assert "The Testing Handbook" in summary
        assert "Test Author" in summary


class TestFindMatchingChapter:
    """Tests for the chapter matching helper."""

    def test_matches_by_chapter_number(self) -> None:
        book = _make_book()
        module_bp = ModuleBlueprint(
            title="Whatever",
            source_chapter_number=2,
        )

        chapter = find_matching_chapter(book, module_bp)

        assert chapter is not None
        assert chapter.chapter_number == 2
        assert chapter.title == "Advanced Testing"

    def test_matches_by_title_fallback(self) -> None:
        book = _make_book()
        module_bp = ModuleBlueprint(
            title="Introduction to Testing",
            source_chapter_number=None,
        )

        chapter = find_matching_chapter(book, module_bp)

        assert chapter is not None
        assert chapter.chapter_number == 1

    def test_returns_none_when_no_match(self) -> None:
        book = _make_book()
        module_bp = ModuleBlueprint(
            title="Nonexistent",
            source_chapter_number=99,
        )

        chapter = find_matching_chapter(book, module_bp)

        assert chapter is None

    def test_matches_by_title_embedded_chapter_number(self) -> None:
        """When extraction numbers chapters sequentially but the PDF title
        says 'CHAPTER 3', the LLM may use 3 as source_chapter_number.
        The matcher should find it via the title-embedded number.

        Here source_chapter_number=5 doesn't match any sequential number
        (chapters are 1-3), so it falls through to title-embedded matching
        and finds 'CHAPTER 5 Modeling Risk Factors'.
        """
        book = _make_book(
            chapters=(
                _make_chapter(number=1, title="CHAPTER 3 Fundamentals of Statistics"),
                _make_chapter(number=2, title="CHAPTER 4 Monte Carlo Methods"),
                _make_chapter(number=3, title="CHAPTER 5 Modeling Risk Factors"),
            )
        )
        # LLM wrote source_chapter_number=5 (from PDF title "CHAPTER 5")
        # No chapter has sequential number 5, so title-embedded matching kicks in
        module_bp = ModuleBlueprint(
            title="Risk Module",
            source_chapter_number=5,
        )

        chapter = find_matching_chapter(book, module_bp)

        assert chapter is not None
        assert chapter.chapter_number == 3
        assert "CHAPTER 5" in chapter.title

    def test_sequential_number_takes_priority_over_title(self) -> None:
        """Exact sequential match should win over title-embedded match."""
        book = _make_book(
            chapters=(
                _make_chapter(number=1, title="CHAPTER 5 Some Topic"),
                _make_chapter(number=2, title="CHAPTER 1 Other Topic"),
            )
        )
        module_bp = ModuleBlueprint(
            title="Anything",
            source_chapter_number=1,
        )

        chapter = find_matching_chapter(book, module_bp)

        # Should match chapter_number=1 (sequential), not the one titled "CHAPTER 1"
        assert chapter is not None
        assert chapter.chapter_number == 1
        assert "CHAPTER 5" in chapter.title

    def test_section_validation_overrides_sequential_when_wrong(self) -> None:
        """When sequential and title-embedded match different chapters,
        section-match count should pick the chapter whose sections actually
        match the blueprint.

        Simulates the Module 4 bug: LLM wrote source_chapter_number=2
        meaning PDF "CHAPTER 2", but sequential chapter 2 is actually
        "CHAPTER 3 Fundamentals of Statistics" (wrong content).
        """
        book = _make_book(
            chapters=(
                _make_chapter(
                    number=1,
                    title="CHAPTER 2 Fundamentals of Probability",
                    sections=(
                        _make_section("2.1 BASIC PROBABILITY"),
                        _make_section("2.2 MULTIVARIATE DISTRIBUTION FUNCTIONS"),
                    ),
                ),
                _make_chapter(
                    number=2,
                    title="CHAPTER 3 Fundamentals of Statistics",
                    sections=(
                        _make_section("3.1 PARAMETER ESTIMATION"),
                        _make_section("3.2 HYPOTHESIS TESTING"),
                    ),
                ),
            )
        )
        # LLM used PDF-embedded number 2, with sections from "CHAPTER 2"
        module_bp = ModuleBlueprint(
            title="Multivariate Distributions",
            source_chapter_number=2,
            sections=[
                SectionBlueprint(
                    title="Multivariate Distributions",
                    source_section_title="2.2 MULTIVARIATE DISTRIBUTION FUNCTIONS",
                ),
            ],
        )

        chapter = find_matching_chapter(book, module_bp)

        # Title-embedded match (sequential ch1, title "CHAPTER 2") wins
        # because the section "2.2 MULTIVARIATE DISTRIBUTION FUNCTIONS"
        # exists there, not in sequential ch2.
        assert chapter is not None
        assert chapter.chapter_number == 1
        assert "CHAPTER 2" in chapter.title

    def test_section_validation_defaults_to_sequential_on_tie(self) -> None:
        """When both chapters have equal section hits (including 0),
        sequential match wins."""
        book = _make_book(
            chapters=(
                _make_chapter(number=1, title="CHAPTER 5 Topic A"),
                _make_chapter(number=2, title="CHAPTER 1 Topic B"),
            )
        )
        # No sections in blueprint → 0-0 tie → sequential wins
        module_bp = ModuleBlueprint(
            title="Anything",
            source_chapter_number=1,
        )

        chapter = find_matching_chapter(book, module_bp)

        assert chapter is not None
        assert chapter.chapter_number == 1

    def test_section_validation_with_no_ambiguity(self) -> None:
        """When sequential and title-embedded match the same chapter,
        return it without needing section validation."""
        book = _make_book(
            chapters=(
                _make_chapter(number=3, title="CHAPTER 3 Risk Models"),
            )
        )
        module_bp = ModuleBlueprint(
            title="Risk",
            source_chapter_number=3,
        )

        chapter = find_matching_chapter(book, module_bp)

        assert chapter is not None
        assert chapter.chapter_number == 3


class TestFindMatchingSection:
    """Tests for the section matching helper."""

    def test_matches_by_source_section_title(self) -> None:
        chapter = _make_chapter(
            sections=(
                _make_section("Section A"),
                _make_section("Section B"),
            )
        )
        section_bp = SectionBlueprint(
            title="Renamed",
            source_section_title="Section B",
        )

        section = find_matching_section(chapter, section_bp)

        assert section is not None
        assert section.title == "Section B"

    def test_falls_back_to_blueprint_title(self) -> None:
        chapter = _make_chapter(
            sections=(_make_section("My Section"),)
        )
        section_bp = SectionBlueprint(
            title="My Section",
            source_section_title="",
        )

        section = find_matching_section(chapter, section_bp)

        assert section is not None
        assert section.title == "My Section"

    def test_matches_subsections(self) -> None:
        sub = Section(
            title="Nested Sub",
            level=3,
            start_page=3,
            end_page=5,
            text="B" * 200,
        )
        parent = Section(
            title="Parent",
            level=2,
            start_page=1,
            end_page=5,
            text="A" * 200,
            subsections=(sub,),
        )
        chapter = _make_chapter(sections=(parent,))
        section_bp = SectionBlueprint(
            title="Nested Sub",
            source_section_title="Nested Sub",
        )

        section = find_matching_section(chapter, section_bp)

        assert section is not None
        assert section.title == "Nested Sub"

    def test_returns_none_when_no_match(self) -> None:
        chapter = _make_chapter(sections=(_make_section("Existing"),))
        section_bp = SectionBlueprint(
            title="Nonexistent",
            source_section_title="Nonexistent",
        )

        section = find_matching_section(chapter, section_bp)

        assert section is None


class TestBuildRichContentSummary:
    """Tests for the analysis-enhanced content summary."""

    def test_includes_concept_inventory(self) -> None:
        from src.transformation.analysis_types import (
            ChapterAnalysis,
            ConceptEntry,
            SectionCharacterization,
        )

        book = _make_book()
        analyses = [
            ChapterAnalysis(
                chapter_number=1,
                chapter_title="Introduction to Testing",
                concepts=[
                    ConceptEntry(
                        name="Unit test",
                        definition="A test of an isolated code unit.",
                        concept_type="definition",
                        section_title="What is Testing",
                        importance="core",
                    ),
                ],
                section_characterizations=[
                    SectionCharacterization(
                        section_title="What is Testing",
                        dominant_content_type="conceptual",
                        has_definitions=True,
                        summary="Introduces testing concepts.",
                    ),
                ],
                logical_flow="Definitions first, then rationale.",
                external_prerequisites=["Programming basics"],
            ),
            ChapterAnalysis(
                chapter_number=2,
                chapter_title="Advanced Testing",
            ),
        ]

        summary = _build_rich_content_summary(book, analyses)

        assert "Unit test" in summary
        assert "definition" in summary
        assert "conceptual" in summary
        assert "Programming basics" in summary
        assert "Definitions first" in summary
        assert "Introduces testing concepts" in summary

    def test_includes_concept_graph(self) -> None:
        from src.transformation.analysis_types import (
            ChapterAnalysis,
            ConceptGraph,
            ResolvedConcept,
        )

        book = _make_book()
        analyses = [
            ChapterAnalysis(chapter_number=1, chapter_title="Ch1"),
            ChapterAnalysis(chapter_number=2, chapter_title="Ch2"),
        ]
        graph = ConceptGraph(
            topological_order=["Foundation", "Intermediate", "Advanced"],
            foundation_concepts=["Foundation"],
            advanced_concepts=["Advanced"],
        )

        summary = _build_rich_content_summary(book, analyses, graph)

        assert "Foundation → Intermediate → Advanced" in summary
        assert "Foundation" in summary
        assert "start here" in summary.lower() or "foundation" in summary.lower()

    def test_backward_compatible_without_analysis(self) -> None:
        """plan_curriculum still works without analysis args."""
        book = _make_book()
        client = MockPlannerClient()

        blueprint = plan_curriculum(book, client)

        assert blueprint.course_title == "Mock Course"
        assert client.call_count == 1

    def test_plan_curriculum_with_analysis(self) -> None:
        """plan_curriculum accepts analysis args and uses rich summary."""
        from src.transformation.analysis_types import ChapterAnalysis, ConceptGraph

        book = _make_book()
        client = MockPlannerClient()
        analyses = [
            ChapterAnalysis(chapter_number=1, chapter_title="Ch1"),
            ChapterAnalysis(chapter_number=2, chapter_title="Ch2"),
        ]
        graph = ConceptGraph(
            topological_order=["A", "B"],
            foundation_concepts=["A"],
        )

        blueprint = plan_curriculum(book, client, analyses, graph)

        assert blueprint.course_title == "Mock Course"
        assert client.call_count == 1

    def test_plan_curriculum_with_document_type(self) -> None:
        """plan_curriculum accepts document_type and enriches the prompt."""
        book = _make_book()

        # Capture the system prompt to verify document profile injection
        class CapturingClient(MockPlannerClient):
            def __init__(self) -> None:
                super().__init__()
                self.last_system_prompt = ""

            def complete_structured(
                self, system_prompt: str, user_prompt: str, response_model: type[T]
            ) -> T:
                self.last_system_prompt = system_prompt
                return super().complete_structured(system_prompt, user_prompt, response_model)

        client = CapturingClient()
        blueprint = plan_curriculum(book, client, document_type="quantitative")

        assert blueprint.course_title == "Mock Course"
        assert "quantitative" in client.last_system_prompt
        assert "Document Profile" in client.last_system_prompt
        assert "worked_example" in client.last_system_prompt

    def test_plan_curriculum_mixed_type_no_profile_section(self) -> None:
        """Document type 'mixed' should NOT inject a document profile section."""
        book = _make_book()

        class CapturingClient(MockPlannerClient):
            def __init__(self) -> None:
                super().__init__()
                self.last_system_prompt = ""

            def complete_structured(
                self, system_prompt: str, user_prompt: str, response_model: type[T]
            ) -> T:
                self.last_system_prompt = system_prompt
                return super().complete_structured(system_prompt, user_prompt, response_model)

        client = CapturingClient()
        blueprint = plan_curriculum(book, client, document_type="mixed")

        assert "Document Profile" not in client.last_system_prompt


# ── _extract_chapter_number_from_title() ─────────────────────────────────────


class TestExtractChapterNumberFromTitle:
    """Tests for the title-embedded chapter number extractor."""

    def test_standard_format(self) -> None:
        assert _extract_chapter_number_from_title("CHAPTER 3 Fundamentals of Probability") == 3

    def test_lowercase(self) -> None:
        assert _extract_chapter_number_from_title("chapter 12 advanced topics") == 12

    def test_abbreviated_ch(self) -> None:
        assert _extract_chapter_number_from_title("Ch. 5 Risk Management") == 5

    def test_ch_no_dot(self) -> None:
        assert _extract_chapter_number_from_title("Ch 7 Portfolio Theory") == 7

    def test_no_chapter_number_returns_none(self) -> None:
        assert _extract_chapter_number_from_title("Introduction to Finance") is None

    def test_number_without_chapter_prefix_returns_none(self) -> None:
        assert _extract_chapter_number_from_title("3 Fundamentals") is None

    def test_leading_zero(self) -> None:
        assert _extract_chapter_number_from_title("Chapter 02 Basics") == 2


# ── Tests for concept-level section splitting ────────────────────────────────


def _make_concept(
    name: str,
    section_title: str = "Test Section",
    importance: str = "core",
):
    from src.transformation.analysis_types import ConceptEntry

    return ConceptEntry(
        name=name,
        definition=f"Definition of {name}",
        concept_type="definition",
        section_title=section_title,
        importance=importance,
    )


class TestSplitOverloadedSections:
    """Tests for the _split_overloaded_sections safety net."""

    def test_no_split_when_few_concepts(self) -> None:
        """Sections with <4 core/supporting concepts are not split."""
        from src.transformation.analysis_types import ChapterAnalysis

        blueprint = CurriculumBlueprint(
            course_title="Test",
            modules=[
                ModuleBlueprint(
                    title="M1",
                    source_chapter_number=1,
                    sections=[
                        SectionBlueprint(
                            title="Sec",
                            source_section_title="Test Section",
                        ),
                    ],
                ),
            ],
        )
        analyses = [
            ChapterAnalysis(
                chapter_number=1,
                chapter_title="Ch1",
                concepts=[
                    _make_concept("A", "Test Section"),
                    _make_concept("B", "Test Section"),
                ],
            ),
        ]

        result = _split_overloaded_sections(blueprint, analyses)

        assert len(result.modules[0].sections) == 1
        assert result.modules[0].sections[0].title == "Sec"

    def test_splits_section_with_4_plus_concepts(self) -> None:
        """Section with 4 core concepts becomes 2 sub-sections."""
        from src.transformation.analysis_types import ChapterAnalysis

        blueprint = CurriculumBlueprint(
            course_title="Test",
            modules=[
                ModuleBlueprint(
                    title="M1",
                    source_chapter_number=1,
                    sections=[
                        SectionBlueprint(
                            title="Dense Section",
                            source_section_title="Dense Section",
                            template="worked_example",
                            bloom_target="understand",
                        ),
                    ],
                ),
            ],
        )
        analyses = [
            ChapterAnalysis(
                chapter_number=1,
                chapter_title="Ch1",
                concepts=[
                    _make_concept("Alpha", "Dense Section"),
                    _make_concept("Beta", "Dense Section"),
                    _make_concept("Gamma", "Dense Section"),
                    _make_concept("Delta", "Dense Section"),
                ],
            ),
        ]

        result = _split_overloaded_sections(blueprint, analyses)

        sections = result.modules[0].sections
        assert len(sections) == 2
        # Each sub-section has focus_concepts
        assert len(sections[0].focus_concepts) == 2
        assert len(sections[1].focus_concepts) == 2
        # Both share the same source_section_title
        assert sections[0].source_section_title == "Dense Section"
        assert sections[1].source_section_title == "Dense Section"

    def test_preserves_sections_with_focus_concepts(self) -> None:
        """Sections already specifying focus_concepts are not re-split."""
        from src.transformation.analysis_types import ChapterAnalysis

        blueprint = CurriculumBlueprint(
            course_title="Test",
            modules=[
                ModuleBlueprint(
                    title="M1",
                    source_chapter_number=1,
                    sections=[
                        SectionBlueprint(
                            title="Already Split",
                            source_section_title="Dense Section",
                            focus_concepts=["A", "B"],
                        ),
                    ],
                ),
            ],
        )
        analyses = [
            ChapterAnalysis(
                chapter_number=1,
                chapter_title="Ch1",
                concepts=[
                    _make_concept(name, "Dense Section")
                    for name in ["A", "B", "C", "D", "E"]
                ],
            ),
        ]

        result = _split_overloaded_sections(blueprint, analyses)

        # Should NOT re-split since focus_concepts already set
        assert len(result.modules[0].sections) == 1

    def test_ignores_peripheral_concepts(self) -> None:
        """Only core/supporting concepts count toward the split threshold."""
        from src.transformation.analysis_types import ChapterAnalysis

        blueprint = CurriculumBlueprint(
            course_title="Test",
            modules=[
                ModuleBlueprint(
                    title="M1",
                    source_chapter_number=1,
                    sections=[
                        SectionBlueprint(
                            title="Sec",
                            source_section_title="Test Section",
                        ),
                    ],
                ),
            ],
        )
        analyses = [
            ChapterAnalysis(
                chapter_number=1,
                chapter_title="Ch1",
                concepts=[
                    _make_concept("A", "Test Section", importance="core"),
                    _make_concept("B", "Test Section", importance="supporting"),
                    _make_concept("C", "Test Section", importance="peripheral"),
                    _make_concept("D", "Test Section", importance="peripheral"),
                    _make_concept("E", "Test Section", importance="peripheral"),
                ],
            ),
        ]

        result = _split_overloaded_sections(blueprint, analyses)

        # Only 2 non-peripheral concepts — below threshold, no split
        assert len(result.modules[0].sections) == 1

    def test_no_split_without_analyses(self) -> None:
        """Returns blueprint unchanged when analyses is None."""
        blueprint = CurriculumBlueprint(
            course_title="Test",
            modules=[
                ModuleBlueprint(
                    title="M1",
                    source_chapter_number=1,
                    sections=[
                        SectionBlueprint(title="Sec", source_section_title="Sec"),
                    ],
                ),
            ],
        )

        result = _split_overloaded_sections(blueprint, None)

        assert result is blueprint  # exact same object

    def test_uses_topological_order(self) -> None:
        """When concept graph available, concepts are ordered by dependency."""
        from src.transformation.analysis_types import (
            ChapterAnalysis,
            ConceptGraph,
        )

        blueprint = CurriculumBlueprint(
            course_title="Test",
            modules=[
                ModuleBlueprint(
                    title="M1",
                    source_chapter_number=1,
                    sections=[
                        SectionBlueprint(
                            title="Sec",
                            source_section_title="Test Section",
                        ),
                    ],
                ),
            ],
        )
        analyses = [
            ChapterAnalysis(
                chapter_number=1,
                chapter_title="Ch1",
                concepts=[
                    _make_concept("Delta", "Test Section"),
                    _make_concept("Alpha", "Test Section"),
                    _make_concept("Beta", "Test Section"),
                    _make_concept("Gamma", "Test Section"),
                ],
            ),
        ]
        graph = ConceptGraph(
            topological_order=["Alpha", "Beta", "Gamma", "Delta"],
            foundation_concepts=["Alpha"],
            advanced_concepts=["Delta"],
        )

        result = _split_overloaded_sections(blueprint, analyses, graph)

        sections = result.modules[0].sections
        assert len(sections) == 2
        # First unit should have the foundation concepts (Alpha, Beta)
        assert "Alpha" in sections[0].focus_concepts
        assert "Beta" in sections[0].focus_concepts


class TestValidateProgressionWithSplits:
    """Tests for validate_progression with concept sub-units."""

    def test_allows_bloom_reset_across_source_sections(self) -> None:
        """Bloom drop is allowed when source_section_title changes."""
        blueprint = CurriculumBlueprint(
            course_title="Test",
            modules=[
                ModuleBlueprint(
                    title="M1",
                    source_chapter_number=1,
                    sections=[
                        SectionBlueprint(
                            title="A1",
                            source_section_title="Section A",
                            bloom_target="apply",
                        ),
                        SectionBlueprint(
                            title="B1",
                            source_section_title="Section B",
                            bloom_target="understand",
                        ),
                    ],
                ),
            ],
        )

        warnings = validate_progression(blueprint)

        assert len(warnings) == 0

    def test_flags_bloom_drop_within_same_source(self) -> None:
        """Bloom drop within same source_section_title is still flagged."""
        blueprint = CurriculumBlueprint(
            course_title="Test",
            modules=[
                ModuleBlueprint(
                    title="M1",
                    source_chapter_number=1,
                    sections=[
                        SectionBlueprint(
                            title="A1",
                            source_section_title="Section A",
                            bloom_target="analyze",
                        ),
                        SectionBlueprint(
                            title="A2",
                            source_section_title="Section A",
                            bloom_target="remember",
                        ),
                    ],
                ),
            ],
        )

        warnings = validate_progression(blueprint)

        assert len(warnings) == 1
        assert "A2" in warnings[0]

    def test_no_warning_for_small_dip_within_source(self) -> None:
        """A 1-level dip within the same source section is allowed."""
        blueprint = CurriculumBlueprint(
            course_title="Test",
            modules=[
                ModuleBlueprint(
                    title="M1",
                    source_chapter_number=1,
                    sections=[
                        SectionBlueprint(
                            title="A1",
                            source_section_title="Section A",
                            bloom_target="apply",
                        ),
                        SectionBlueprint(
                            title="A2",
                            source_section_title="Section A",
                            bloom_target="understand",
                        ),
                    ],
                ),
            ],
        )

        warnings = validate_progression(blueprint)

        assert len(warnings) == 0
