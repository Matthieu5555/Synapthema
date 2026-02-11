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
    find_matching_chapter,
    find_matching_section,
    plan_curriculum,
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
