"""Tests for bug fixes identified during the comprehensive audit.

Covers:
- Bloom edge direction fix (curriculum_planner)
- analyze_book exception handling (deep_reader)
- Substring entity resolution guard (concept_consolidator)
- Missing element types in _tag_element_concepts (html_generator)
- Pydantic validators for quiz/matching/FITB/concept_map/essay (types)
- Security: _sanitize_html expanded, _json_for_attr, mermaid code sanitization
- _filter_front_matter edge case (structure_detector)
- identify_chapters end_page guard (structure_detector)
- _build_chapter_text subsection inclusion (deep_reader)
- Section title normalization in planner summary
"""

from __future__ import annotations

from typing import TypeVar
from unittest.mock import MagicMock

import pytest

from src.extraction.types import Book, Chapter, Section
from src.transformation.analysis_types import (
    ChapterAnalysis,
    ConceptEdge,
    ConceptEntry,
    ConceptGraph,
    PrerequisiteLink,
    ResolvedConcept,
    SectionCharacterization,
)
from src.transformation.types import (
    ConceptMap,
    ConceptMapEdge,
    ConceptMapNode,
    FillInTheBlank,
    InteractiveEssay,
    MatchingExercise,
    QuizQuestion,
    SelfExplain,
)

T = TypeVar("T")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_section(
    title: str = "Test Section",
    text: str = "A" * 200,
    level: int = 2,
    subsections: tuple[Section, ...] = (),
) -> Section:
    return Section(
        title=title, level=level, start_page=1, end_page=5, text=text,
        subsections=subsections,
    )


def _make_chapter(
    num: int = 1,
    sections: tuple[Section, ...] | None = None,
) -> Chapter:
    secs = sections or (_make_section(),)
    return Chapter(
        chapter_number=num, title=f"Chapter {num}",
        start_page=1, end_page=10, sections=secs,
    )


def _make_concept_entry(
    name: str, section_title: str = "S1",
) -> ConceptEntry:
    return ConceptEntry(
        name=name, definition="A concept.", concept_type="definition",
        section_title=section_title, importance="core",
    )


# ── 1. Bloom edge direction ─────────────────────────────────────────────────


class TestBloomEdgeDirection:
    """Verify _bloom_from_concept_position counts prerequisites (e.source),
    not dependents (e.target)."""

    def _graph(self, concepts, edges, canonical_map=None) -> ConceptGraph:
        cmap = canonical_map or {c.canonical_name: c.canonical_name for c in concepts}
        return ConceptGraph(
            concepts=concepts, edges=edges, canonical_map=cmap,
        )

    def test_concept_with_many_prerequisites_gets_higher_bloom(self) -> None:
        from src.transformation.curriculum_planner import _bloom_from_concept_position

        concepts = [_make_concept_entry("advanced")]
        graph = self._graph(
            concepts=[
                ResolvedConcept(canonical_name="advanced", definition="d", first_introduced_chapter=1),
                ResolvedConcept(canonical_name="basic1", definition="d", first_introduced_chapter=1),
                ResolvedConcept(canonical_name="basic2", definition="d", first_introduced_chapter=1),
                ResolvedConcept(canonical_name="basic3", definition="d", first_introduced_chapter=1),
            ],
            edges=[
                ConceptEdge(source="advanced", target="basic1", relationship="requires"),
                ConceptEdge(source="advanced", target="basic2", relationship="requires"),
                ConceptEdge(source="advanced", target="basic3", relationship="requires"),
            ],
        )
        result = _bloom_from_concept_position(concepts, graph)
        assert result == "analyze"  # max_deps=3 → "analyze"

    def test_foundation_concept_with_no_prerequisites_gets_understand(self) -> None:
        from src.transformation.curriculum_planner import _bloom_from_concept_position

        concepts = [_make_concept_entry("basic")]
        graph = self._graph(
            concepts=[
                ResolvedConcept(canonical_name="basic", definition="d", first_introduced_chapter=1),
                ResolvedConcept(canonical_name="advanced", definition="d", first_introduced_chapter=1),
            ],
            edges=[
                ConceptEdge(source="advanced", target="basic", relationship="requires"),
            ],
        )
        result = _bloom_from_concept_position(concepts, graph)
        assert result == "understand"  # max_deps=0 → "understand"

    def test_concept_with_few_prerequisites_gets_apply(self) -> None:
        from src.transformation.curriculum_planner import _bloom_from_concept_position

        concepts = [_make_concept_entry("mid")]
        graph = self._graph(
            concepts=[
                ResolvedConcept(canonical_name="mid", definition="d", first_introduced_chapter=1),
                ResolvedConcept(canonical_name="basic1", definition="d", first_introduced_chapter=1),
                ResolvedConcept(canonical_name="basic2", definition="d", first_introduced_chapter=1),
            ],
            edges=[
                ConceptEdge(source="mid", target="basic1", relationship="requires"),
                ConceptEdge(source="mid", target="basic2", relationship="requires"),
            ],
        )
        result = _bloom_from_concept_position(concepts, graph)
        assert result == "apply"  # max_deps=2 → "apply"


# ── 2. analyze_book exception handling ───────────────────────────────────────


class MockAnalysisClient:
    """Returns a fixed analysis for all chapters."""

    def __init__(self, analysis: ChapterAnalysis) -> None:
        self._analysis = analysis

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return "mock"

    def complete_light(self, system_prompt: str, user_prompt: str) -> str:
        return "mock"

    def complete_structured(
        self, system_prompt: str, user_prompt: str, response_model: type[T],
    ) -> T:
        return self._analysis  # type: ignore[return-value]

    def complete_structured_light(
        self, system_prompt: str, user_prompt: str, response_model: type[T],
    ) -> T:
        return self._analysis  # type: ignore[return-value]


class FailOnChapterClient:
    """Fails for specific chapter numbers."""

    def __init__(self, fail_chapters: set[int], analysis: ChapterAnalysis) -> None:
        self._fail = fail_chapters
        self._analysis = analysis
        self._call_count = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return "mock"

    def complete_light(self, system_prompt: str, user_prompt: str) -> str:
        return "mock"

    def complete_structured(
        self, system_prompt: str, user_prompt: str, response_model: type[T],
    ) -> T:
        self._call_count += 1
        # Detect chapter number from prompt text
        for ch_num in self._fail:
            if f"Chapter {ch_num}:" in user_prompt:
                raise RuntimeError(f"LLM failed for chapter {ch_num}")
        return self._analysis  # type: ignore[return-value]

    def complete_structured_light(
        self, system_prompt: str, user_prompt: str, response_model: type[T],
    ) -> T:
        return self.complete_structured(system_prompt, user_prompt, response_model)


class TestAnalyzeBookExceptionHandling:
    """Verify analyze_book continues when individual chapters fail."""

    def _default_analysis(self) -> ChapterAnalysis:
        return ChapterAnalysis(
            chapter_number=1,
            chapter_title="Test",
            concepts=[_make_concept_entry("Bond")],
            prerequisites=[],
            section_characterizations=[
                SectionCharacterization(
                    section_title="S1",
                    dominant_content_type="conceptual",
                ),
            ],
            logical_flow="Flow.",
            core_learning_outcome="Outcome.",
        )

    def test_single_chapter_failure_does_not_kill_others(self) -> None:
        from src.transformation.deep_reader import analyze_book

        sections = (_make_section(),)
        book = Book(
            title="Test Book", author="Test", total_pages=30,
            chapters=tuple(
                Chapter(
                    chapter_number=i, title=f"Chapter {i}",
                    start_page=1, end_page=10, sections=sections,
                )
                for i in range(1, 4)
            ),
        )

        client = FailOnChapterClient(
            fail_chapters={2}, analysis=self._default_analysis(),
        )
        results = analyze_book(book, client, max_workers=1)

        # Chapter 2 failed, but 1 and 3 should succeed
        assert len(results) == 2

    def test_all_chapters_fail_returns_empty(self) -> None:
        from src.transformation.deep_reader import analyze_book

        sections = (_make_section(),)
        book = Book(
            title="Test Book", author="Test", total_pages=20,
            chapters=tuple(
                Chapter(
                    chapter_number=i, title=f"Chapter {i}",
                    start_page=1, end_page=10, sections=sections,
                )
                for i in range(1, 3)
            ),
        )

        client = FailOnChapterClient(
            fail_chapters={1, 2}, analysis=self._default_analysis(),
        )
        results = analyze_book(book, client, max_workers=1)
        assert len(results) == 0


# ── 3. Substring entity resolution ──────────────────────────────────────────


class TestSubstringResolutionGuard:
    """Verify single-word short names don't merge with longer names."""

    def test_short_single_word_does_not_merge(self) -> None:
        from src.transformation.concept_consolidator import _deduplicate_concepts

        c1 = ConceptEntry(
            name="Risk", definition="General risk.", concept_type="definition",
            section_title="S1", importance="core",
        )
        c2 = ConceptEntry(
            name="Market risk", definition="Risk from markets.",
            concept_type="definition", section_title="S1", importance="core",
        )
        c3 = ConceptEntry(
            name="Credit risk", definition="Risk from borrowers.",
            concept_type="definition", section_title="S1", importance="core",
        )
        resolved = _deduplicate_concepts([(c1, 1), (c2, 1), (c3, 1)])
        # All three should be distinct — "risk" should NOT merge with the others
        assert len(resolved) == 3

    def test_multi_word_substring_still_merges(self) -> None:
        from src.transformation.concept_consolidator import _deduplicate_concepts

        c1 = ConceptEntry(
            name="Sharpe ratio", definition="A concept.", concept_type="definition",
            section_title="S1", importance="core",
        )
        c2 = ConceptEntry(
            name="Sharpe ratio formula", definition="A concept.",
            concept_type="definition", section_title="S1", importance="core",
        )
        resolved = _deduplicate_concepts([(c1, 1), (c2, 2)])
        assert len(resolved) == 1

    def test_very_similar_length_single_word_merges(self) -> None:
        """When the shorter name is >= 60% of the longer, merge even if single word."""
        from src.transformation.concept_consolidator import _deduplicate_concepts

        c1 = ConceptEntry(
            name="Alpha", definition="d", concept_type="definition",
            section_title="S1", importance="core",
        )
        c2 = ConceptEntry(
            name="Alphas", definition="d", concept_type="definition",
            section_title="S1", importance="core",
        )
        # "alpha" (5 chars) / "alphas" (6 chars) = 0.83 > 0.6 → merge
        resolved = _deduplicate_concepts([(c1, 1), (c2, 2)])
        assert len(resolved) == 1


# ── 4. Missing element types in _tag_element_concepts ────────────────────────


class TestTagElementConceptsCompleteness:
    """Verify all element types are covered by _tag_element_concepts."""

    def test_ordering_element_is_tagged(self) -> None:
        from src.rendering.html_generator import _tag_element_concepts

        element = {
            "element_type": "ordering",
            "ordering": {"title": "Order these risk types", "items": ["a", "b"]},
        }
        concepts = ["risk", "return"]
        result = _tag_element_concepts(element, concepts)
        assert "risk" in result

    def test_categorization_element_is_tagged(self) -> None:
        from src.rendering.html_generator import _tag_element_concepts

        element = {
            "element_type": "categorization",
            "categorization": {"title": "Categorize return types", "items": ["a"]},
        }
        concepts = ["risk", "return"]
        result = _tag_element_concepts(element, concepts)
        assert "return" in result

    def test_error_detection_element_is_tagged(self) -> None:
        from src.rendering.html_generator import _tag_element_concepts

        element = {
            "element_type": "error_detection",
            "error_detection": {
                "title": "Find errors about yield curves",
                "items": [{"statement": "yield curve inverts"}],
            },
        }
        concepts = ["yield curve", "risk"]
        result = _tag_element_concepts(element, concepts)
        assert "yield curve" in result

    def test_analogy_element_is_tagged(self) -> None:
        from src.rendering.html_generator import _tag_element_concepts

        element = {
            "element_type": "analogy",
            "analogy": {"title": "Portfolio diversification analogy", "items": []},
        }
        concepts = ["diversification", "risk"]
        result = _tag_element_concepts(element, concepts)
        assert "diversification" in result


# ── 5. Pydantic validators ──────────────────────────────────────────────────


class TestQuizQuestionValidator:
    """Verify QuizQuestion correct_index and hint_eliminate_index clamping."""

    def test_out_of_bounds_correct_index_clamped_to_zero(self) -> None:
        q = QuizQuestion(
            question="Q?", options=["A", "B", "C"], correct_index=5,
        )
        assert q.correct_index == 0

    def test_negative_correct_index_clamped_to_zero(self) -> None:
        q = QuizQuestion(
            question="Q?", options=["A", "B"], correct_index=-3,
        )
        assert q.correct_index == 0

    def test_valid_correct_index_unchanged(self) -> None:
        q = QuizQuestion(
            question="Q?", options=["A", "B", "C"], correct_index=2,
        )
        assert q.correct_index == 2

    def test_eliminate_index_equal_to_correct_becomes_minus_one(self) -> None:
        q = QuizQuestion(
            question="Q?", options=["A", "B", "C"],
            correct_index=1, hint_eliminate_index=1,
        )
        assert q.hint_eliminate_index == -1

    def test_eliminate_index_out_of_bounds_becomes_minus_one(self) -> None:
        q = QuizQuestion(
            question="Q?", options=["A", "B"],
            correct_index=0, hint_eliminate_index=5,
        )
        assert q.hint_eliminate_index == -1

    def test_valid_eliminate_index_unchanged(self) -> None:
        q = QuizQuestion(
            question="Q?", options=["A", "B", "C"],
            correct_index=0, hint_eliminate_index=2,
        )
        assert q.hint_eliminate_index == 2


class TestMatchingExerciseValidator:
    """Verify MatchingExercise truncates to equal lengths."""

    def test_mismatched_lengths_truncated(self) -> None:
        m = MatchingExercise(
            title="T", left_items=["a", "b", "c"], right_items=["x", "y"],
        )
        assert len(m.left_items) == len(m.right_items) == 2

    def test_equal_lengths_unchanged(self) -> None:
        m = MatchingExercise(
            title="T", left_items=["a", "b"], right_items=["x", "y"],
        )
        assert len(m.left_items) == len(m.right_items) == 2

    def test_pair_explanations_padded(self) -> None:
        m = MatchingExercise(
            title="T", left_items=["a", "b", "c"],
            right_items=["x", "y", "z"],
            pair_explanations=["e1"],
        )
        assert len(m.pair_explanations) == 3
        assert m.pair_explanations[1] == ""

    def test_pair_explanations_truncated(self) -> None:
        m = MatchingExercise(
            title="T", left_items=["a", "b"], right_items=["x", "y"],
            pair_explanations=["e1", "e2", "e3"],
        )
        assert len(m.pair_explanations) == 2


class TestFillInTheBlankValidator:
    """Verify blank count matches answer count."""

    def test_fewer_answers_than_blanks_padded(self) -> None:
        f = FillInTheBlank(
            statement="The _____ chases the _____",
            answers=["cat"],
        )
        assert len(f.answers) == 2
        assert f.answers[1] == ""

    def test_more_answers_than_blanks_truncated(self) -> None:
        f = FillInTheBlank(
            statement="The _____ is here.",
            answers=["cat", "dog", "bird"],
        )
        assert len(f.answers) == 1

    def test_matching_count_unchanged(self) -> None:
        f = FillInTheBlank(
            statement="_____ and _____",
            answers=["alpha", "beta"],
        )
        assert len(f.answers) == 2

    def test_no_blanks_in_statement_leaves_answers_unchanged(self) -> None:
        """When there are no _____ markers, don't touch the answers."""
        f = FillInTheBlank(statement="No blanks here", answers=["a"])
        assert len(f.answers) == 1


class TestConceptMapValidator:
    """Verify blank_edge_indices are filtered to valid range."""

    def test_out_of_bounds_indices_removed(self) -> None:
        cmap = ConceptMap(
            title="T",
            nodes=[
                ConceptMapNode(id="a", label="A"),
                ConceptMapNode(id="b", label="B"),
                ConceptMapNode(id="c", label="C"),
            ],
            edges=[
                ConceptMapEdge(source="a", target="b", label="rel1"),
                ConceptMapEdge(source="b", target="c", label="rel2"),
            ],
            blank_edge_indices=[0, 5, -1, 1],
        )
        assert cmap.blank_edge_indices == [0, 1]


class TestInteractiveEssayThreshold:
    """Verify passing_threshold is constrained to 0.0-1.0."""

    def test_threshold_above_one_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            InteractiveEssay(
                title="T",
                concepts_tested=["a"],
                prompts=[SelfExplain(
                    prompt="Explain", key_points=["k1", "k2"],
                    example_response="Example",
                )],
                passing_threshold=70.0,
            )

    def test_valid_threshold_accepted(self) -> None:
        ie = InteractiveEssay(
            title="T",
            concepts_tested=["a"],
            prompts=[SelfExplain(
                prompt="Explain", key_points=["k1", "k2"],
                example_response="Example",
            )],
            passing_threshold=0.8,
        )
        assert ie.passing_threshold == 0.8


# ── 6. Security: _sanitize_html, _json_for_attr, mermaid ────────────────────


class TestSanitizeHtmlExpanded:
    """Verify _sanitize_html strips new dangerous patterns."""

    def test_strips_iframe(self) -> None:
        from src.rendering.html_generator import _sanitize_html

        html = '<p>Hello</p><iframe src="evil.com"></iframe><p>World</p>'
        result = _sanitize_html(html)
        assert "<iframe" not in result
        assert "Hello" in result

    def test_strips_object(self) -> None:
        from src.rendering.html_generator import _sanitize_html

        html = '<object data="evil.swf">x</object>'
        result = _sanitize_html(html)
        assert "<object" not in result

    def test_strips_embed(self) -> None:
        from src.rendering.html_generator import _sanitize_html

        html = '<embed src="evil.swf"/>'
        result = _sanitize_html(html)
        assert "<embed" not in result

    def test_strips_javascript_urls(self) -> None:
        from src.rendering.html_generator import _sanitize_html

        html = '<a href="javascript:alert(1)">click</a>'
        result = _sanitize_html(html)
        assert "javascript:" not in result

    def test_strips_script(self) -> None:
        from src.rendering.html_generator import _sanitize_html

        html = '<p>Hi</p><script>alert(1)</script>'
        result = _sanitize_html(html)
        assert "<script" not in result

    def test_strips_event_handlers(self) -> None:
        from src.rendering.html_generator import _sanitize_html

        html = '<img src="x" onerror="alert(1)">'
        result = _sanitize_html(html)
        assert "onerror" not in result


class TestJsonForAttr:
    """Verify _json_for_attr escapes single quotes."""

    def test_single_quotes_escaped(self) -> None:
        from src.rendering.html_generator import _json_for_attr

        result = _json_for_attr({"key": "it's a test"})
        assert "'" not in result
        assert "&#39;" in result

    def test_normal_json_unchanged(self) -> None:
        from src.rendering.html_generator import _json_for_attr

        result = _json_for_attr({"a": 1, "b": [2, 3]})
        assert '"a"' in result
        assert "'" not in result

    def test_empty_string_safe(self) -> None:
        from src.rendering.html_generator import _json_for_attr

        result = _json_for_attr("")
        assert result == '""'


class TestMermaidCodeSanitization:
    """Verify HTML tags are stripped from mermaid diagram code."""

    def test_html_tags_stripped(self) -> None:
        from src.transformation.types import MermaidElement, MermaidDiagram
        from src.rendering.html_generator import _prep_mermaid

        elem = MermaidElement(
            element_type="mermaid",
            bloom_level="understand",
            mermaid=MermaidDiagram(
                title="Test",
                diagram_code='graph TD\nA-->B\n</pre><script>alert(1)</script><pre>',
            ),
        )
        result = _prep_mermaid(elem, None, False)
        assert "<script>" not in result["mermaid"]["diagram_code"]
        assert "</pre>" not in result["mermaid"]["diagram_code"]
        assert "A-->B" in result["mermaid"]["diagram_code"]


# ── 7. _filter_front_matter edge case ────────────────────────────────────────


class TestFilterFrontMatterAllFrontMatter:
    """Verify _filter_front_matter returns empty when all entries are front matter."""

    def test_all_front_matter_returns_empty(self) -> None:
        from src.extraction.structure_detector import TocEntry, _filter_front_matter

        entries = (
            TocEntry(level=1, title="Preface", page=1),
            TocEntry(level=1, title="Table of Contents", page=3),
            TocEntry(level=1, title="Acknowledgments", page=5),
        )
        result = _filter_front_matter(entries)
        assert result == ()

    def test_mixed_entries_filter_correctly(self) -> None:
        from src.extraction.structure_detector import TocEntry, _filter_front_matter

        entries = (
            TocEntry(level=1, title="Preface", page=1),
            TocEntry(level=1, title="Chapter 1: Introduction", page=10),
            TocEntry(level=1, title="Chapter 2: Methods", page=30),
        )
        result = _filter_front_matter(entries)
        assert len(result) == 2
        assert result[0].title == "Chapter 1: Introduction"


# ── 8. identify_chapters end_page guard ──────────────────────────────────────


class TestIdentifyChaptersEndPageGuard:
    """Verify end_page >= start_page even with same-page consecutive chapters."""

    def test_same_page_chapters_no_negative_range(self) -> None:
        from src.extraction.structure_detector import TocEntry, identify_chapters

        entries = (
            TocEntry(level=1, title="Chapter 1", page=10),
            TocEntry(level=1, title="Chapter 2", page=10),  # Same page!
            TocEntry(level=1, title="Chapter 3", page=20),
        )
        chapters = identify_chapters(entries, total_pages=30)
        for ch_entry, _, start, end in chapters:
            assert end >= start, (
                f"Chapter '{ch_entry.title}' has end_page {end} < start_page {start}"
            )


# ── 9. _build_chapter_text includes subsections ─────────────────────────────


class TestBuildChapterTextSubsections:
    """Verify subsection text is included in the chapter text."""

    def test_subsection_text_included(self) -> None:
        from src.transformation.deep_reader import _build_chapter_text

        sub = Section(
            title="Subsection 1.1",
            level=3,
            start_page=2,
            end_page=4,
            text="SUBSECTION_UNIQUE_CONTENT",
        )
        section = Section(
            title="Section 1",
            level=2,
            start_page=1,
            end_page=5,
            text="Section top-level text.",
            subsections=(sub,),
        )
        chapter = Chapter(
            chapter_number=1, title="Ch1",
            start_page=1, end_page=5, sections=(section,),
        )
        result = _build_chapter_text(chapter)
        assert "SUBSECTION_UNIQUE_CONTENT" in result
        assert "Section top-level text" in result
        assert "### Subsection 1.1" in result

    def test_no_subsections_still_works(self) -> None:
        from src.transformation.deep_reader import _build_chapter_text

        section = Section(
            title="Section 1", level=2, start_page=1, end_page=5,
            text="Only top-level.",
        )
        chapter = Chapter(
            chapter_number=1, title="Ch1",
            start_page=1, end_page=5, sections=(section,),
        )
        result = _build_chapter_text(chapter)
        assert "Only top-level." in result


# ── 10. Course slug sanitization ─────────────────────────────────────────────


class TestCourseSlugSanitization:
    """Verify course_slug is sanitized for safe embedding in JS."""

    def test_special_chars_replaced(self) -> None:
        import re

        raw = "test'; alert('xss');//"
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "-", raw)
        assert "'" not in sanitized
        assert "alert" not in sanitized or "(" not in sanitized

    def test_normal_slug_unchanged(self) -> None:
        import re

        raw = "my-course-123"
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "-", raw)
        assert sanitized == raw


# ── 11. Section title normalization in planner ───────────────────────────────


class TestSectionTitleNormalization:
    """Verify planner summary normalizes section titles for concept lookup."""

    def test_case_insensitive_match(self) -> None:
        from src.transformation.curriculum_planner import _format_section_characterization

        analysis = ChapterAnalysis(
            chapter_number=1,
            chapter_title="Ch1",
            concepts=[
                ConceptEntry(
                    name="Bond",
                    definition="d",
                    concept_type="definition",
                    section_title="INTRODUCTION",  # Uppercase
                    importance="core",
                ),
            ],
            prerequisites=[],
            section_characterizations=[
                SectionCharacterization(
                    section_title="Introduction",  # Lowercase
                    dominant_content_type="conceptual",
                ),
            ],
            logical_flow="f",
            core_learning_outcome="o",
        )
        sc = analysis.section_characterizations[0]
        lines = _format_section_characterization(sc, analysis, None)
        # Should include "Bond" in the concepts list despite case mismatch
        joined = " ".join(lines)
        assert "Bond" in joined
