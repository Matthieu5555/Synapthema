"""Tests for the deep reader — Stage 1.25 explorer agent."""

from __future__ import annotations

from typing import TypeVar

import pytest

from src.extraction.types import Chapter, Section
from src.transformation.analysis_types import (
    ChapterAnalysis,
    ConceptEntry,
    PrerequisiteLink,
    SectionCharacterization,
)
from src.transformation.deep_reader import (
    analyze_book,
    analyze_chapter,
    _smart_truncate,
    _compute_signals,
)

T = TypeVar("T")


# ── Test doubles ─────────────────────────────────────────────────────────────


class MockDeepReaderClient:
    """Returns a fixed ChapterAnalysis."""

    def __init__(self, analysis: ChapterAnalysis | None = None) -> None:
        self._analysis = analysis or ChapterAnalysis(
            chapter_number=1,
            chapter_title="Mock Chapter",
            concepts=[
                ConceptEntry(
                    name="Bond",
                    definition="A fixed-income instrument.",
                    concept_type="definition",
                    section_title="Intro",
                    importance="core",
                ),
                ConceptEntry(
                    name="Yield",
                    definition="Return on a bond.",
                    concept_type="formula",
                    section_title="Returns",
                    importance="supporting",
                ),
            ],
            prerequisites=[
                PrerequisiteLink(
                    source_concept="Yield",
                    target_concept="Bond",
                    relationship="requires",
                ),
            ],
            section_characterizations=[
                SectionCharacterization(
                    section_title="Intro",
                    dominant_content_type="conceptual",
                    has_definitions=True,
                ),
            ],
            logical_flow="Starts with definitions, then formulas.",
            core_learning_outcome="Understand bond basics.",
        )
        self.call_count = 0
        self.last_system_prompt = ""
        self.last_user_prompt = ""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return "mock"

    def complete_light(self, system_prompt: str, user_prompt: str) -> str:
        return self.complete(system_prompt, user_prompt)

    def complete_structured(
        self, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> T:
        self.call_count += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self._analysis  # type: ignore[return-value]

    def complete_structured_light(
        self, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> T:
        return self.complete_structured(system_prompt, user_prompt, response_model)


class FailingDeepReaderClient:
    """Always fails — tests fallback behavior."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("fail")

    def complete_light(self, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("fail")

    def complete_structured(
        self, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> T:
        raise RuntimeError("LLM unavailable")

    def complete_structured_light(
        self, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> T:
        raise RuntimeError("LLM unavailable")


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_chapter(
    num_sections: int = 2,
    text_length: int = 300,
) -> Chapter:
    sections = tuple(
        Section(
            title=f"Section {i+1}",
            level=2,
            start_page=i * 5 + 1,
            end_page=(i + 1) * 5,
            text="A" * text_length,
        )
        for i in range(num_sections)
    )
    return Chapter(
        chapter_number=1,
        title="Test Chapter",
        start_page=1,
        end_page=num_sections * 5,
        sections=sections,
    )


# ── Tests ────────────────────────────────────────────────────────────────────


class TestAnalyzeChapter:
    """Tests for analyze_chapter()."""

    def test_returns_chapter_analysis(self) -> None:
        client = MockDeepReaderClient()
        chapter = _make_chapter()

        result = analyze_chapter(chapter, client)

        assert isinstance(result, ChapterAnalysis)
        assert result.chapter_number == 1
        assert result.chapter_title == "Test Chapter"
        assert len(result.concepts) == 2
        assert client.call_count == 1

    def test_includes_prior_concepts_in_prompt(self) -> None:
        client = MockDeepReaderClient()
        chapter = _make_chapter()
        prior = ["Present value", "Discount rate"]

        analyze_chapter(chapter, client, prior_chapter_concepts=prior)

        assert "Present value" in client.last_user_prompt
        assert "Discount rate" in client.last_user_prompt

    def test_includes_signals_in_prompt(self) -> None:
        """Pre-analysis signals should appear in the user prompt."""
        client = MockDeepReaderClient()
        section = Section(
            title="Math Section",
            level=2,
            start_page=1,
            end_page=5,
            text="Step 1: Calculate. Step 2: Verify. The formula $E=mc^2$ applies.",
        )
        chapter = Chapter(
            chapter_number=1,
            title="Math",
            start_page=1,
            end_page=5,
            sections=(section,),
        )

        analyze_chapter(chapter, client)

        assert "signals" in client.last_user_prompt.lower()

    def test_raises_on_llm_failure(self) -> None:
        """When LLM fails, the error propagates instead of silently degrading."""
        client = FailingDeepReaderClient()
        chapter = _make_chapter()

        with pytest.raises(RuntimeError):
            analyze_chapter(chapter, client)


class TestAnalyzeBook:
    """Tests for analyze_book()."""

    def test_analyzes_all_chapters(self) -> None:
        from src.extraction.types import Book

        ch1 = _make_chapter()
        ch2 = Chapter(
            chapter_number=2,
            title="Chapter 2",
            start_page=11,
            end_page=20,
            sections=(
                Section(title="S1", level=2, start_page=11, end_page=20, text="B" * 300),
            ),
        )
        book = Book(
            title="Test Book",
            author="Author",
            total_pages=20,
            chapters=(ch1, ch2),
        )
        client = MockDeepReaderClient()

        results = analyze_book(book, client)

        assert len(results) == 2
        assert client.call_count == 2

    def test_accumulates_concepts_across_chapters(self) -> None:
        """Prior chapter concepts should be passed to subsequent chapters."""
        from src.extraction.types import Book

        ch1 = _make_chapter()
        ch2 = Chapter(
            chapter_number=2,
            title="Chapter 2",
            start_page=11,
            end_page=20,
            sections=(
                Section(title="S1", level=2, start_page=11, end_page=20, text="B" * 300),
            ),
        )
        book = Book(
            title="Test Book",
            author="Author",
            total_pages=20,
            chapters=(ch1, ch2),
        )

        prompts: list[str] = []

        class CapturingClient:
            def complete(self, system_prompt: str, user_prompt: str) -> str:
                return "mock"

            def complete_light(self, system_prompt: str, user_prompt: str) -> str:
                return "mock"

            def complete_structured(
                self, system_prompt: str, user_prompt: str, response_model: type
            ):
                prompts.append(user_prompt)
                return ChapterAnalysis(
                    chapter_number=1,
                    chapter_title="X",
                    concepts=[
                        ConceptEntry(
                            name=f"Concept from ch{len(prompts)}",
                            definition="Def.",
                            concept_type="definition",
                            section_title="S",
                        ),
                    ],
                )

            def complete_structured_light(
                self, system_prompt: str, user_prompt: str, response_model: type
            ):
                return self.complete_structured(system_prompt, user_prompt, response_model)

        analyze_book(book, CapturingClient())

        # Second prompt should mention the concept from chapter 1
        assert len(prompts) == 2
        assert "Concept from ch1" in prompts[1]


class TestSmartTruncate:
    """Tests for _smart_truncate()."""

    def test_short_text_unchanged(self) -> None:
        text = "Short text."
        assert _smart_truncate(text, 100) == text

    def test_truncates_at_paragraph(self) -> None:
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = _smart_truncate(text, 40)
        assert result.endswith("[... content truncated for length ...]")
        assert "Third" not in result

    def test_truncates_within_limit(self) -> None:
        text = "x" * 100
        result = _smart_truncate(text, 50)
        assert len(result) <= 100  # truncated + suffix


