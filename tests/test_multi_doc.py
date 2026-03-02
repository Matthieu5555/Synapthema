"""Tests for multi-document course discovery (Task 1).

Tests plan_multi_document_curriculum(), multi-doc content summary,
passthrough multi-doc blueprint, config with multiple PDFs,
and backward compatibility with single-PDF mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest

from src.config import Config, InputSource, _resolve_input_sources, _slugify_pdf_name
from src.extraction.types import Book, Chapter, Section
from src.transformation.curriculum_planner import (
    _build_multi_doc_content_summary,
    plan_multi_document_curriculum,
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


def _make_book(
    title: str = "Book A",
    author: str = "Author A",
    chapters: tuple[Chapter, ...] | None = None,
) -> Book:
    if chapters is None:
        chapters = (
            _make_chapter(
                number=1,
                title="Foundations",
                sections=(
                    _make_section("Basic Concepts"),
                    _make_section("Core Definitions"),
                ),
            ),
            _make_chapter(
                number=2,
                title="Applications",
                sections=(
                    _make_section("Practical Use"),
                ),
            ),
        )
    return Book(
        title=title,
        author=author,
        total_pages=50,
        chapters=chapters,
    )


def _make_two_books() -> list[Book]:
    book_a = _make_book(
        title="Intro to Widgets",
        author="Alice",
        chapters=(
            _make_chapter(
                number=1,
                title="What are Widgets",
                sections=(
                    _make_section("Widget History"),
                    _make_section("Widget Types"),
                ),
            ),
        ),
    )
    book_b = _make_book(
        title="Advanced Gadgets",
        author="Bob",
        chapters=(
            _make_chapter(
                number=1,
                title="Gadget Fundamentals",
                sections=(
                    _make_section("Gadget Theory"),
                ),
            ),
            _make_chapter(
                number=2,
                title="Gadget Applications",
                sections=(
                    _make_section("Real-World Gadgets"),
                    _make_section("Gadget vs Widget"),
                ),
            ),
        ),
    )
    return [book_a, book_b]


class MockMultiDocClient:
    """Returns a multi-doc CurriculumBlueprint."""

    def __init__(self) -> None:
        self._blueprint = CurriculumBlueprint(
            course_title="Unified Widgets & Gadgets",
            course_summary="A unified course covering widgets and gadgets.",
            learner_journey="Foundations → Applications → Comparisons",
            modules=[
                ModuleBlueprint(
                    title="What are Widgets",
                    source_chapter_number=1,
                    source_book_index=0,
                    summary="Intro to widgets",
                    sections=[
                        SectionBlueprint(
                            title="Widget History",
                            source_section_title="Widget History",
                            source_book_index=0,
                            template="narrative",
                            bloom_target="understand",
                        ),
                    ],
                ),
                ModuleBlueprint(
                    title="Gadget Fundamentals",
                    source_chapter_number=1,
                    source_book_index=1,
                    summary="Gadget basics",
                    sections=[
                        SectionBlueprint(
                            title="Gadget Theory",
                            source_section_title="Gadget Theory",
                            source_book_index=1,
                            template="socratic",
                            bloom_target="understand",
                        ),
                    ],
                ),
                ModuleBlueprint(
                    title="Gadget Applications",
                    source_chapter_number=2,
                    source_book_index=1,
                    summary="Applied gadgets",
                    sections=[
                        SectionBlueprint(
                            title="Gadget vs Widget",
                            source_section_title="Gadget vs Widget",
                            source_book_index=1,
                            template="compare_contrast",
                            bloom_target="analyze",
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


class FailingClient:
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


# ── Tests: multi-doc curriculum planning ──────────────────────────────────────


class TestPlanMultiDocumentCurriculum:
    """Tests for plan_multi_document_curriculum()."""

    def test_returns_unified_blueprint(self) -> None:
        books = _make_two_books()
        client = MockMultiDocClient()

        blueprint = plan_multi_document_curriculum(books, client)

        assert blueprint.course_title == "Unified Widgets & Gadgets"
        assert len(blueprint.modules) == 3
        assert client.call_count == 1

    def test_modules_reference_correct_books(self) -> None:
        books = _make_two_books()
        client = MockMultiDocClient()

        blueprint = plan_multi_document_curriculum(books, client)

        assert blueprint.modules[0].source_book_index == 0
        assert blueprint.modules[1].source_book_index == 1
        assert blueprint.modules[2].source_book_index == 1

    def test_raises_on_llm_failure(self) -> None:
        from src.transformation.llm_client import LLMError

        books = _make_two_books()
        client = FailingClient()

        with pytest.raises(LLMError):
            plan_multi_document_curriculum(books, client)


# ── Tests: multi-doc content summary ──────────────────────────────────────────


class TestBuildMultiDocContentSummary:
    """Tests for the multi-doc content summary builder."""

    def test_includes_all_book_titles(self) -> None:
        books = _make_two_books()

        summary = _build_multi_doc_content_summary(books)

        assert "Intro to Widgets" in summary
        assert "Advanced Gadgets" in summary

    def test_includes_book_indices(self) -> None:
        books = _make_two_books()

        summary = _build_multi_doc_content_summary(books)

        assert "book_index=0" in summary
        assert "book_index=1" in summary

    def test_includes_sections_from_all_books(self) -> None:
        books = _make_two_books()

        summary = _build_multi_doc_content_summary(books)

        assert "Widget History" in summary
        assert "Gadget Theory" in summary
        assert "Gadget vs Widget" in summary

    def test_includes_corpus_count(self) -> None:
        books = _make_two_books()

        summary = _build_multi_doc_content_summary(books)

        assert "2 documents" in summary


class TestRichMultiDocContentSummary:
    """Tests for _build_rich_multi_doc_content_summary concept resolution."""

    def test_rich_multi_doc_summary_resolves_concepts(self) -> None:
        """Per-book summaries include concept graph info when graph provided."""
        from src.transformation.analysis_types import (
            ChapterAnalysis,
            ConceptGraph,
        )
        from src.transformation.curriculum_planner import (
            _build_rich_multi_doc_content_summary,
        )

        books = _make_two_books()
        # Minimal analyses (empty concepts list is fine — we test graph passthrough)
        analyses_per_book: list[list[ChapterAnalysis]] = [
            [
                ChapterAnalysis(chapter_number=1, chapter_title="What are Widgets"),
            ],
            [
                ChapterAnalysis(chapter_number=1, chapter_title="Gadget Fundamentals"),
                ChapterAnalysis(chapter_number=2, chapter_title="Gadget Applications"),
            ],
        ]
        graph = ConceptGraph(
            topological_order=["Widget", "Gadget", "Hybrid"],
            foundation_concepts=["Widget"],
            advanced_concepts=["Hybrid"],
        )

        result = _build_rich_multi_doc_content_summary(
            books, analyses_per_book, graph,
        )

        # The per-book rich summaries should include concept dependency info
        # (from _build_rich_content_summary receiving the graph)
        assert "Widget" in result
        assert "Gadget" in result
        assert "Hybrid" in result
        # Global concept order header
        assert "Concept Dependency Order" in result


# ── Tests: config with multiple PDFs ──────────────────────────────────────────


class TestInputSourceConfig:
    """Tests for InputSource and multi-source config."""

    def test_input_source_default_type(self) -> None:
        src = InputSource(path=Path("/tmp/doc.pdf"))
        assert src.source_type == "pdf"

    def test_config_pdf_path_backward_compat(self) -> None:
        """Config.pdf_path property returns first source's path."""
        config = Config(
            input_sources=[
                InputSource(path=Path("/tmp/a.pdf")),
                InputSource(path=Path("/tmp/b.pdf")),
            ],
            extracted_dir=Path("/tmp/extracted"),
            output_dir=Path("/tmp/output"),
            llm_api_key="key",
            llm_base_url="http://example.com",
            llm_model="test",
            llm_model_light="test",
            llm_temperature=0.3,
            llm_max_tokens=8192,
            embed_images=True,
            vision_enabled=False,
            document_type="auto",
            max_concurrent_llm=4,
        )
        assert config.pdf_path == Path("/tmp/a.pdf")

    def test_resolve_input_sources_single_pdf(self, tmp_path: Path) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        sources = _resolve_input_sources(pdf_path=pdf, pdf_paths=None, input_dir=None)

        assert len(sources) == 1
        assert sources[0].path == pdf.resolve()

    def test_resolve_input_sources_multiple_pdfs(self, tmp_path: Path) -> None:
        pdf_a = tmp_path / "a.pdf"
        pdf_b = tmp_path / "b.pdf"
        pdf_a.write_bytes(b"%PDF-1.4")
        pdf_b.write_bytes(b"%PDF-1.4")

        sources = _resolve_input_sources(
            pdf_path=None, pdf_paths=[pdf_a, pdf_b], input_dir=None
        )

        assert len(sources) == 2

    def test_resolve_input_sources_input_dir(self, tmp_path: Path) -> None:
        (tmp_path / "doc1.pdf").write_bytes(b"%PDF-1.4")
        (tmp_path / "doc2.pdf").write_bytes(b"%PDF-1.4")
        (tmp_path / "notes.txt").write_text("not a pdf")

        sources = _resolve_input_sources(
            pdf_path=None, pdf_paths=None, input_dir=tmp_path
        )

        assert len(sources) == 2
        assert all(s.path.suffix == ".pdf" for s in sources)

    def test_resolve_input_sources_pdf_paths_takes_precedence(self, tmp_path: Path) -> None:
        """pdf_paths takes precedence over pdf_path."""
        pdf_a = tmp_path / "a.pdf"
        pdf_b = tmp_path / "b.pdf"
        pdf_single = tmp_path / "single.pdf"
        for p in [pdf_a, pdf_b, pdf_single]:
            p.write_bytes(b"%PDF-1.4")

        sources = _resolve_input_sources(
            pdf_path=pdf_single, pdf_paths=[pdf_a, pdf_b], input_dir=None
        )

        assert len(sources) == 2
        assert sources[0].path.name == "a.pdf"
