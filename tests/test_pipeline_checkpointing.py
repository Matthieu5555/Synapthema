"""Tests for pipeline checkpointing and resume (Task 10)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.extraction.types import Book, Chapter, Section
from src.pipeline import (
    _load_analyses_checkpoint,
    _load_blueprint_checkpoint,
    _load_books_checkpoint,
    _load_training_modules_checkpoint,
    _regroup_analyses_by_book,
    _save_blueprint_json,
    _save_books_json,
    _save_training_json,
    _save_analyses_json,
)
from src.transformation.analysis_types import (
    ChapterAnalysis,
    ConceptGraph,
)
from src.transformation.types import (
    CurriculumBlueprint,
    Flashcard,
    FlashcardElement,
    ModuleBlueprint,
    SectionBlueprint,
    Slide,
    SlideElement,
    TrainingModule,
    TrainingSection,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_book(num_chapters: int = 2) -> Book:
    chapters = tuple(
        Chapter(
            chapter_number=i + 1,
            title=f"Chapter {i + 1}",
            start_page=i * 10 + 1,
            end_page=(i + 1) * 10,
            sections=(
                Section(
                    title=f"Section {i + 1}.1",
                    level=3,
                    start_page=i * 10 + 1,
                    end_page=(i + 1) * 10,
                    text="A" * 200,
                ),
            ),
        )
        for i in range(num_chapters)
    )
    return Book(
        title="Test Book",
        author="Test Author",
        total_pages=num_chapters * 10,
        chapters=chapters,
    )


def _make_analysis(chapter_number: int) -> ChapterAnalysis:
    return ChapterAnalysis(
        chapter_number=chapter_number,
        chapter_title=f"Chapter {chapter_number}",
    )


def _make_concept_graph() -> ConceptGraph:
    return ConceptGraph(
        concepts=[],
        edges=[],
        topological_order=[],
        foundation_concepts=[],
        advanced_concepts=[],
    )


def _make_blueprint(num_modules: int = 2) -> CurriculumBlueprint:
    return CurriculumBlueprint(
        course_title="Test Course",
        course_summary="A test course.",
        modules=[
            ModuleBlueprint(
                title=f"Module {i + 1}",
                source_chapter_number=i + 1,
                sections=[
                    SectionBlueprint(
                        title=f"Section {i + 1}.1",
                        source_section_title=f"Section {i + 1}.1",
                    ),
                ],
            )
            for i in range(num_modules)
        ],
    )


def _make_training_module(chapter_number: int) -> TrainingModule:
    return TrainingModule(
        chapter_number=chapter_number,
        title=f"Module {chapter_number}",
        sections=[
            TrainingSection(
                title=f"Section {chapter_number}.1",
                elements=[
                    SlideElement(
                        bloom_level="understand",
                        slide=Slide(title="Test Slide", content="Test content."),
                    ),
                    FlashcardElement(
                        bloom_level="remember",
                        flashcard=Flashcard(front="Q", back="A"),
                    ),
                ],
            ),
        ],
    )


# ── Checkpoint loader tests ─────────────────────────────────────────────────


class TestLoadBooksCheckpoint:
    def test_loads_valid_checkpoint(self, tmp_path: Path) -> None:
        book = _make_book()
        path = tmp_path / "book_structure.json"
        _save_books_json([book], path)

        result = _load_books_checkpoint(path)
        assert result is not None
        assert len(result) == 1
        assert result[0].title == "Test Book"
        assert len(result[0].chapters) == 2

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.json"
        assert _load_books_checkpoint(path) is None

    def test_returns_none_for_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not valid json", encoding="utf-8")
        assert _load_books_checkpoint(path) is None

    def test_returns_none_for_empty_list(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        path.write_text("[]", encoding="utf-8")
        assert _load_books_checkpoint(path) is None


class TestLoadAnalysesCheckpoint:
    def test_loads_valid_checkpoint(self, tmp_path: Path) -> None:
        analyses = [_make_analysis(1), _make_analysis(2)]
        graph = _make_concept_graph()
        path = tmp_path / "chapter_analyses.json"
        _save_analyses_json(analyses, graph, path)

        result = _load_analyses_checkpoint(path, expected_count=2)
        assert result is not None
        loaded_analyses, loaded_graph = result
        assert len(loaded_analyses) == 2
        assert loaded_analyses[0].chapter_number == 1

    def test_returns_none_for_wrong_count(self, tmp_path: Path) -> None:
        analyses = [_make_analysis(1)]
        graph = _make_concept_graph()
        path = tmp_path / "chapter_analyses.json"
        _save_analyses_json(analyses, graph, path)

        result = _load_analyses_checkpoint(path, expected_count=3)
        assert result is None

    def test_returns_none_for_missing_concept_graph(self, tmp_path: Path) -> None:
        path = tmp_path / "no_graph.json"
        data = {"chapter_analyses": [_make_analysis(1).model_dump(mode="json")]}
        path.write_text(json.dumps(data), encoding="utf-8")

        result = _load_analyses_checkpoint(path, expected_count=1)
        assert result is None

    def test_returns_none_for_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{broken", encoding="utf-8")
        assert _load_analyses_checkpoint(path, expected_count=1) is None


class TestLoadBlueprintCheckpoint:
    def test_loads_valid_checkpoint(self, tmp_path: Path) -> None:
        blueprint = _make_blueprint()
        path = tmp_path / "curriculum_blueprint.json"
        _save_blueprint_json(blueprint, path)

        result = _load_blueprint_checkpoint(path)
        assert result is not None
        assert result.course_title == "Test Course"
        assert len(result.modules) == 2

    def test_returns_none_for_empty_modules(self, tmp_path: Path) -> None:
        path = tmp_path / "empty_bp.json"
        data = CurriculumBlueprint(
            course_title="Empty", modules=[],
        ).model_dump(mode="json")
        path.write_text(json.dumps(data), encoding="utf-8")

        assert _load_blueprint_checkpoint(path) is None

    def test_returns_none_for_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("nope", encoding="utf-8")
        assert _load_blueprint_checkpoint(path) is None


class TestLoadTrainingModulesCheckpoint:
    def test_loads_valid_checkpoint(self, tmp_path: Path) -> None:
        modules = [_make_training_module(1), _make_training_module(2)]
        path = tmp_path / "training_modules.json"
        _save_training_json(modules, path)

        result = _load_training_modules_checkpoint(path)
        assert result is not None
        assert len(result) == 2
        assert result[0].chapter_number == 1

    def test_returns_none_for_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        assert _load_training_modules_checkpoint(path) is None


# ── Helper tests ────────────────────────────────────────────────────────────


class TestRegroupAnalysesByBook:
    def test_regroups_correctly(self) -> None:
        analyses = [_make_analysis(1), _make_analysis(2), _make_analysis(3)]
        books = [_make_book(num_chapters=2), _make_book(num_chapters=1)]

        result = _regroup_analyses_by_book(analyses, books)
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 1
        assert result[0][0].chapter_number == 1
        assert result[1][0].chapter_number == 3


class TestSaveBlueprintJson:
    def test_roundtrip(self, tmp_path: Path) -> None:
        blueprint = _make_blueprint()
        path = tmp_path / "bp.json"
        _save_blueprint_json(blueprint, path)

        assert path.exists()
        loaded = _load_blueprint_checkpoint(path)
        assert loaded is not None
        assert loaded.course_title == blueprint.course_title
        assert len(loaded.modules) == len(blueprint.modules)

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "bp.json"
        _save_blueprint_json(_make_blueprint(), path)
        assert path.exists()
