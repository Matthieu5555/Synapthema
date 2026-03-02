"""Tests for generic checkpoint load/save (Story 1.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.checkpoint import load_checkpoint, save_checkpoint, save_checkpoint_raw
from src.transformation.analysis_types import ChapterAnalysis, ConceptGraph
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


# ── load_checkpoint tests ──────────────────────────────────────────────────


class TestLoadCheckpointMissing:
    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.json"
        assert load_checkpoint(path, CurriculumBlueprint) is None

    def test_returns_none_for_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not valid json", encoding="utf-8")
        assert load_checkpoint(path, CurriculumBlueprint) is None

    def test_returns_none_for_wrong_schema(self, tmp_path: Path) -> None:
        path = tmp_path / "wrong.json"
        path.write_text('{"totally": "wrong"}', encoding="utf-8")
        assert load_checkpoint(path, CurriculumBlueprint) is None

    def test_returns_none_for_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        path.write_text("", encoding="utf-8")
        assert load_checkpoint(path, CurriculumBlueprint) is None


class TestLoadCheckpointPydanticModels:
    def test_roundtrip_blueprint(self, tmp_path: Path) -> None:
        blueprint = _make_blueprint()
        path = tmp_path / "bp.json"
        save_checkpoint(path, blueprint)

        loaded = load_checkpoint(path, CurriculumBlueprint)
        assert loaded is not None
        assert loaded.course_title == "Test Course"
        assert len(loaded.modules) == 2

    def test_roundtrip_training_module_list(self, tmp_path: Path) -> None:
        modules = [_make_training_module(1), _make_training_module(2)]
        path = tmp_path / "modules.json"
        save_checkpoint(path, modules)

        loaded = load_checkpoint(path, list[TrainingModule])
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].chapter_number == 1
        assert loaded[1].chapter_number == 2

    def test_roundtrip_chapter_analysis(self, tmp_path: Path) -> None:
        analysis = _make_analysis(3)
        path = tmp_path / "analysis.json"
        save_checkpoint(path, analysis)

        loaded = load_checkpoint(path, ChapterAnalysis)
        assert loaded is not None
        assert loaded.chapter_number == 3
        assert loaded.chapter_title == "Chapter 3"

    def test_roundtrip_concept_graph(self, tmp_path: Path) -> None:
        graph = _make_concept_graph()
        path = tmp_path / "graph.json"
        save_checkpoint(path, graph)

        loaded = load_checkpoint(path, ConceptGraph)
        assert loaded is not None
        assert loaded.concepts == []
        assert loaded.edges == []


class TestSaveCheckpoint:
    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "bp.json"
        save_checkpoint(path, _make_blueprint())
        assert path.exists()

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "bp.json"
        save_checkpoint(path, _make_blueprint(num_modules=1))
        save_checkpoint(path, _make_blueprint(num_modules=3))

        loaded = load_checkpoint(path, CurriculumBlueprint)
        assert loaded is not None
        assert len(loaded.modules) == 3

    def test_pydantic_model_produces_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bp.json"
        save_checkpoint(path, _make_blueprint())
        content = path.read_text(encoding="utf-8")
        assert '"course_title"' in content
        assert '"Test Course"' in content


class TestSaveCheckpointRaw:
    def test_saves_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "raw.json"
        save_checkpoint_raw(path, {"key": "value", "count": 42})
        content = path.read_text(encoding="utf-8")
        assert '"key": "value"' in content
        assert '"count": 42' in content

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "b" / "raw.json"
        save_checkpoint_raw(path, [1, 2, 3])
        assert path.exists()
