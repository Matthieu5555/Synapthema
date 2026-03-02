"""Tests for the structure detector — TOC parsing and chapter identification."""

import json
from unittest.mock import MagicMock

from src.extraction.structure_detector import (
    TocEntry,
    _collapse_spaced_letters,
    _filter_front_matter,
    _is_chapter_entry,
    _normalize_title,
    _page_range_fallback,
    detect_subsections_with_llm,
    identify_chapters,
)
from src.transformation.llm_client import LLMError


class TestNormalizeTitle:
    def test_collapses_whitespace(self) -> None:
        assert _normalize_title("  hello   world  ") == "hello world"

    def test_strips_edges(self) -> None:
        assert _normalize_title("  title  ") == "title"

    def test_handles_tabs_and_newlines(self) -> None:
        assert _normalize_title("hello\t\nworld") == "hello world"


class TestIsChapterEntry:
    def test_recognizes_standard_chapter(self) -> None:
        entry = TocEntry(level=2, title="Chapter 1: Introduction", page=5)
        assert _is_chapter_entry(entry)

    def test_recognizes_uppercase_chapter(self) -> None:
        entry = TocEntry(level=2, title="CHAPTER 12 Risk", page=100)
        assert _is_chapter_entry(entry)

    def test_recognizes_level1_chapter(self) -> None:
        entry = TocEntry(level=1, title="Chapter 3: Analysis", page=30)
        assert _is_chapter_entry(entry)

    def test_accepts_level_3_chapter(self) -> None:
        """Level 3 chapters are valid (e.g., books with Part > Chapter nesting)."""
        entry = TocEntry(level=3, title="Chapter 1: Intro", page=5)
        assert _is_chapter_entry(entry)

    def test_rejects_non_chapter_level(self) -> None:
        entry = TocEntry(level=4, title="Chapter 1: Intro", page=5)
        assert not _is_chapter_entry(entry)

    def test_rejects_non_chapter_title(self) -> None:
        entry = TocEntry(level=2, title="Appendix A", page=200)
        assert not _is_chapter_entry(entry)


class TestFilterFrontMatter:
    def test_filters_known_front_matter(self) -> None:
        entries = (
            TocEntry(level=1, title="Cover", page=1),
            TocEntry(level=1, title="Table of Contents", page=3),
            TocEntry(level=1, title="Part I", page=10),
            TocEntry(level=2, title="Chapter 1", page=15),
        )
        result = _filter_front_matter(entries)
        assert len(result) == 2
        assert result[0].title == "Part I"

    def test_no_front_matter(self) -> None:
        entries = (
            TocEntry(level=2, title="Chapter 1", page=1),
            TocEntry(level=2, title="Chapter 2", page=20),
        )
        result = _filter_front_matter(entries)
        assert len(result) == 2


class TestIdentifyChapters:
    def test_identifies_chapters_with_sections(self) -> None:
        entries = (
            TocEntry(level=2, title="Chapter 1: Intro", page=1),
            TocEntry(level=3, title="1.1 Background", page=2),
            TocEntry(level=3, title="1.2 Motivation", page=5),
            TocEntry(level=2, title="Chapter 2: Methods", page=10),
            TocEntry(level=3, title="2.1 Approach", page=11),
        )
        result = identify_chapters(entries, total_pages=20)

        assert len(result) == 2

        ch1_entry, ch1_children, ch1_start, ch1_end = result[0]
        assert ch1_entry.title == "Chapter 1: Intro"
        assert len(ch1_children) == 2
        assert ch1_start == 1
        assert ch1_end == 9  # page before chapter 2

        ch2_entry, ch2_children, ch2_start, ch2_end = result[1]
        assert ch2_entry.title == "Chapter 2: Methods"
        assert len(ch2_children) == 1
        assert ch2_end == 20  # last page

    def test_short_titles_fall_back_to_page_chunks(self) -> None:
        """Entries with short titles that don't meet MIN_CHAPTER_TITLE_LENGTH
        should trigger page-range fallback."""
        entries = (
            TocEntry(level=3, title="Section A", page=1),
            TocEntry(level=3, title="Section B", page=5),
        )
        result = identify_chapters(entries, total_pages=10)
        # Should get synthetic page-range chunks, not 0 results
        assert len(result) >= 1
        assert result[0][0].title.startswith("Part")

    def test_shallowest_level_used_as_chapter(self) -> None:
        """When no 'Chapter N' pattern, the shallowest level with long titles
        should be used as chapters — regardless of what that level number is."""
        entries = (
            TocEntry(level=3, title="Introduction to Fixed Income", page=1),
            TocEntry(level=4, title="Bond Basics", page=3),
            TocEntry(level=3, title="Advanced Bond Analytics", page=10),
            TocEntry(level=4, title="Duration", page=12),
        )
        result = identify_chapters(entries, total_pages=20)
        assert len(result) == 2
        assert result[0][0].title == "Introduction to Fixed Income"
        assert result[1][0].title == "Advanced Bond Analytics"

    def test_single_chapter_no_sections(self) -> None:
        entries = (TocEntry(level=2, title="Chapter 1 Only", page=1),)
        result = identify_chapters(entries, total_pages=50)

        assert len(result) == 1
        _, children, start, end = result[0]
        assert len(children) == 0
        assert start == 1
        assert end == 50

    def test_level1_entries_as_chapters(self) -> None:
        """L1 entries from 3-level font detection should be treated as chapters."""
        entries = (
            TocEntry(level=1, title="LEARNING MODULE 1 The Term Structure", page=11),
            TocEntry(level=2, title="SPOT RATES AND FORWARD RATES", page=12),
            TocEntry(level=2, title="YIELD CURVE MOVEMENTS", page=20),
            TocEntry(level=1, title="LEARNING MODULE 2 Credit Analysis", page=30),
            TocEntry(level=2, title="CREDIT RISK OVERVIEW", page=31),
        )
        result = identify_chapters(entries, total_pages=50)

        assert len(result) == 2
        ch1_entry, ch1_children, _, _ = result[0]
        assert ch1_entry.title == "LEARNING MODULE 1 The Term Structure"
        assert len(ch1_children) == 2  # L2 sections
        assert ch1_children[0].title == "SPOT RATES AND FORWARD RATES"


class TestCollapseSpacedLetters:
    def test_collapses_spaced_word(self) -> None:
        assert _collapse_spaced_letters("L E A R N I N G") == "LEARNING"

    def test_collapses_spaced_phrase(self) -> None:
        result = _collapse_spaced_letters("L E A R N I N G  M O D U L E")
        assert result == "LEARNING MODULE"

    def test_preserves_normal_text(self) -> None:
        assert _collapse_spaced_letters("Normal text here") == "Normal text here"

    def test_preserves_short_text(self) -> None:
        assert _collapse_spaced_letters("AB") == "AB"


class TestPageRangeFallback:
    def test_splits_into_chunks(self) -> None:
        result = _page_range_fallback(100)
        assert len(result) >= 2
        # First chunk starts at page 1
        assert result[0][2] == 1
        # Last chunk ends at page 100
        assert result[-1][3] == 100

    def test_single_chunk_for_short_doc(self) -> None:
        result = _page_range_fallback(10)
        assert len(result) == 1
        assert result[0][2] == 1
        assert result[0][3] == 10

    def test_no_gaps_between_chunks(self) -> None:
        result = _page_range_fallback(75)
        for i in range(len(result) - 1):
            current_end = result[i][3]
            next_start = result[i + 1][2]
            assert next_start == current_end + 1


# ── LLM subsection detection ────────────────────────────────────────────────


def _make_mock_doc(pages: dict[int, str]) -> MagicMock:
    """Create a mock fitz.Document with given page texts (0-indexed keys)."""
    doc = MagicMock()
    doc.__len__ = lambda self: max(pages.keys()) + 1 if pages else 0

    def getitem(self: object, idx: int) -> MagicMock:
        page = MagicMock()
        page.get_text.return_value = pages.get(idx, "")
        return page

    doc.__getitem__ = getitem
    return doc


class TestDetectSubsectionsWithLLM:
    """Tests for the LLM-based subsection detection."""

    def test_returns_entries_on_success(self) -> None:
        doc = _make_mock_doc({0: "page 1 text", 1: "page 2 text"})
        llm = MagicMock()
        llm.complete_light.return_value = json.dumps([
            {"level": 2, "title": "2.1 Random Variables", "page": 1},
            {"level": 2, "title": "2.2 Distributions", "page": 2},
            {"level": 3, "title": "2.2.1 Normal", "page": 2},
        ])

        result = detect_subsections_with_llm(doc, 1, 2, "Chapter 2", llm)

        assert len(result) == 3
        assert result[0].title == "2.1 Random Variables"
        assert result[0].level == 2
        assert result[0].page == 1
        assert result[2].title == "2.2.1 Normal"
        assert result[2].level == 3

    def test_returns_empty_on_llm_failure(self) -> None:
        doc = _make_mock_doc({0: "page 1 text"})
        llm = MagicMock()
        llm.complete_light.side_effect = LLMError("API error")

        result = detect_subsections_with_llm(doc, 1, 1, "Chapter 1", llm)

        assert result == ()

    def test_returns_empty_when_fewer_than_2_entries(self) -> None:
        doc = _make_mock_doc({0: "page 1 text"})
        llm = MagicMock()
        llm.complete_light.return_value = json.dumps([
            {"level": 2, "title": "Only One Section", "page": 1},
        ])

        result = detect_subsections_with_llm(doc, 1, 1, "Chapter 1", llm)

        assert result == ()

    def test_returns_empty_on_invalid_json(self) -> None:
        doc = _make_mock_doc({0: "page 1 text"})
        llm = MagicMock()
        llm.complete_light.return_value = "not valid json at all"

        result = detect_subsections_with_llm(doc, 1, 1, "Chapter 1", llm)

        assert result == ()

    def test_clamps_pages_to_chapter_range(self) -> None:
        doc = _make_mock_doc({0: "p1", 1: "p2", 2: "p3"})
        llm = MagicMock()
        llm.complete_light.return_value = json.dumps([
            {"level": 2, "title": "Section A", "page": 1},
            {"level": 2, "title": "Section B", "page": 2},
            {"level": 2, "title": "Section C", "page": 99},  # above range
        ])

        result = detect_subsections_with_llm(doc, 1, 3, "Chapter", llm)

        assert len(result) == 3
        assert result[0].page == 1
        assert result[1].page == 2
        assert result[2].page == 3   # clamped down from 99

    def test_sorts_by_page_and_level(self) -> None:
        doc = _make_mock_doc({0: "p1", 1: "p2"})
        llm = MagicMock()
        llm.complete_light.return_value = json.dumps([
            {"level": 2, "title": "Late Section", "page": 2},
            {"level": 2, "title": "Early Section", "page": 1},
            {"level": 3, "title": "Subsection of Early", "page": 1},
        ])

        result = detect_subsections_with_llm(doc, 1, 2, "Chapter", llm)

        assert result[0].title == "Early Section"
        assert result[1].title == "Subsection of Early"
        assert result[2].title == "Late Section"

    def test_returns_empty_for_empty_pages(self) -> None:
        doc = _make_mock_doc({0: "", 1: ""})
        llm = MagicMock()

        result = detect_subsections_with_llm(doc, 1, 2, "Chapter", llm)

        assert result == ()
        llm.complete_light.assert_not_called()
