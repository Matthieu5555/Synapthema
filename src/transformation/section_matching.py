"""Chapter/section matching utilities.

Extracted from curriculum_planner.py so both the planner and content_designer
can import these helpers without coupling to each other.
"""

from __future__ import annotations

import logging
import re

from src.extraction.types import Book, Chapter, Section
from src.transformation.types import ModuleBlueprint, SectionBlueprint

logger = logging.getLogger(__name__)


_CHAPTER_NUM_RE = re.compile(r"(?:chapter|ch\.?)\s+(\d+)", re.IGNORECASE)


def _extract_chapter_number_from_title(title: str) -> int | None:
    """Extract a chapter number from a title like 'CHAPTER 3 Fundamentals'."""
    m = _CHAPTER_NUM_RE.search(title)
    return int(m.group(1)) if m else None


def _normalize_section_title(title: str) -> str:
    """Normalize a section title for fuzzy matching.

    Strips numbering-period differences (e.g. "3.2.1 X" vs "3.2.1. X"),
    lowercases, and collapses whitespace.
    """
    t = title.strip().lower()
    # Normalize numbering: "3.2.1. " and "3.2.1 " → same form
    # Strip leading "X.Y.Z. " or "X.Y.Z " prefix entirely for comparison
    t = re.sub(r"^[\d]+\.[\d.]*\s*", "", t)
    # Also strip "CHAPTER N: " prefix
    t = re.sub(r"^chapter\s+\d+[:\s]*", "", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def find_matching_section(
    chapter: Chapter, section_bp: SectionBlueprint
) -> Section | None:
    """Find the extracted Section matching a blueprint section.

    Checks source_section_title against section titles, including subsections.
    Uses exact match first, then falls back to normalized fuzzy matching
    (strips numbering prefixes, case-insensitive).
    """
    target = section_bp.source_section_title.strip()
    if not target:
        target = section_bp.title.strip()

    # Collect all candidate sections (top-level + subsections)
    candidates: list[Section] = []
    for section in chapter.sections:
        candidates.append(section)
        for sub in section.subsections:
            candidates.append(sub)

    # Pass 1: exact match
    for section in candidates:
        if section.title.strip() == target:
            return section

    # Pass 2: normalized match (strips numbering prefixes, case-insensitive)
    target_norm = _normalize_section_title(target)
    if target_norm:
        for section in candidates:
            if _normalize_section_title(section.title) == target_norm:
                return section

    # Pass 3: substring containment (target text found in section title or vice versa)
    if target_norm and len(target_norm) >= 5:
        for section in candidates:
            section_norm = _normalize_section_title(section.title)
            if section_norm and (target_norm in section_norm or section_norm in target_norm):
                return section

    return None


def _count_section_matches(
    chapter: Chapter, section_bps: list[SectionBlueprint],
) -> int:
    """Count how many blueprint sections match sections in *chapter*."""
    return sum(
        1 for sbp in section_bps
        if find_matching_section(chapter, sbp) is not None
    )


def find_matching_chapter(book: Book, module_bp: ModuleBlueprint) -> Chapter | None:
    """Find the extracted Chapter that matches a blueprint module.

    Tries sequential chapter_number and title-embedded chapter number in
    parallel.  When both match **different** chapters, uses section-match
    count from the blueprint as a validation signal to pick the winner
    (preferring sequential on ties).  Falls back to title text matching.

    The title-embedded path handles the common case where PDFs start
    mid-book (e.g. "CHAPTER 3" is the first extracted chapter, numbered
    chapter_number=1 sequentially).  The LLM may reference the PDF's own
    chapter number (3) rather than the extraction's sequential number (1).
    """
    if module_bp.source_chapter_number is not None:
        # Pass 1: exact sequential chapter_number match
        sequential_match: Chapter | None = None
        for ch in book.chapters:
            if ch.chapter_number == module_bp.source_chapter_number:
                sequential_match = ch
                break

        # Pass 2: match against chapter number embedded in title
        # e.g. "CHAPTER 3 Fundamentals of Statistics" → 3
        title_match: Chapter | None = None
        for ch in book.chapters:
            title_num = _extract_chapter_number_from_title(ch.title)
            if title_num is not None and title_num == module_bp.source_chapter_number:
                title_match = ch
                break

        if sequential_match and not title_match:
            return sequential_match
        if title_match and not sequential_match:
            return title_match
        if sequential_match and title_match:
            if sequential_match is title_match:
                return sequential_match
            # Ambiguity: two different chapters claim the number.
            # Use blueprint section matches as a tiebreaker.
            if module_bp.sections:
                seq_hits = _count_section_matches(sequential_match, module_bp.sections)
                title_hits = _count_section_matches(title_match, module_bp.sections)
                if title_hits > seq_hits:
                    logger.info(
                        "Chapter resolution: title-embedded match '%s' "
                        "has more section hits (%d) than sequential match '%s' (%d)",
                        title_match.title, title_hits,
                        sequential_match.title, seq_hits,
                    )
                    return title_match
            return sequential_match

    # Pass 3: match by title text
    bp_title_lower = module_bp.title.lower().strip()
    for ch in book.chapters:
        if ch.title.lower().strip() == bp_title_lower:
            return ch

    return None
