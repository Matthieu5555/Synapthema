"""Chapter and section boundary detection from PDF table of contents.

Internal module — not part of the public API. Called by pdf_parser.py to
determine the hierarchical structure of the document.

Detection strategies (in priority order):
1. Native PDF TOC (bookmarks/outline) — most reliable
2. LLM-based TOC extraction from first pages — reads actual content
3. Page-range fallback — splits into even chunks

The font-based approach was removed because it picks up decorative text
(70pt drop caps, stylized headers) and produces garbage hierarchies.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import fitz
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.protocols import LLMClient

from src.protocols import LLMError

logger = logging.getLogger(__name__)

# ── Configurable constants ────────────────────────────────────────────────────

# Minimum character length for a top-level heading to qualify as a chapter.
# Prevents short noise like "1", "CONTENTS" from being treated as chapters.
MIN_CHAPTER_TITLE_LENGTH = 10

# How many pages per chunk when no headings are found at all.
PAGES_PER_CHUNK = 15

# How many pages to send to the LLM for TOC detection.
LLM_TOC_PAGES = 15


@dataclass(frozen=True)
class TocEntry:
    """A single entry from the document's table of contents.

    Attributes:
        level: Nesting depth (1=top-level chapter, 2=section, 3=subsection).
        title: Heading text.
        page: 1-indexed page number.
    """

    level: int
    title: str
    page: int


# ── Pydantic response models for structured LLM output ───────────────────────


class TocEntryResponse(BaseModel):
    """A single entry from LLM TOC extraction."""

    level: int = Field(default=1, description="Nesting depth (1=chapter, 2=section, 3=subsection)")
    title: str = Field(description="The heading text, cleaned up")
    page: int = Field(description="1-indexed page number where this section starts")


class TocExtractionResponse(BaseModel):
    """Complete TOC extraction result from the LLM."""

    entries: list[TocEntryResponse] = Field(
        default_factory=list,
        description="All structural headings found in the document, ordered by page number",
    )


class SubsectionExtractionResponse(BaseModel):
    """Subsection extraction result for a single chapter."""

    entries: list[TocEntryResponse] = Field(
        default_factory=list,
        description="Section and subsection headings within this chapter",
    )


# ── TOC-based detection ─────────────────────────────────────────────────────


def extract_toc_entries(doc: fitz.Document) -> tuple[TocEntry, ...]:
    """Extract and clean TOC entries from a PDF document.

    Filters out front-matter entries (cover, title page, copyright, etc.)
    that appear before the first substantive content. Normalizes whitespace
    in titles.

    Args:
        doc: An open PyMuPDF document.

    Returns:
        Ordered tuple of TocEntry objects. Empty if no TOC is embedded.
    """
    raw_toc = doc.get_toc()
    if not raw_toc:
        logger.warning("No embedded TOC found in document")
        return ()

    logger.info("Found %d raw TOC entries", len(raw_toc))

    entries = tuple(
        TocEntry(level=level, title=_normalize_title(title), page=page)
        for level, title, page in raw_toc
        if page > 0  # Filter invalid page references
    )

    return _filter_front_matter(entries)


# ── LLM-based TOC detection ────────────────────────────────────────────────


# System prompt for TOC extraction — tells the LLM what to look for.
_TOC_EXTRACTION_PROMPT = """\
You are analyzing a document to extract its structure (table of contents).

Your task: identify ALL chapters, sections, and subsections with their page numbers.

Look for:
1. A literal "Table of Contents" or "Contents" page listing sections with page numbers
2. Major chapter/module headings (e.g., "Chapter 1: ...", "Learning Module 1: ...", "Part I: ...")
3. Section headings within chapters
4. Any structural markers that indicate document organization

Rules:
- Include EVERY structural heading you can find, at all levels
- Page numbers must be integers, 1-indexed
- Titles should be clean text (no page numbers, no dots/leaders)
- If there's a TOC page, use the page numbers FROM the TOC (they refer to where content starts)
- If there's no TOC page, infer structure from headings you see in the text
- Order entries by page number
- Do NOT include front matter (cover, copyright, table of contents itself, preface)
- level: 1 = chapter/module, 2 = section, 3 = subsection"""


def detect_toc_with_llm(
    doc: fitz.Document,
    llm_client: LLMClient,
) -> tuple[TocEntry, ...]:
    """Use an LLM to extract document structure from the first pages.

    Reads the first LLM_TOC_PAGES pages of text and sends them to the LLM
    to identify the table of contents / document structure. Uses structured
    output (Pydantic model) for automatic validation and retry.

    Args:
        doc: An open PyMuPDF document.
        llm_client: An LLM client with a complete_structured_light() method.

    Returns:
        Tuple of TocEntry objects. Empty if detection fails.
    """
    pages_to_read = min(LLM_TOC_PAGES, len(doc))
    pages_text: list[str] = []

    for page_idx in range(pages_to_read):
        page = doc[page_idx]
        text: str = page.get_text()  # pyright: ignore[reportAssignmentType] -- PyMuPDF untyped
        text = text.strip()
        if text:
            pages_text.append(f"--- PAGE {page_idx + 1} ---\n{text}")

    if not pages_text:
        return ()

    combined_text = "\n\n".join(pages_text)

    # Truncate if too long (keep under ~12k chars to leave room for response)
    if len(combined_text) > 12000:
        combined_text = combined_text[:12000] + "\n\n[... truncated ...]"

    user_prompt = (
        f"Here are the first {pages_to_read} pages of a document. "
        f"Extract the complete table of contents / document structure.\n\n"
        f"{combined_text}"
    )

    try:
        result = llm_client.complete_structured_light(
            _TOC_EXTRACTION_PROMPT, user_prompt, TocExtractionResponse,
        )

        entries: list[TocEntry] = []
        for item in result.entries:
            if not item.title.strip() or item.page < 1:
                continue
            title = _collapse_spaced_letters(_normalize_title(item.title))
            entries.append(TocEntry(level=item.level, title=title, page=item.page))

        entries.sort(key=lambda e: (e.page, e.level))

        if entries:
            logger.info("LLM detected %d TOC entries", len(entries))
            return _filter_front_matter(tuple(entries))

        return ()

    except (LLMError, ValueError, TypeError, KeyError) as exc:
        logger.warning("LLM TOC detection failed: %s", exc)
        return ()


# ── LLM-based subsection detection ─────────────────────────────────────────


_SUBSECTION_EXTRACTION_PROMPT = """\
You are analyzing a single chapter from a document to extract its internal structure.

Your task: identify ALL section and subsection headings within this chapter, \
with their page numbers.

Look for:
1. Numbered headings (e.g., "2.1 Title", "2.1.1 Subtitle")
2. Named sections with consistent formatting
3. Any structural markers that indicate subsections within this chapter

Rules:
- Include EVERY heading you find within this chapter
- Page numbers must be integers, 1-indexed
- Titles should be clean text (no page numbers, no dots/leaders)
- Order entries by page number
- Do NOT include the chapter title itself (only its subsections)
- Do NOT include "Answers", "Solutions", or back-matter sections
- level: 2 = section, 3 = subsection (level 1 is the chapter itself)
- If you cannot identify clear subsections, return an empty entries list"""


def detect_subsections_with_llm(
    doc: fitz.Document,
    start_page: int,
    end_page: int,
    chapter_title: str,
    llm_client: LLMClient,
) -> tuple[TocEntry, ...]:
    """Use an LLM to detect subsection structure within a chapter.

    Sends the chapter's page text to the LLM and gets back structured
    section entries with titles, levels, and page numbers. Uses structured
    output (Pydantic model) for automatic validation and retry.

    Args:
        doc: An open PyMuPDF document.
        start_page: 1-indexed first page of the chapter (inclusive).
        end_page: 1-indexed last page of the chapter (inclusive).
        chapter_title: The chapter's title (for prompt context).
        llm_client: An LLM client with a complete_structured_light() method.

    Returns:
        Tuple of TocEntry objects. Empty if detection fails or finds < 2.
    """
    pages_text: list[str] = []
    for page_idx in range(max(0, start_page - 1), min(end_page, len(doc))):
        page = doc[page_idx]
        text: str = page.get_text()  # pyright: ignore[reportAssignmentType] -- PyMuPDF untyped
        text = text.strip()
        if text:
            pages_text.append(f"--- PAGE {page_idx + 1} ---\n{text}")

    if not pages_text:
        return ()

    combined_text = "\n\n".join(pages_text)

    # Truncate if too long (keep under ~12k chars to leave room for response)
    if len(combined_text) > 12000:
        combined_text = combined_text[:12000] + "\n\n[... truncated ...]"

    user_prompt = (
        f"Here is the chapter '{chapter_title}' "
        f"(pages {start_page}-{end_page}). "
        f"Extract all section and subsection headings.\n\n"
        f"{combined_text}"
    )

    try:
        result = llm_client.complete_structured_light(
            _SUBSECTION_EXTRACTION_PROMPT, user_prompt,
            SubsectionExtractionResponse,
        )

        entries: list[TocEntry] = []
        for item in result.entries:
            if not item.title.strip() or item.page < 1:
                continue
            # Clamp to chapter's page range
            page = max(start_page, min(item.page, end_page))
            entries.append(TocEntry(level=item.level, title=item.title, page=page))

        entries.sort(key=lambda e: (e.page, e.level))

        if len(entries) < 2:
            logger.debug(
                "LLM found %d subsections for '%s' (below threshold, ignoring)",
                len(entries), chapter_title,
            )
            return ()

        logger.info(
            "LLM detected %d subsections in '%s' (pages %d-%d)",
            len(entries), chapter_title, start_page, end_page,
        )
        return tuple(entries)

    except (LLMError, ValueError, TypeError, KeyError) as exc:
        logger.warning(
            "LLM subsection detection failed for '%s': %s", chapter_title, exc,
        )
        return ()


# ── Chapter identification ─────────────────────────────────────────────────


def identify_chapters(
    toc_entries: tuple[TocEntry, ...],
    total_pages: int,
) -> tuple[tuple[TocEntry, tuple[TocEntry, ...], int, int], ...]:
    """Group TOC entries into chapters with their child sections and page ranges.

    Uses the shallowest level present as the chapter boundary. This is
    fully adaptive — works whether the document uses "Chapter N" titles,
    numbered modules, descriptive headings, or any other convention.

    Strategies tried in order:
    1. Entries matching "Chapter N" patterns (any level)
    2. Entries at the shallowest (min) level with meaningful titles
    3. Page-range fallback — split document into even chunks

    Args:
        toc_entries: Cleaned TOC entries from extract_toc_entries or
            detect_toc_with_llm.
        total_pages: Total page count of the document.

    Returns:
        Tuple of (chapter_entry, child_sections, start_page, end_page).
    """
    # Strategy 1: explicit "Chapter N" / "CHAPTER N" patterns
    chapter_indices = [
        i for i, entry in enumerate(toc_entries)
        if _is_chapter_entry(entry)
    ]

    # Strategy 2: shallowest level with meaningful titles
    if not chapter_indices and toc_entries:
        min_level = min(e.level for e in toc_entries)
        chapter_indices = [
            i for i, entry in enumerate(toc_entries)
            if entry.level == min_level
            and len(entry.title) >= MIN_CHAPTER_TITLE_LENGTH
            and entry.title.lower() not in _BACK_MATTER_TITLES
            and entry.title.lower() not in _FRONT_MATTER_TITLES
            and not _is_front_matter_title(entry.title.lower())
        ]

    # Strategy 3: page-range fallback
    if not chapter_indices:
        logger.warning("No chapter headings detected — using page-range chunks")
        return _page_range_fallback(total_pages)

    logger.info("Detected %d chapters", len(chapter_indices))

    chapters: list[tuple[TocEntry, tuple[TocEntry, ...], int, int]] = []

    for idx, chapter_idx in enumerate(chapter_indices):
        chapter_entry = toc_entries[chapter_idx]
        chapter_level = chapter_entry.level

        # End page
        if idx + 1 < len(chapter_indices):
            next_chapter_idx = chapter_indices[idx + 1]
            end_page = max(chapter_entry.page, toc_entries[next_chapter_idx].page - 1)
        else:
            end_page = total_pages
            for subsequent in toc_entries[chapter_idx + 1:]:
                if subsequent.level <= chapter_level:
                    end_page = subsequent.page - 1
                    break

        # Child sections: everything deeper than chapter level
        boundary = chapter_indices[idx + 1] if idx + 1 < len(chapter_indices) else len(toc_entries)
        child_sections = tuple(
            entry for entry in toc_entries[chapter_idx + 1: boundary]
            if entry.level > chapter_level
        )

        chapters.append((chapter_entry, child_sections, chapter_entry.page, end_page))

    return tuple(chapters)


def _page_range_fallback(
    total_pages: int,
) -> tuple[tuple[TocEntry, tuple[TocEntry, ...], int, int], ...]:
    """Split the document into even page-range chunks when no structure is found.

    Returns synthetic chapter entries so the rest of the pipeline still works.
    """
    num_chunks = max(1, math.ceil(total_pages / PAGES_PER_CHUNK))
    pages_per = math.ceil(total_pages / num_chunks)

    chapters: list[tuple[TocEntry, tuple[TocEntry, ...], int, int]] = []
    for i in range(num_chunks):
        start = i * pages_per + 1
        end = min((i + 1) * pages_per, total_pages)
        entry = TocEntry(level=1, title=f"Part {i + 1}", page=start)
        chapters.append((entry, (), start, end))

    logger.info("Created %d page-range chunks (%d pages each)", num_chunks, pages_per)
    return tuple(chapters)


# ── Internal helpers ─────────────────────────────────────────────────────────

# Pattern matching common chapter/module title formats:
# "Chapter 1", "CHAPTER 1:", "Learning Module 1: Title",
# "Module 3", "Unit 2", "Lesson 5", etc.
_CHAPTER_PATTERN = re.compile(
    r"^(?:chapter|(?:learning\s+)?module|unit|lesson)\s+\d+", re.IGNORECASE
)


def _is_chapter_entry(entry: TocEntry) -> bool:
    """Check if a TOC entry represents a chapter heading.

    Allows up to level 3 to handle books where chapters are nested under
    part headings (e.g., L1: Book Title → L2: PART ONE → L3: CHAPTER 1).
    """
    return entry.level <= 3 and bool(_CHAPTER_PATTERN.search(entry.title))


def _normalize_title(title: str) -> str:
    """Collapse whitespace and strip leading/trailing spaces from a title."""
    return re.sub(r"\s+", " ", title).strip()


def _collapse_spaced_letters(text: str) -> str:
    """Collapse spaced-letter text into normal words.

    Detects patterns like "L E A R N I N G  M O D U L E" where single
    characters are separated by spaces, and collapses them into "LEARNING MODULE".
    """
    if re.match(r"^([A-Z] ){2,}", text):
        words = text.split("  ")
        collapsed = " ".join(w.replace(" ", "") for w in words)
        return collapsed.strip()
    return text


# Back-matter / structural titles that should not be treated as chapters.
_BACK_MATTER_TITLES = frozenset({
    "summary",
    "references",
    "practice problems",
    "solutions",
    "glossary",
    "index",
    "appendix",
    "bibliography",
    "contents",
    "about the author",
    "about the authors",
    "disclaimer",
    "legal notice",
    "terms of use",
    "terms and conditions",
    "important notice",
    "important information",
    "important disclosures",
    "risk disclosures",
})

# Front-matter titles to skip — these appear before substantive content.
_FRONT_MATTER_TITLES = frozenset({
    "cover",
    "half title",
    "series page",
    "title page",
    "copyright",
    "dedication",
    "contents",
    "table of contents",
    "foreword",
    "preface",
    "acknowledgments",
    "acknowledgements",
    "about the author",
    "about the authors",
    "list of figures",
    "list of tables",
    "notation",
    "conventions",
    "how to use the cfa program curriculum",
    "disclaimer",
    "legal notice",
    "terms of use",
    "terms and conditions",
    "privacy policy",
    "important notice",
    "important information",
    "important disclosures",
    "general disclosures",
    "risk disclosures",
    "risk factors",
    "about this document",
    "about this publication",
    # Study guide / meta-content titles
    "how to use this book",
    "how to use this text",
    "how to use this guide",
    "other feedback",
    "errata",
    "feedback",
})


def _filter_front_matter(entries: tuple[TocEntry, ...]) -> tuple[TocEntry, ...]:
    """Remove front-matter entries that appear before the first real content.

    Keeps all entries from the first one whose title doesn't match known
    front-matter patterns.
    """
    first_content_idx: int | None = None
    for i, entry in enumerate(entries):
        title_lower = entry.title.lower().strip()
        if title_lower not in _FRONT_MATTER_TITLES and not _is_front_matter_title(title_lower):
            first_content_idx = i
            break

    if first_content_idx is None:
        # All entries are front matter — return empty
        logger.info("Filtered all %d TOC entries as front matter", len(entries))
        return ()

    filtered = entries[first_content_idx:]
    skipped = len(entries) - len(filtered)
    if skipped:
        logger.info("Filtered %d front-matter TOC entries", skipped)

    return filtered


def _is_front_matter_title(title_lower: str) -> bool:
    """Check if a title looks like front matter via substring patterns."""
    front_patterns = (
        "program curriculum", "volume", "level ii", "level i ",
        "disclaimer", "legal notice", "terms of use",
        "important notice", "risk disclos",
        "how to use", "study program", "study guide",
        "learning ecosystem", "other feedback",
        "about this curriculum", "about this program",
    )
    return any(p in title_lower for p in front_patterns)
