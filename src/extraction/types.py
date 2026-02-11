"""Data types for extracted PDF content.

These frozen dataclasses represent the structured content tree extracted from
a PDF document. The hierarchy is: Book → Chapter → Section, with ImageRef
and Table as leaf nodes attached to sections.

All collections use tuples for immutability. Paths are relative to the
extraction output directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ImageRef:
    """A reference to an extracted image file.

    Attributes:
        path: Relative path from the extraction output directory to the image file.
        page: 1-indexed page number where the image appears in the PDF.
        caption: Detected caption text, or empty string if none found.
        bbox: Bounding box as (x0, y0, x1, y1) in PDF coordinate space.
    """

    path: Path
    page: int
    caption: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class Table:
    """A table extracted from the PDF.

    Attributes:
        page: 1-indexed page number where the table appears.
        headers: Column header strings. Empty tuple if no headers detected.
        rows: Tuple of row tuples, each containing cell strings.
    """

    page: int
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Section:
    """A section of content within a chapter.

    Attributes:
        title: Section heading text.
        level: Nesting depth (1=module/part, 2=section, 3=subsection).
        start_page: 1-indexed first page of this section.
        end_page: 1-indexed last page of this section (inclusive).
        text: Full extracted text content of this section.
        images: Images found within this section's page range.
        tables: Tables found within this section's page range.
        subsections: Nested subsections within this section.
    """

    title: str
    level: int
    start_page: int
    end_page: int
    text: str
    images: tuple[ImageRef, ...] = field(default=())
    tables: tuple[Table, ...] = field(default=())
    subsections: tuple[Section, ...] = field(default=())


@dataclass(frozen=True)
class Chapter:
    """A chapter within the book.

    Attributes:
        chapter_number: Sequential chapter number (1-indexed).
        title: Chapter title text.
        start_page: 1-indexed first page of the chapter.
        end_page: 1-indexed last page of the chapter (inclusive).
        sections: Child sections within this chapter.
    """

    chapter_number: int
    title: str
    start_page: int
    end_page: int
    sections: tuple[Section, ...]


@dataclass(frozen=True)
class Book:
    """Complete structured representation of an extracted PDF book.

    Attributes:
        title: Book title from PDF metadata or TOC.
        author: Author name from PDF metadata.
        total_pages: Total number of pages in the PDF.
        chapters: Ordered sequence of chapters.
    """

    title: str
    author: str
    total_pages: int
    chapters: tuple[Chapter, ...]
