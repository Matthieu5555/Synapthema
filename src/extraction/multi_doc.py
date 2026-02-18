"""Multi-document extraction coordinator.

Orchestrates extraction across multiple input documents, calling
extract_book() for each and returning a list of Book objects.
Each book gets its own subdirectory under the shared extraction output.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import InputSource
from src.extraction.pdf_parser import extract_book
from src.extraction.types import Book

if TYPE_CHECKING:
    from src.transformation.llm_client import LLMClient

logger = logging.getLogger(__name__)


def extract_corpus(
    sources: list[InputSource],
    output_dir: Path,
    llm_client: LLMClient,
) -> list[Book]:
    """Extract all input documents, returning a list of Book objects.

    Each source gets its own subdirectory under output_dir for images
    and structure JSON. The existing extract_book() function is called
    per-document without modification.

    Args:
        sources: List of input documents to extract.
        output_dir: Shared extraction output directory.
        llm_client: LLM client for TOC detection fallback.

    Returns:
        List of Book objects, one per input source, in the same order
        as the input sources list.
    """
    books: list[Book] = []
    for idx, source in enumerate(sources):
        slug = source_slug(source, idx)
        book_output_dir = output_dir / slug
        logger.info(
            "Extracting document %d/%d: %s → %s",
            idx + 1, len(sources), source.path.name, book_output_dir,
        )
        book = extract_book(source.path, book_output_dir, llm_client)
        books.append(book)
        logger.info(
            "Document %d extracted: '%s' (%d chapters)",
            idx + 1, book.title, len(book.chapters),
        )

    logger.info(
        "Corpus extraction complete: %d documents, %d total chapters",
        len(books),
        sum(len(b.chapters) for b in books),
    )
    return books


def source_slug(source: InputSource, index: int) -> str:
    """Generate a unique subdirectory slug for a source document."""
    stem = source.path.stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-") or "unknown"
    return f"{index:02d}_{slug}"
