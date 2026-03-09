"""PDF content extraction — the deep module for Stage 1.

Single public entry point: extract_book(). Internally handles all the
complexity of text extraction, structure detection, image extraction,
table extraction, and JSON serialization. The caller passes a PDF path
and gets back a fully structured Book.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import fitz
import pdfplumber
import pymupdf4llm

# Optional: marker-pdf for math-aware extraction (requires PyTorch)
try:
    from marker.converters.pdf import PdfConverter as _MarkerPdfConverter
    from marker.models import create_model_dict as _marker_create_model_dict
    from marker.output import text_from_rendered as _marker_text_from_rendered

    _MARKER_AVAILABLE = True
except ImportError:
    _MARKER_AVAILABLE = False

if TYPE_CHECKING:
    from src.protocols import LLMClient

from src.extraction.structure_detector import (
    TocEntry,
    detect_subsections_with_llm,
    detect_toc_with_llm,
    extract_toc_entries,
    identify_chapters,
)
from src.extraction.types import Book, Chapter, ImageRef, Section, Table

logger = logging.getLogger(__name__)

# Minimum width and height (pixels) for an extracted image to be considered
# content-relevant. Images smaller than this are typically decorative elements,
# bullets, or separator lines. Increasing this value may exclude small but
# meaningful diagrams; decreasing it may include visual noise.
# Used by: _extract_all_images()
MIN_IMAGE_DIMENSION = 50

# Vertical search distance (PDF points) below an image bounding box to look
# for caption text. 60pt ≈ ~0.8 inches, enough for a 1-2 line caption.
# Increasing this may pick up unrelated body text; decreasing may miss
# multi-line captions.
# Used by: _detect_caption()
CAPTION_SEARCH_MARGIN = 60.0


def extract_book(pdf_path: Path, output_dir: Path, llm_client: LLMClient | None = None) -> Book:
    """Extract structured content from a PDF and persist to disk.

    This is the single entry point for Stage 1. It opens the PDF, detects
    structure, extracts text/images/tables per section, builds the Book
    dataclass, and writes both the JSON structure and extracted images to
    output_dir.

    Args:
        pdf_path: Path to the input PDF file.
        output_dir: Directory to write extraction output (JSON + images/).
        llm_client: Optional LLM client for TOC detection fallback.

    Returns:
        A fully populated Book dataclass representing the document.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    # Pre-extract full document text with marker if available (math-aware).
    # Returns a dict mapping 0-indexed page number → markdown text.
    marker_pages = _extract_with_marker(pdf_path, total_pages)

    title, author = _extract_metadata(doc)
    logger.info("Extracting '%s' by %s (%d pages)", title, author, total_pages)

    toc_entries = extract_toc_entries(doc)
    if not toc_entries and llm_client is not None:
        logger.info("No TOC found, using LLM to detect document structure")
        toc_entries = detect_toc_with_llm(doc, llm_client)
    if not toc_entries:
        logger.warning("No structure detected — will use page-range fallback")

    chapter_groups = identify_chapters(toc_entries, total_pages)

    # Pre-extract all images and tables for fast lookup by page
    all_images = _extract_all_images(doc, images_dir)
    all_tables = _extract_all_tables(pdf_path)

    chapters = tuple(
        _build_chapter(
            chapter_number=idx + 1,
            chapter_entry=chapter_entry,
            child_toc_entries=children,
            start_page=start,
            end_page=end,
            doc=doc,
            all_images=all_images,
            all_tables=all_tables,
            llm_client=llm_client,
            marker_pages=marker_pages,
        )
        for idx, (chapter_entry, children, start, end) in enumerate(chapter_groups)
    )

    doc.close()

    book = Book(
        title=title,
        author=author,
        total_pages=total_pages,
        chapters=chapters,
    )

    _write_json(book, output_dir / "book_structure.json")
    logger.info(
        "Extraction complete: %d chapters, %d total images, %d total tables",
        len(chapters),
        sum(len(img_list) for img_list in all_images.values()),
        sum(len(tbl_list) for tbl_list in all_tables.values()),
    )

    return book


# ── Internal: metadata ───────────────────────────────────────────────────────


def _extract_metadata(doc: fitz.Document) -> tuple[str, str]:
    """Pull title and author from PDF metadata, with fallbacks."""
    metadata: dict[str, str] = doc.metadata or {}  # pyright: ignore[reportAssignmentType] -- PyMuPDF untyped
    title = metadata.get("title", "").strip()
    author = metadata.get("author", "").strip()

    # PDF metadata titles can be verbose; clean up common patterns
    if not title:
        title = "Untitled Document"
    if not author:
        author = "Unknown Author"

    return title, author


# ── Internal: Unicode math → LaTeX normalization ────────────────────────────

# Maps isolated Unicode math symbols to LaTeX wrapped in $ delimiters.
# Applied after text extraction so the LLM sees LaTeX notation.
_UNICODE_TO_LATEX: dict[str, str] = {
    # Greek lowercase
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\epsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    # Greek uppercase
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma", "Φ": r"\Phi",
    "Ψ": r"\Psi", "Ω": r"\Omega",
    # Operators
    "∑": r"\sum", "∏": r"\prod", "∫": r"\int", "∂": r"\partial",
    "∇": r"\nabla", "√": r"\sqrt{}",
    # Relations
    "≤": r"\leq", "≥": r"\geq", "≠": r"\neq", "≈": r"\approx",
    "∈": r"\in", "∉": r"\notin", "⊂": r"\subset", "⊃": r"\supset",
    "⊆": r"\subseteq", "⊇": r"\supseteq", "∀": r"\forall", "∃": r"\exists",
    "≡": r"\equiv", "∝": r"\propto", "≪": r"\ll", "≫": r"\gg",
    # Arrows
    "→": r"\to", "←": r"\leftarrow", "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow", "⇐": r"\Leftarrow", "⇔": r"\Leftrightarrow",
    "↦": r"\mapsto",
    # Special
    "∞": r"\infty", "×": r"\times", "÷": r"\div", "±": r"\pm",
    "∓": r"\mp", "·": r"\cdot", "⊕": r"\oplus", "⊗": r"\otimes",
    "†": r"\dagger",
    # Set / logic
    "∅": r"\emptyset", "∪": r"\cup", "∩": r"\cap",
    "¬": r"\neg", "∧": r"\land", "∨": r"\lor",
    # Miscellaneous
    "ℝ": r"\mathbb{R}", "ℤ": r"\mathbb{Z}", "ℕ": r"\mathbb{N}",
    "ℂ": r"\mathbb{C}", "ℚ": r"\mathbb{Q}",
}

# Superscript and subscript digits/letters → LaTeX
_SUPERSCRIPTS: dict[str, str] = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "⁼": "=", "ⁿ": "n", "ⁱ": "i",
}
_SUBSCRIPTS: dict[str, str] = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "₊": "+", "₋": "-", "₌": "=", "ₙ": "n", "ᵢ": "i",
    "ⱼ": "j", "ₖ": "k",
}

# Precompiled regex: matches any Unicode math symbol from the map.
_UNICODE_MATH_RE = re.compile(
    "|".join(re.escape(ch) for ch in _UNICODE_TO_LATEX)
)
_SUPERSCRIPT_RE = re.compile(
    "([" + "".join(re.escape(ch) for ch in _SUPERSCRIPTS) + "]+)"
)
_SUBSCRIPT_RE = re.compile(
    "([" + "".join(re.escape(ch) for ch in _SUBSCRIPTS) + "]+)"
)

# Matches regions already inside $ delimiters (to skip them).
_MATH_SPAN_RE = re.compile(r"\$\$.*?\$\$|\$[^$]+?\$", re.DOTALL)


def _normalize_unicode_math(text: str) -> str:
    """Convert Unicode math symbols to LaTeX notation.

    Replaces isolated Unicode math characters (Greek letters, operators,
    relations, etc.) with their LaTeX equivalents wrapped in $ delimiters.
    Symbols already inside $...$ or $$...$$ are left untouched.
    """
    if not text:
        return text

    # Find all math spans so we can skip them
    protected: set[int] = set()
    for m in _MATH_SPAN_RE.finditer(text):
        protected.update(range(m.start(), m.end()))

    # Replace Unicode symbols with LaTeX
    result = list(text)
    # Process in reverse order to preserve indices during replacement
    replacements: list[tuple[int, int, str]] = []

    for m in _UNICODE_MATH_RE.finditer(text):
        if m.start() in protected:
            continue
        latex = _UNICODE_TO_LATEX[m.group()]
        replacements.append((m.start(), m.end(), f"${latex}$"))

    # Superscripts (group consecutive superscript chars)
    for m in _SUPERSCRIPT_RE.finditer(text):
        if m.start() in protected:
            continue
        digits = "".join(_SUPERSCRIPTS.get(ch, ch) for ch in m.group())
        replacements.append((m.start(), m.end(), f"$^{{{digits}}}$"))

    # Subscripts (group consecutive subscript chars)
    for m in _SUBSCRIPT_RE.finditer(text):
        if m.start() in protected:
            continue
        digits = "".join(_SUBSCRIPTS.get(ch, ch) for ch in m.group())
        replacements.append((m.start(), m.end(), f"$_{{{digits}}}$"))

    # Apply replacements in reverse order to preserve positions
    replacements.sort(key=lambda r: r[0], reverse=True)
    chars = list(text)
    for start, end, replacement in replacements:
        chars[start:end] = list(replacement)

    return "".join(chars)


# ── Internal: text extraction ────────────────────────────────────────────────


def _extract_with_marker(pdf_path: Path, total_pages: int) -> dict[int, str] | None:
    """Extract full document text using marker-pdf (math-aware).

    Returns a dict mapping 0-indexed page numbers to markdown text,
    or None if marker is not available or extraction fails.
    """
    if not _MARKER_AVAILABLE:
        return None

    try:
        logger.info("Using marker-pdf for math-aware text extraction")
        converter = _MarkerPdfConverter(
            artifact_dict=_marker_create_model_dict(),
        )
        rendered = converter(str(pdf_path))
        full_md, _, _ = _marker_text_from_rendered(rendered)

        if not full_md or not full_md.strip():
            logger.warning("marker-pdf returned empty text, falling back to pymupdf4llm")
            return None

        # marker returns a single markdown string. Split by page markers if
        # present, otherwise store as page 0 (whole-document).
        # marker inserts page break comments: <!-- Page N -->
        page_pattern = re.compile(r"<!--\s*Page\s+(\d+)\s*-->")
        pages: dict[int, str] = {}
        current_page = 0
        current_chunks: list[str] = []

        for line in full_md.split("\n"):
            m = page_pattern.match(line)
            if m:
                if current_chunks:
                    pages[current_page] = "\n".join(current_chunks)
                current_page = int(m.group(1)) - 1  # Convert to 0-indexed
                current_chunks = []
            else:
                current_chunks.append(line)

        if current_chunks:
            pages[current_page] = "\n".join(current_chunks)

        # If no page markers found, store all text under page 0
        if not pages:
            pages[0] = full_md

        logger.info(
            "marker-pdf extracted %d pages of text",
            len(pages),
        )
        return pages
    except Exception as exc:
        logger.warning("marker-pdf extraction failed, falling back to pymupdf4llm: %s", exc)
        return None


def _extract_text_for_page_range(
    doc: fitz.Document,
    start_page: int,
    end_page: int,
    marker_pages: dict[int, str] | None = None,
) -> str:
    """Extract and concatenate text from a range of pages.

    Priority: marker-pdf (math-aware) → pymupdf4llm (markdown) → plain PyMuPDF.
    Unicode math symbols are normalized to LaTeX on all paths.

    Args:
        doc: Open PyMuPDF document.
        start_page: 1-indexed first page (inclusive).
        end_page: 1-indexed last page (inclusive).
        marker_pages: Pre-extracted marker text by 0-indexed page, or None.

    Returns:
        Extracted text with Unicode math normalized to LaTeX.
    """
    pages = list(range(max(0, start_page - 1), min(end_page, len(doc))))
    if not pages:
        return ""

    # Primary path: marker-pdf (math-aware, pre-extracted)
    if marker_pages is not None:
        chunks = [marker_pages[p] for p in pages if p in marker_pages]
        if chunks:
            return _normalize_unicode_math("\n\n".join(chunks))
        # If marker has no page-level data but has full-doc text (page 0),
        # fall through to pymupdf4llm for page-level extraction.

    # Secondary path: pymupdf4llm markdown extraction
    try:
        md_text: str = pymupdf4llm.to_markdown(doc, pages=pages)
        if md_text.strip():
            return _normalize_unicode_math(md_text)
    except Exception:
        logger.warning("pymupdf4llm extraction failed, falling back to plain text")

    # Fallback: original plain text extraction
    pages_text: list[str] = []
    for page_idx in pages:
        page_text: str = doc[page_idx].get_text()  # pyright: ignore[reportAssignmentType]
        if page_text.strip():
            pages_text.append(page_text)

    return _normalize_unicode_math("\n\n".join(pages_text))


# ── Internal: image extraction ───────────────────────────────────────────────


def _extract_all_images(
    doc: fitz.Document, images_dir: Path
) -> dict[int, list[ImageRef]]:
    """Extract all images from the document, organized by 1-indexed page number.

    Saves each image as a PNG file in images_dir with a naming scheme of
    page{NNN}_img{NNN}.png. Returns a dict mapping page numbers to lists
    of ImageRef objects.
    """
    images_by_page: dict[int, list[ImageRef]] = {}
    image_counter = 0

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_number = page_idx + 1
        image_list = page.get_images(full=True)

        if not image_list:
            continue

        page_images: list[ImageRef] = []

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                pix = fitz.Pixmap(doc, xref)

                if pix.width < MIN_IMAGE_DIMENSION or pix.height < MIN_IMAGE_DIMENSION:
                    continue

                # Convert non-RGB (e.g. CMYK, grayscale+alpha) to RGB
                if pix.n - pix.alpha != 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                image_counter += 1
                filename = f"page{page_number:03d}_img{image_counter:03d}.png"
                image_path = images_dir / filename
                pix.save(str(image_path))

                # Get image position on page via the image's bbox
                bbox = _get_image_bbox(page, xref)

                caption = _detect_caption(page, bbox, doc)

                page_images.append(
                    ImageRef(
                        path=Path("images") / filename,
                        page=page_number,
                        caption=caption,
                        bbox=bbox,
                    )
                )
            except (ValueError, RuntimeError, OSError) as exc:
                logger.warning(
                    "Failed to extract image xref=%d on page %d: %s",
                    xref, page_number, exc,
                )

        if page_images:
            images_by_page[page_number] = page_images

    logger.info("Extracted %d images across %d pages", image_counter, len(images_by_page))
    return images_by_page


def _get_image_bbox(
    page: fitz.Page, xref: int
) -> tuple[float, float, float, float]:
    """Find the bounding box of an image on a page by its xref.

    Falls back to full-page bbox if the image reference isn't found in
    the page's display list.
    """
    for img_block in page.get_image_info(xrefs=True):
        if img_block.get("xref") == xref:
            bbox = img_block["bbox"]
            return (bbox[0], bbox[1], bbox[2], bbox[3])

    # Fallback: use the full page rect
    rect = page.rect
    return (rect.x0, rect.y0, rect.x1, rect.y1)


def _detect_caption(
    page: fitz.Page,
    image_bbox: tuple[float, float, float, float],
    doc: fitz.Document,
) -> str:
    """Attempt to detect a figure caption near an image.

    Looks for text blocks immediately below the image that start with
    common caption patterns like "Figure", "Fig.", "Exhibit", "Table".
    Searches within a vertical margin below the image bounding box.
    """
    _, _, x1, y1 = image_bbox
    x0_img = image_bbox[0]

    # Search area: full page width, starting just below image
    search_rect = fitz.Rect(x0_img - 20, y1, x1 + 20, y1 + CAPTION_SEARCH_MARGIN)

    blocks = page.get_text("blocks")
    for block in blocks:
        block_rect = fitz.Rect(block[:4])
        if search_rect.intersects(block_rect):
            text = block[4].strip()
            if _looks_like_caption(text):
                return text

    return ""


def _looks_like_caption(text: str) -> bool:
    """Check if text looks like a figure/table caption."""
    lower = text.lower().lstrip()
    return any(
        lower.startswith(prefix)
        for prefix in ("figure", "fig.", "fig ", "exhibit", "table", "chart", "graph")
    )


# ── Internal: table extraction ───────────────────────────────────────────────


def _extract_all_tables(pdf_path: Path) -> dict[int, list[Table]]:
    """Extract all tables from the PDF using pdfplumber, organized by page.

    pdfplumber is more reliable than PyMuPDF for table detection. Returns
    a dict mapping 1-indexed page numbers to lists of Table objects.
    """
    tables_by_page: dict[int, list[Table]] = {}
    table_count = 0

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_number = page_idx + 1
                page_tables = page.extract_tables()

                if not page_tables:
                    continue

                parsed_tables: list[Table] = []

                for raw_table in page_tables:
                    if not raw_table or len(raw_table) < 2:
                        continue

                    # First row as headers, rest as data
                    headers = tuple(
                        cell.strip() if cell else "" for cell in raw_table[0]
                    )
                    rows = tuple(
                        tuple(cell.strip() if cell else "" for cell in row)
                        for row in raw_table[1:]
                    )

                    table_count += 1
                    parsed_tables.append(
                        Table(page=page_number, headers=headers, rows=rows)
                    )

                if parsed_tables:
                    tables_by_page[page_number] = parsed_tables

    except (OSError, TypeError, ValueError, KeyError) as exc:
        logger.warning("Table extraction failed, continuing without tables: %s", exc)

    logger.info("Extracted %d tables across %d pages", table_count, len(tables_by_page))
    return tables_by_page


# ── Internal: chapter/section building ───────────────────────────────────────


def _build_chapter(
    chapter_number: int,
    chapter_entry: TocEntry,
    child_toc_entries: tuple[TocEntry, ...],
    start_page: int,
    end_page: int,
    doc: fitz.Document,
    all_images: dict[int, list[ImageRef]],
    all_tables: dict[int, list[Table]],
    llm_client: LLMClient | None = None,
    marker_pages: dict[int, str] | None = None,
) -> Chapter:
    """Construct a Chapter dataclass from a TOC entry and its children.

    Builds Section objects for each child TOC entry, assigning text, images,
    and tables based on page ranges. When no child TOC entries exist and an
    LLM client is available, uses the LLM to detect subsection structure.
    """
    if not child_toc_entries:
        # Try LLM-based subsection detection when no TOC children exist
        detected_entries: tuple[TocEntry, ...] = ()
        if llm_client is not None:
            detected_entries = detect_subsections_with_llm(
                doc, start_page, end_page, chapter_entry.title, llm_client,
            )

        if detected_entries:
            logger.info(
                "Chapter %d '%s': LLM detected %d subsections",
                chapter_number, chapter_entry.title, len(detected_entries),
            )
            sections = _build_sections(
                detected_entries, end_page, doc, all_images, all_tables,
                marker_pages=marker_pages,
            )
        else:
            # Genuine single-section chapter: treat entire chapter as one section
            text = _extract_text_for_page_range(
                doc, start_page, end_page, marker_pages=marker_pages,
            )
            images = _collect_items_for_page_range(all_images, start_page, end_page)
            tables = _collect_items_for_page_range(all_tables, start_page, end_page)

            sections = (
                Section(
                    title=chapter_entry.title,
                    level=chapter_entry.level,
                    start_page=start_page,
                    end_page=end_page,
                    text=text,
                    images=tuple(images),
                    tables=tuple(tables),
                ),
            )
    else:
        sections = _build_sections(
            child_toc_entries, end_page, doc, all_images, all_tables,
            marker_pages=marker_pages,
        )

    return Chapter(
        chapter_number=chapter_number,
        title=chapter_entry.title,
        start_page=start_page,
        end_page=end_page,
        sections=sections,
    )


def _build_sections(
    toc_entries: tuple[TocEntry, ...],
    chapter_end: int,
    doc: fitz.Document,
    all_images: dict[int, list[ImageRef]],
    all_tables: dict[int, list[Table]],
    marker_pages: dict[int, str] | None = None,
) -> tuple[Section, ...]:
    """Build Section dataclasses from TOC entries within a chapter.

    Groups entries hierarchically: L2 entries become sections, and any L3+
    entries between consecutive L2 entries become subsections of the preceding
    L2 section. This preserves the document's structural hierarchy.
    """
    if not toc_entries:
        return ()

    # Find the minimum level to identify "top-level" sections within this chapter
    min_level = min(e.level for e in toc_entries)

    # Group: top-level entries with their children
    groups: list[tuple[TocEntry, list[TocEntry]]] = []
    for entry in toc_entries:
        if entry.level == min_level:
            groups.append((entry, []))
        elif groups:
            groups[-1][1].append(entry)

    sections: list[Section] = []

    for group_idx, (parent_entry, children) in enumerate(groups):
        # Determine page range for this section
        if group_idx + 1 < len(groups):
            section_end = groups[group_idx + 1][0].page - 1
        else:
            section_end = chapter_end
        section_end = max(parent_entry.page, section_end)

        text = _extract_text_for_page_range(
            doc, parent_entry.page, section_end, marker_pages=marker_pages,
        )
        images = _collect_items_for_page_range(all_images, parent_entry.page, section_end)
        tables = _collect_items_for_page_range(all_tables, parent_entry.page, section_end)

        # Build subsections from children
        subsections: list[Section] = []
        for child_idx, child_entry in enumerate(children):
            if child_idx + 1 < len(children):
                child_end = children[child_idx + 1].page - 1
            else:
                child_end = section_end
            child_end = max(child_entry.page, child_end)

            child_text = _extract_text_for_page_range(
                doc, child_entry.page, child_end, marker_pages=marker_pages,
            )
            child_images = _collect_items_for_page_range(all_images, child_entry.page, child_end)
            child_tables = _collect_items_for_page_range(all_tables, child_entry.page, child_end)

            subsections.append(
                Section(
                    title=child_entry.title,
                    level=child_entry.level,
                    start_page=child_entry.page,
                    end_page=child_end,
                    text=child_text,
                    images=tuple(child_images),
                    tables=tuple(child_tables),
                )
            )

        sections.append(
            Section(
                title=parent_entry.title,
                level=parent_entry.level,
                start_page=parent_entry.page,
                end_page=section_end,
                text=text,
                images=tuple(images),
                tables=tuple(tables),
                subsections=tuple(subsections),
            )
        )

    return tuple(sections)


_T = TypeVar("_T")


def _collect_items_for_page_range(
    items_by_page: dict[int, list[_T]], start_page: int, end_page: int
) -> list[_T]:
    """Collect all items (images or tables) that fall within a page range."""
    collected: list[_T] = []
    for page_num in range(start_page, end_page + 1):
        if page_num in items_by_page:
            collected.extend(items_by_page[page_num])
    return collected


# ── Internal: JSON serialization ─────────────────────────────────────────────


def _write_json(book: Book, output_path: Path) -> None:
    """Serialize the Book dataclass to a JSON file.

    Converts Path objects to strings for JSON compatibility.
    """

    def _serialize(obj: object) -> object:
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    data = asdict(book)
    output_path.write_text(
        json.dumps(data, indent=2, default=_serialize, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote book structure to %s", output_path)
