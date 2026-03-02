"""Deep chapter reading — the explorer agent for Stage 1.25.

Dedicated LLM pass per chapter that reads the full text and produces
a structured ChapterAnalysis via Instructor. This is the "note-taking"
stage: the LLM acts as a pedagogical analyst, not a content creator.

Public entry points:
- analyze_chapter(): Single chapter analysis.
- analyze_book(): Parallel analysis of all chapters in a book.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from src.extraction.types import Book, Chapter
from src.transformation.analysis_types import ChapterAnalysis
from src.transformation.content_pre_analyzer import (
    ChapterSignals,
    analyze_chapter_sections,
)

if TYPE_CHECKING:
    from src.transformation.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Maximum characters of chapter text to send to the LLM.
# ~20K chars ≈ ~5K tokens, leaves room for the response.
MAX_CHAPTER_TEXT_LENGTH = 20_000


# ── System prompt ────────────────────────────────────────────────────────────

_DEEP_READER_SYSTEM_PROMPT = """\
You are a **pedagogical analyst** — NOT a content creator. Your job is to
deeply read a chapter from a textbook and produce structured notes that a
curriculum planner will use to design an effective learning experience.

Your analysis must include:

1. **Concept inventory**: Every concept taught in the chapter.
   - Name, one-sentence definition, type (definition/formula/process/\
comparison/principle/example/theorem/heuristic).
   - Which section it appears in.
   - Key terms and formulas associated with it.
   - Importance: core (central to the chapter), supporting (helps understand \
core concepts), or peripheral (mentioned but not deeply taught).

2. **Prerequisite links**: Dependencies between concepts.
   - Which concept requires or builds on which other concept.
   - Both within this chapter AND from prior chapters (use the provided list \
of prior chapter concepts).

3. **Section characterizations**: For each section:
   - What type of content dominates (conceptual/procedural/comparative/\
theoretical/applied/mixed).
   - Boolean flags: has_formulas, has_procedures, has_comparisons, \
has_definitions, has_examples.
   - Difficulty estimate (introductory/intermediate/advanced).
   - 2-3 sentence summary of what the section teaches.

4. **Logical flow**: How the chapter's sections connect — what leads to what, \
where the narrative builds.

5. **Core learning outcome**: The single most important thing a learner should \
take away from this chapter.

6. **External prerequisites**: Concepts this chapter assumes the reader already \
knows from prior chapters.

7. **Difficulty progression**: How difficulty changes across the chapter \
(e.g., "starts introductory, builds to advanced by the end").

Be thorough. Miss nothing. Every concept matters."""


# ── Public API ───────────────────────────────────────────────────────────────


def analyze_chapter(
    chapter: Chapter,
    client: LLMClient,
    chapter_signals: ChapterSignals | None = None,
    prior_chapter_concepts: list[str] | None = None,
) -> ChapterAnalysis:
    """Deeply read a single chapter and produce structured analysis.

    Args:
        chapter: A Chapter from Stage 1 extraction.
        client: LLM client with complete_structured().
        chapter_signals: Optional pre-computed regex signals (used in prompt).
        prior_chapter_concepts: Concept names from previously analyzed chapters.

    Returns:
        ChapterAnalysis with concepts, prerequisites, characterizations.

    Raises:
        LLMError: If the LLM call fails.
    """
    if chapter_signals is None:
        chapter_signals = _compute_signals(chapter)

    chapter_text = _build_chapter_text(chapter)
    user_prompt = _build_user_prompt(
        chapter, chapter_text, chapter_signals, prior_chapter_concepts or [],
    )

    analysis = client.complete_structured(
        _DEEP_READER_SYSTEM_PROMPT,
        user_prompt,
        ChapterAnalysis,
    )
    # Ensure chapter metadata matches
    analysis.chapter_number = chapter.chapter_number
    analysis.chapter_title = chapter.title
    logger.info(
        "Deep read chapter %d '%s': %d concepts, %d prerequisites, %d sections",
        chapter.chapter_number,
        chapter.title,
        len(analysis.concepts),
        len(analysis.prerequisites),
        len(analysis.section_characterizations),
    )
    return analysis


def analyze_book(
    book: Book,
    client: LLMClient,
    max_workers: int = 4,
) -> list[ChapterAnalysis]:
    """Analyze all chapters of a book in parallel.

    Chapters are analyzed concurrently using a thread pool. Cross-chapter
    prerequisite detection is handled downstream by the concept consolidator
    (Stage 1.3) rather than by threading prior concepts through sequential
    analysis — this trades a minor reduction in within-analysis prerequisite
    annotations for a significant speedup (N sequential calls → N/max_workers
    batches).

    Args:
        book: A Book from Stage 1 extraction.
        client: LLM client with complete_structured().
        max_workers: Maximum parallel LLM calls. Set to 1 for sequential.

    Returns:
        List of ChapterAnalysis, one per chapter, in order.
    """
    total = len(book.chapters)

    def _analyze_one(chapter: Chapter) -> ChapterAnalysis:
        logger.info(
            "Deep reading chapter %d/%d: '%s'",
            chapter.chapter_number, total, chapter.title,
        )
        signals = _compute_signals(chapter)
        return analyze_chapter(chapter, client, signals, prior_chapter_concepts=None)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(_analyze_one, ch): i
            for i, ch in enumerate(book.chapters)
        }
        results: list[ChapterAnalysis | None] = [None] * total
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                chapter = book.chapters[idx]
                logger.error(
                    "Failed to analyze chapter %d '%s' — skipping",
                    chapter.chapter_number, chapter.title,
                    exc_info=True,
                )
                # results[idx] stays None, filtered out below

    analyses = [a for a in results if a is not None]

    logger.info(
        "Deep reading complete: %d chapters, %d total concepts",
        len(analyses),
        sum(len(a.concepts) for a in analyses),
    )
    return analyses


# ── Internal helpers ─────────────────────────────────────────────────────────


def _compute_signals(chapter: Chapter) -> ChapterSignals:
    """Run regex pre-analysis on a chapter's sections."""
    sections = [
        (section.title, section.text) for section in chapter.sections
    ]
    return analyze_chapter_sections(chapter.title, sections)


def _build_chapter_text(chapter: Chapter) -> str:
    """Concatenate section texts (including subsections) with headers, smartly truncated."""
    parts: list[str] = []
    for section in chapter.sections:
        parts.append(f"## {section.title}\n\n{section.text}")
        for sub in section.subsections:
            parts.append(f"### {sub.title}\n\n{sub.text}")

    full_text = "\n\n---\n\n".join(parts)
    return _smart_truncate(full_text, MAX_CHAPTER_TEXT_LENGTH)


def _smart_truncate(text: str, max_length: int) -> str:
    """Truncate text at a paragraph boundary near max_length."""
    if len(text) <= max_length:
        return text

    # Try to cut at a paragraph break
    truncated = text[:max_length]
    last_break = truncated.rfind("\n\n")
    if last_break > max_length * 0.7:
        truncated = truncated[:last_break]

    return truncated + "\n\n[... content truncated for length ...]"


def _build_user_prompt(
    chapter: Chapter,
    chapter_text: str,
    signals: ChapterSignals,
    prior_concepts: list[str],
) -> str:
    """Build the user prompt with chapter text, signals, and prior concepts."""
    parts: list[str] = []

    parts.append(
        f"# Chapter {chapter.chapter_number}: {chapter.title}\n"
        f"Pages {chapter.start_page}-{chapter.end_page} "
        f"({len(chapter.sections)} sections)\n"
    )

    # Prior concepts context
    if prior_concepts:
        parts.append(
            "## Concepts from prior chapters (the learner already knows these):\n"
            + ", ".join(prior_concepts[:50])  # Cap at 50 to avoid prompt bloat
            + "\n"
        )

    # Pre-analysis signals to help the LLM focus
    if signals.sections:
        signal_lines = ["## Pre-analysis signals (from regex scan):"]
        for s in signals.sections:
            flags = []
            if s.formula_count:
                flags.append(f"formulas={s.formula_count}")
            if s.procedural_count:
                flags.append(f"procedures={s.procedural_count}")
            if s.comparison_count:
                flags.append(f"comparisons={s.comparison_count}")
            if s.definition_count:
                flags.append(f"definitions={s.definition_count}")
            if s.example_count:
                flags.append(f"examples={s.example_count}")
            if s.key_terms:
                flags.append(f"key_terms=[{', '.join(s.key_terms[:10])}]")
            if flags:
                signal_lines.append(f"- {s.section_title}: {'; '.join(flags)}")
        if len(signal_lines) > 1:
            parts.append("\n".join(signal_lines) + "\n")

    parts.append(
        "## Full chapter text:\n\n" + chapter_text
    )

    return "\n\n".join(parts)


