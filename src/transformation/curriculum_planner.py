"""Curriculum planner — Stage 1.5 between extraction and content generation.

Single public entry point: plan_curriculum(). Analyzes extracted Book content
and designs a learning path (CurriculumBlueprint) that the content designer
follows instead of blindly rotating templates.

For well-structured documents: preserves existing structure, adds learning
objectives and content-aware template assignments.

For unstructured notes: imposes logical topic grouping and learning progression.
"""

from __future__ import annotations

import logging
import re

from src.extraction.types import Book, Chapter, Section
from src.transformation.analysis_types import ChapterAnalysis, ConceptGraph
from src.transformation.content_pre_analyzer import (
    DOCUMENT_TYPE_TEMPLATE_WEIGHTS,
    DocumentType,
    format_document_type_guidance,
)
from src.transformation.llm_client import LLMClient, LLMError
from src.transformation.types import (
    CurriculumBlueprint,
    ModuleBlueprint,
    SectionBlueprint,
)

logger = logging.getLogger(__name__)

# Maximum characters for the content summary sent to the LLM.
_MAX_SUMMARY_LENGTH = 10_000

# Maximum characters of section text snippet included in the summary.
_SECTION_SNIPPET_LENGTH = 200

_PLANNER_SYSTEM_PROMPT = """\
You are a curriculum architect. You analyze extracted book/document content \
and design an optimal learning path.

Your input is a content summary: chapter titles, section titles, page ranges, \
text snippets, and document stats. When deep reading analysis is available, \
you also receive a concept inventory, dependency graph, and section \
characterizations. Your output is a CurriculumBlueprint JSON.

## CURRICULUM DESIGN RULES

1. **PRESERVE good structure** — if the source has clear chapters and sections, \
keep them. Don't reorganize well-organized material.
2. **REORGANIZE bad structure** — if topics are scattered or the document has no \
clear chapters, group related topics logically and create a learning progression.
3. **SCAFFOLD learning** — order topics from foundational → advanced within each module. \
When a concept dependency graph is provided, FOLLOW it: foundation concepts first, \
then concepts that build on them, then advanced concepts last.
4. **Map back to source** — every section in the blueprint must set \
`source_section_title` to the EXACT title of the extracted section it draws from. \
This is how the content generator finds the text.
5. **Assign templates based on CONTENT TYPE**, not rotation. When section \
characterizations are provided, use the dominant_content_type to guide your choice:
   - conceptual / has_definitions → "analogy_first" or "narrative"
   - theoretical / has_formulas → "worked_example"
   - comparative / has_comparisons → "compare_contrast"
   - procedural / has_procedures → "visual_walkthrough" or "problem_first"
   - applied / has_examples → "vignette" or "problem_first"
   - mixed → "socratic" or "analogy_first"
   - Section summaries with 5+ interrelated concepts → "visual_summary" (includes concept_map)
   - Common mistakes → "error_identification"
6. **Set bloom_target** based on content analysis. When a concept dependency \
graph is provided, use concept position:
   - Foundation concepts (no prerequisites) → "remember" or "understand"
   - Concepts that build on 1-2 others → "understand" or "apply"
   - Concepts that build on 3+ others → "apply" or "analyze"
   - Advanced concepts (most prerequisites) → "analyze" or "evaluate"
   When section characterizations are provided, also consider difficulty_estimate:
   - introductory → "remember" or "understand"
   - intermediate → "apply" or "analyze"
   - advanced → "analyze" or "evaluate"
7. **Add 1-3 learning objectives** per section — measurable, verb-based. \
When concept names are provided, reference them by name \
(e.g., "Explain the Sharpe ratio and how it differs from the Sortino ratio").
8. **Identify prerequisites** — if section B requires understanding section A, \
list section A's title in B's prerequisites. When a concept dependency graph \
is provided, USE IT to determine prerequisites rather than guessing.
9. **source_chapter_number** must match the chapter number from the extracted content.
10. **COVER ALL CHAPTERS** — you MUST create exactly one module per chapter in the \
source material. Every chapter listed in the content summary must appear as a module \
in your output. Do NOT skip or omit any chapters.

## OUTPUT FORMAT

Return ONLY a JSON object matching this schema (no markdown fences):

{
  "course_title": "...",
  "course_summary": "2-3 sentence overview",
  "learner_journey": "Module A → Module B → ...",
  "modules": [
    {
      "title": "...",
      "source_chapter_number": 1,
      "summary": "...",
      "sections": [
        {
          "title": "...",
          "source_section_title": "exact title from extracted content",
          "learning_objectives": ["Explain ...", "Calculate ..."],
          "template": "analogy_first",
          "bloom_target": "understand",
          "prerequisites": [],
          "rationale": "This section introduces key terminology, so analogy_first helps build intuition."
        }
      ]
    }
  ]
}"""


def plan_curriculum(
    book: Book,
    client: LLMClient,
    chapter_analyses: list[ChapterAnalysis] | None = None,
    concept_graph: ConceptGraph | None = None,
    document_type: DocumentType = "mixed",
) -> CurriculumBlueprint:
    """Analyze extracted content and design a curriculum blueprint.

    Sends a compact content summary to the LLM, which returns a structured
    blueprint with template assignments, learning objectives, and Bloom's
    targets per section.

    When deep reading analysis is available, builds a rich content summary
    with concept inventories, dependency order, and section characterizations
    instead of 200-character text snippets.

    Args:
        book: Extracted Book from Stage 1.
        client: LLM client for the planning call.
        chapter_analyses: Optional deep reading analyses per chapter.
        concept_graph: Optional consolidated concept dependency graph.
        document_type: Detected or configured document type (default "mixed").

    Returns:
        A CurriculumBlueprint guiding the content designer.

    Raises:
        LLMError: If the LLM call fails.
    """
    if chapter_analyses:
        summary = _build_rich_content_summary(book, chapter_analyses, concept_graph)
    else:
        summary = _build_content_summary(book)

    logger.info(
        "Planning curriculum for '%s' (%d chapters, summary: %d chars%s, type=%s)",
        book.title,
        len(book.chapters),
        len(summary),
        ", with analysis" if chapter_analyses else "",
        document_type,
    )

    # Build system prompt with document profile section
    system_prompt = _PLANNER_SYSTEM_PROMPT
    if document_type != "mixed":
        guidance = format_document_type_guidance(document_type)
        system_prompt += (
            f"\n\n### Document Profile\n"
            f"This is a **{document_type}** document. "
            f"When assigning templates, weight toward:\n{guidance}\n"
            f"Section-level content_type overrides this global prior when they conflict."
        )

    user_prompt = (
        f"Design a curriculum for the following extracted content.\n\n"
        f"{summary}"
    )

    blueprint = client.complete_structured(
        system_prompt, user_prompt, CurriculumBlueprint
    )
    logger.info(
        "Curriculum planned: %d modules, %d total sections",
        len(blueprint.modules),
        sum(len(m.sections) for m in blueprint.modules),
    )

    # Fill in passthrough modules for any chapters the LLM skipped
    blueprint = _ensure_all_chapters_covered(
        blueprint, book, chapter_analyses, document_type,
    )

    # Validate Bloom's progression
    progression_warnings = validate_progression(blueprint)
    for warning in progression_warnings:
        logger.warning("Progression issue: %s", warning)

    return blueprint


# ── Multi-document curriculum planning ────────────────────────────────────────

_MULTI_DOC_PLANNER_SYSTEM_PROMPT = """\
You are a curriculum architect. You analyze content from MULTIPLE source \
documents and design a single unified learning path.

Your input is a content summary covering N documents: their chapter titles, \
section titles, page ranges, text snippets, and document stats. \
Your output is a single CurriculumBlueprint JSON that interleaves content \
from all documents into one coherent course.

## MULTI-DOCUMENT CURRICULUM DESIGN RULES

1. **IDENTIFY ALL TOPICS** across all documents — build a complete topic inventory.
2. **DETECT OVERLAP** — if the same concept appears in multiple documents, \
pick the BEST explanation and reference supplementary material from others.
3. **DETECT PREREQUISITES** — for each topic, determine what must be understood \
first. Build a DAG of prerequisites, then linearize it.
4. **DIFFICULTY RAMP** — the first modules should cover foundational vocabulary \
and concepts. Middle modules should build on these with applications. Final \
modules should require synthesis across topics.
5. **SOURCE ATTRIBUTION** — every module must set `source_book_index` to the \
index of the book it primarily draws from (0-based). Every section must set \
`source_section_title` to the EXACT title of the extracted section.
6. **source_chapter_number** must match the chapter number from the source book.
7. **Assign templates based on CONTENT TYPE**, not rotation:
   - Definitions/concepts → "analogy_first" or "narrative"
   - Formulas/calculations → "worked_example"
   - Multiple related items → "compare_contrast"
   - Problem-solving → "problem_first"
   - Theory/principles → "socratic"
   - Processes/workflows → "visual_walkthrough"
   - Common mistakes → "error_identification"
   - Real-world applications → "vignette"
8. **Set bloom_target** based on what the content demands:
   - Terminology/definitions → "remember"
   - Explanations/concepts → "understand"
   - Procedures/calculations → "apply"
   - Comparisons/trade-offs → "analyze"
   - Judgment/evaluation → "evaluate"
9. **Add 1-3 learning objectives** per section — measurable, verb-based.
10. **Identify prerequisites** — list section titles that should be covered first.

## OUTPUT FORMAT

Return ONLY a JSON object matching this schema (no markdown fences):

{
  "course_title": "Unified course title",
  "course_summary": "2-3 sentence overview covering all source documents",
  "learner_journey": "Module A → Module B → ...",
  "modules": [
    {
      "title": "...",
      "source_chapter_number": 1,
      "source_book_index": 0,
      "summary": "...",
      "sections": [
        {
          "title": "...",
          "source_section_title": "exact title from extracted content",
          "source_book_index": 0,
          "learning_objectives": ["Explain ...", "Calculate ..."],
          "template": "analogy_first",
          "bloom_target": "understand",
          "prerequisites": [],
          "rationale": "..."
        }
      ]
    }
  ]
}"""

# Maximum characters for multi-document content summary.
_MAX_MULTI_DOC_SUMMARY_LENGTH = 20_000


def plan_multi_document_curriculum(
    books: list[Book],
    client: LLMClient,
    chapter_analyses_per_book: list[list[ChapterAnalysis]] | None = None,
    concept_graph: ConceptGraph | None = None,
    document_type: DocumentType = "mixed",
) -> CurriculumBlueprint:
    """Analyze multiple books and design a unified curriculum blueprint.

    Builds a combined content summary across all books, sends it to the LLM
    for cross-document analysis, and returns a single CurriculumBlueprint
    that may interleave sections from different books.

    When deep reading analysis is available, uses rich summaries with
    concept inventories and dependency graphs.

    Args:
        books: List of extracted Books from Stage 1.
        client: LLM client for the planning call.
        chapter_analyses_per_book: Optional analyses per book (list of lists).
        concept_graph: Optional consolidated concept dependency graph.
        document_type: Detected or configured document type (default "mixed").

    Returns:
        A CurriculumBlueprint with source_book_index on each module.

    Raises:
        LLMError: If the LLM call fails.
    """
    if chapter_analyses_per_book:
        summary = _build_rich_multi_doc_content_summary(
            books, chapter_analyses_per_book, concept_graph,
        )
    else:
        summary = _build_multi_doc_content_summary(books)
    total_chapters = sum(len(b.chapters) for b in books)
    logger.info(
        "Planning multi-document curriculum for %d books (%d total chapters, summary: %d chars, type=%s)",
        len(books), total_chapters, len(summary), document_type,
    )

    # Build system prompt with document profile section
    system_prompt = _MULTI_DOC_PLANNER_SYSTEM_PROMPT
    if document_type != "mixed":
        guidance = format_document_type_guidance(document_type)
        system_prompt += (
            f"\n\n### Document Profile\n"
            f"This is a **{document_type}** document. "
            f"When assigning templates, weight toward:\n{guidance}\n"
            f"Section-level content_type overrides this global prior when they conflict."
        )

    user_prompt = (
        f"Design a unified curriculum from the following {len(books)} documents.\n\n"
        f"{summary}"
    )

    blueprint = client.complete_structured(
        system_prompt, user_prompt, CurriculumBlueprint
    )
    logger.info(
        "Multi-doc curriculum planned: %d modules, %d total sections",
        len(blueprint.modules),
        sum(len(m.sections) for m in blueprint.modules),
    )

    progression_warnings = validate_progression(blueprint)
    for warning in progression_warnings:
        logger.warning("Progression issue: %s", warning)

    return blueprint


def _build_multi_doc_content_summary(books: list[Book]) -> str:
    """Build a combined content summary across multiple books."""
    parts: list[str] = [
        f"# Corpus: {len(books)} documents",
        "",
    ]

    for book_idx, book in enumerate(books):
        parts.append("---")
        parts.append(f"# Document {book_idx} (book_index={book_idx}): {book.title}")
        parts.append(f"Author: {book.author}")
        parts.append(f"Total pages: {book.total_pages}")
        parts.append(f"Chapters: {len(book.chapters)}")
        parts.append("")

        for chapter in book.chapters:
            section_count = len(chapter.sections)
            parts.append(
                f"## Chapter {chapter.chapter_number}: {chapter.title} "
                f"(pp. {chapter.start_page}-{chapter.end_page}, {section_count} sections)"
            )

            for section in chapter.sections:
                text_len = len(section.text.strip())
                snippet = section.text.strip()[:_SECTION_SNIPPET_LENGTH]
                if text_len > _SECTION_SNIPPET_LENGTH:
                    snippet += "..."

                extras = []
                if section.images:
                    extras.append(f"{len(section.images)} images")
                if section.tables:
                    extras.append(f"{len(section.tables)} tables")
                extras_str = f" [{', '.join(extras)}]" if extras else ""

                parts.append(
                    f"  - {section.title} (pp. {section.start_page}-{section.end_page}, "
                    f"{text_len} chars{extras_str})"
                )
                if snippet:
                    parts.append(f"    > {snippet}")

                for sub in section.subsections:
                    parts.append(
                        f"    - {sub.title} (pp. {sub.start_page}-{sub.end_page}, "
                        f"{len(sub.text.strip())} chars)"
                    )

            parts.append("")

    result = "\n".join(parts)

    if len(result) > _MAX_MULTI_DOC_SUMMARY_LENGTH:
        result = result[:_MAX_MULTI_DOC_SUMMARY_LENGTH] + "\n\n[... truncated ...]"

    return result


# ── Single-document content summary ──────────────────────────────────────────


def _build_content_summary(book: Book) -> str:
    """Build a compact content summary for the planner LLM.

    Includes chapter/section titles, page ranges, text length stats,
    and short text snippets to give the LLM topic context.
    """
    parts: list[str] = [
        f"# Document: {book.title}",
        f"Author: {book.author}",
        f"Total pages: {book.total_pages}",
        f"Chapters: {len(book.chapters)}",
        "",
    ]

    for chapter in book.chapters:
        section_count = len(chapter.sections)
        parts.append(
            f"## Chapter {chapter.chapter_number}: {chapter.title} "
            f"(pp. {chapter.start_page}-{chapter.end_page}, {section_count} sections)"
        )

        for section in chapter.sections:
            text_len = len(section.text.strip())
            snippet = section.text.strip()[:_SECTION_SNIPPET_LENGTH]
            if len(section.text.strip()) > _SECTION_SNIPPET_LENGTH:
                snippet += "..."

            extras = []
            if section.images:
                extras.append(f"{len(section.images)} images")
            if section.tables:
                extras.append(f"{len(section.tables)} tables")
            extras_str = f" [{', '.join(extras)}]" if extras else ""

            parts.append(
                f"  - {section.title} (pp. {section.start_page}-{section.end_page}, "
                f"{text_len} chars{extras_str})"
            )
            if snippet:
                parts.append(f"    > {snippet}")

            # Include subsections
            for sub in section.subsections:
                parts.append(
                    f"    - {sub.title} (pp. {sub.start_page}-{sub.end_page}, "
                    f"{len(sub.text.strip())} chars)"
                )

        parts.append("")

    result = "\n".join(parts)

    # Truncate if too long
    if len(result) > _MAX_SUMMARY_LENGTH:
        result = result[:_MAX_SUMMARY_LENGTH] + "\n\n[... truncated ...]"

    return result


def _build_rich_content_summary(
    book: Book,
    chapter_analyses: list[ChapterAnalysis],
    concept_graph: ConceptGraph | None = None,
) -> str:
    """Build a rich content summary using deep reading analysis.

    Replaces 200-char text snippets with structured concept inventories,
    dependency order, and section characterizations.
    """
    parts: list[str] = [
        f"# Document: {book.title}",
        f"Author: {book.author}",
        f"Total pages: {book.total_pages}",
        f"Chapters: {len(book.chapters)}",
        "",
    ]

    # Add concept dependency order from the graph
    if concept_graph and concept_graph.topological_order:
        parts.append("# Concept Dependency Order (learn in this sequence):")
        parts.append(" → ".join(concept_graph.topological_order[:30]))
        parts.append("")

    if concept_graph and concept_graph.foundation_concepts:
        parts.append(f"# Foundation concepts (start here): {', '.join(concept_graph.foundation_concepts[:10])}")
        parts.append("")

    if concept_graph and concept_graph.advanced_concepts:
        parts.append(f"# Advanced concepts (build toward): {', '.join(concept_graph.advanced_concepts[:10])}")
        parts.append("")

    # Per-chapter analysis
    analyses_by_num = {a.chapter_number: a for a in chapter_analyses}

    for chapter in book.chapters:
        analysis = analyses_by_num.get(chapter.chapter_number)
        section_count = len(chapter.sections)
        parts.append(
            f"## Chapter {chapter.chapter_number}: {chapter.title} "
            f"(pp. {chapter.start_page}-{chapter.end_page}, {section_count} sections)"
        )

        if analysis:
            # Concepts in this chapter (compact: names only to fit all chapters)
            if analysis.concepts:
                concept_names = [
                    concept_graph.resolve(c.name) if concept_graph else c.name
                    for c in analysis.concepts
                ]
                parts.append(f"  Concepts ({len(concept_names)}): {', '.join(concept_names)}")

            # External prerequisites
            if analysis.external_prerequisites:
                parts.append(f"  Prerequisites from prior chapters: {', '.join(analysis.external_prerequisites)}")

            # Logical flow
            if analysis.logical_flow:
                parts.append(f"  Logical flow: {analysis.logical_flow}")

            # Section characterizations
            for sc in analysis.section_characterizations:
                flags = []
                if sc.has_formulas:
                    flags.append("formulas")
                if sc.has_procedures:
                    flags.append("procedures")
                if sc.has_comparisons:
                    flags.append("comparisons")
                if sc.has_definitions:
                    flags.append("definitions")
                if sc.has_examples:
                    flags.append("examples")
                flags_str = f" [{', '.join(flags)}]" if flags else ""
                # Find concepts in this section
                section_concepts = [
                    concept_graph.resolve(c.name) if concept_graph else c.name
                    for c in analysis.concepts if c.section_title == sc.section_title
                ]
                concepts_str = f" concepts=[{', '.join(section_concepts)}]" if section_concepts else ""
                parts.append(
                    f"  Section: {sc.section_title} — {sc.dominant_content_type}, "
                    f"{sc.difficulty_estimate}{flags_str}{concepts_str}"
                )
                if sc.summary:
                    parts.append(f"    Summary: {sc.summary}")
        else:
            # Fallback to snippets if no analysis for this chapter
            for section in chapter.sections:
                snippet = section.text.strip()[:_SECTION_SNIPPET_LENGTH]
                if len(section.text.strip()) > _SECTION_SNIPPET_LENGTH:
                    snippet += "..."
                parts.append(
                    f"  - {section.title} (pp. {section.start_page}-{section.end_page})"
                )
                if snippet:
                    parts.append(f"    > {snippet}")

        parts.append("")

    result = "\n".join(parts)
    if len(result) > _MAX_SUMMARY_LENGTH * 4:  # Allow generous budget for rich summaries
        result = result[:_MAX_SUMMARY_LENGTH * 4] + "\n\n[... truncated ...]"

    return result


def _build_rich_multi_doc_content_summary(
    books: list[Book],
    chapter_analyses_per_book: list[list[ChapterAnalysis]],
    concept_graph: ConceptGraph | None = None,
) -> str:
    """Build a rich multi-document summary using deep reading analysis."""
    parts: list[str] = [
        f"# Corpus: {len(books)} documents",
        "",
    ]

    # Global concept dependency order
    if concept_graph and concept_graph.topological_order:
        parts.append("# Concept Dependency Order (learn in this sequence):")
        parts.append(" → ".join(concept_graph.topological_order[:30]))
        parts.append("")

    if concept_graph and concept_graph.foundation_concepts:
        parts.append(f"# Foundation concepts: {', '.join(concept_graph.foundation_concepts[:10])}")
        parts.append("")

    for book_idx, book in enumerate(books):
        analyses = chapter_analyses_per_book[book_idx] if book_idx < len(chapter_analyses_per_book) else []
        parts.append("---")
        parts.append(f"# Document {book_idx} (book_index={book_idx}): {book.title}")
        parts.append(f"Author: {book.author} | Pages: {book.total_pages} | Chapters: {len(book.chapters)}")
        parts.append("")

        rich = _build_rich_content_summary(book, analyses, None)
        # Skip the header lines (already added), take the chapter details
        for line in rich.split("\n"):
            if line.startswith("## Chapter") or line.startswith("  "):
                parts.append(line)

        parts.append("")

    result = "\n".join(parts)
    if len(result) > _MAX_MULTI_DOC_SUMMARY_LENGTH * 2:
        result = result[:_MAX_MULTI_DOC_SUMMARY_LENGTH * 2] + "\n\n[... truncated ...]"

    return result


# ── Blueprint completion (fill missing chapters) ──────────────────────────

# Template rotation based on document type for passthrough modules.
_TEMPLATE_ROTATION = [
    "analogy_first", "worked_example", "compare_contrast",
    "problem_first", "socratic", "narrative", "vignette",
    "visual_walkthrough", "error_identification",
]


def _ensure_all_chapters_covered(
    blueprint: CurriculumBlueprint,
    book: Book,
    chapter_analyses: list[ChapterAnalysis] | None = None,
    document_type: DocumentType = "mixed",
) -> CurriculumBlueprint:
    """Fill in passthrough modules for any chapters the LLM skipped.

    After the LLM returns a (potentially incomplete) blueprint, this function
    checks which chapters are missing and adds deterministic passthrough
    modules for them. This guarantees every chapter in the book appears in
    the blueprint.
    """
    covered = {m.source_chapter_number for m in blueprint.modules if m.source_chapter_number is not None}
    missing = [ch for ch in book.chapters if ch.chapter_number not in covered]

    if not missing:
        return blueprint

    logger.warning(
        "LLM blueprint covers %d/%d chapters. Adding passthrough modules for %d missing chapters.",
        len(covered), len(book.chapters), len(missing),
    )

    analyses_by_num = {}
    if chapter_analyses:
        analyses_by_num = {a.chapter_number: a for a in chapter_analyses}

    # Get weighted template list for this document type
    weights = DOCUMENT_TYPE_TEMPLATE_WEIGHTS.get(document_type, DOCUMENT_TYPE_TEMPLATE_WEIGHTS["mixed"])
    templates_by_weight = sorted(weights.keys(), key=lambda t: -weights[t])

    new_modules = list(blueprint.modules)

    for chapter in missing:
        analysis = analyses_by_num.get(chapter.chapter_number)

        # Strip "CHAPTER N: " prefix from chapter title for the module title
        title = re.sub(r"^CHAPTER\s+\d+[:\s]*", "", chapter.title).strip() or chapter.title

        sections: list[SectionBlueprint] = []
        for idx, section in enumerate(chapter.sections):
            # Pick template based on analysis or rotation
            template = templates_by_weight[idx % len(templates_by_weight)]

            # Try to use section characterization for smarter template choice
            if analysis:
                for sc in analysis.section_characterizations:
                    if sc.section_title == section.title:
                        ct = sc.dominant_content_type
                        if ct in ("theoretical", "conceptual"):
                            template = "analogy_first"
                        elif ct == "procedural":
                            template = "visual_walkthrough"
                        elif ct == "comparative":
                            template = "compare_contrast"
                        elif ct == "applied":
                            template = "worked_example"
                        break

            sections.append(SectionBlueprint(
                title=section.title,
                source_section_title=section.title,
                template=template,
                bloom_target="understand",
            ))

        new_modules.append(ModuleBlueprint(
            title=title,
            source_chapter_number=chapter.chapter_number,
            summary=f"Chapter {chapter.chapter_number}: {title}",
            sections=sections,
        ))

    # Sort modules by chapter number to maintain order
    new_modules.sort(key=lambda m: m.source_chapter_number or 0)

    return CurriculumBlueprint(
        course_title=blueprint.course_title,
        course_summary=blueprint.course_summary,
        learner_journey=blueprint.learner_journey,
        modules=new_modules,
    )


# ── Bloom's progression validation ─────────────────────────────────────────

# Canonical ordering of Bloom's levels for progression validation.
_BLOOM_ORDER: dict[str, int] = {
    "remember": 1,
    "understand": 2,
    "apply": 3,
    "analyze": 4,
    "evaluate": 5,
    "create": 6,
}


def validate_progression(blueprint: CurriculumBlueprint) -> list[str]:
    """Check that Bloom's levels generally ramp up within each module.

    Allows small dips (one level) but flags significant regressions where
    a section drops more than one Bloom level below its predecessor.

    Args:
        blueprint: The curriculum blueprint to validate.

    Returns:
        List of warning strings. Empty if progression is valid.
    """
    warnings: list[str] = []

    for module in blueprint.modules:
        prev_level = 0
        for section in module.sections:
            level = _BLOOM_ORDER.get(section.bloom_target, 2)
            if level < prev_level - 1:
                warnings.append(
                    f"Module '{module.title}': section '{section.title}' "
                    f"({section.bloom_target}, level {level}) drops significantly "
                    f"from previous section (level {prev_level})"
                )
            prev_level = level

    return warnings


# ── Internal: chapter/section matching ───────────────────────────────────────


def find_matching_chapter(book: Book, module_bp: ModuleBlueprint) -> Chapter | None:
    """Find the extracted Chapter that matches a blueprint module.

    Tries source_chapter_number first, then falls back to title matching.
    """
    if module_bp.source_chapter_number is not None:
        for ch in book.chapters:
            if ch.chapter_number == module_bp.source_chapter_number:
                return ch

    # Fallback: match by title
    bp_title_lower = module_bp.title.lower().strip()
    for ch in book.chapters:
        if ch.title.lower().strip() == bp_title_lower:
            return ch

    return None


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
