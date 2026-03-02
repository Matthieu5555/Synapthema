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
from src.transformation.analysis_types import ChapterAnalysis, ConceptGraph, SectionCharacterization
from src.transformation.content_pre_analyzer import (
    DOCUMENT_TYPE_TEMPLATE_WEIGHTS,
    DocumentType,
    format_document_type_guidance,
)
from src.transformation.llm_client import LLMClient
from src.transformation.types import (
    BloomLevel,
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

1. **SPLIT sections into atomic learning units** — when a source section \
covers 2+ distinct concepts (listed in the section characterization), create \
MULTIPLE blueprint sections from that single source section. Each sub-section \
should focus on exactly 1 concept, with its own learning objectives, \
template, and bloom_target. All sub-sections sharing a source section MUST use \
the SAME `source_section_title`. Use `focus_concepts` to specify which concept \
name each sub-section covers. Only keep a section as a single blueprint when it \
covers just 1 concept.
2. **REORGANIZE bad structure** — if topics are scattered or the document has no \
clear chapters, group related topics logically and create a learning progression.
3. **SCAFFOLD learning** — order topics from foundational → advanced within each module. \
When a concept dependency graph is provided, FOLLOW it: foundation concepts first, \
then concepts that build on them, then advanced concepts last.
4. **Map back to source** — every section in the blueprint must set \
`source_section_title` to the EXACT title of the extracted section it draws from. \
Multiple blueprint sections CAN share the same `source_section_title` when a \
source section is split into concept-focused units. This is how the content \
generator finds the source text.
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
11. **Set focus_concepts** — for each blueprint section, list the concept names \
(from the concept inventory) that this section should teach and practice. \
Use EXACT concept names from the inventory. When a source section is split \
into multiple learning units, each sub-section's focus_concepts should be a \
disjoint subset covering all the section's concepts. When a source section is \
kept as one unit, focus_concepts can be empty (covers all).

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
          "title": "Understanding Basis",
          "source_section_title": "3.1 Vector Spaces",
          "learning_objectives": ["Define a basis and explain its uniqueness property"],
          "template": "analogy_first",
          "bloom_target": "understand",
          "prerequisites": [],
          "rationale": "Basis is a foundational concept that must be understood before span.",
          "focus_concepts": ["basis"]
        },
        {
          "title": "Span and Its Relationship to Basis",
          "source_section_title": "3.1 Vector Spaces",
          "learning_objectives": ["Explain span and how it relates to basis"],
          "template": "analogy_first",
          "bloom_target": "understand",
          "prerequisites": ["Understanding Basis"],
          "rationale": "Span builds directly on the concept of basis.",
          "focus_concepts": ["span"]
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

    # Split sections that cover too many concepts into bite-size units
    blueprint = _split_overloaded_sections(
        blueprint, chapter_analyses, concept_graph,
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
Your output is a single CurriculumBlueprint JSON that MERGES content \
from all documents into one coherent course.

## MULTI-DOCUMENT CURRICULUM DESIGN RULES

1. **IDENTIFY ALL TOPICS** across all documents — build a complete topic inventory.
2. **MERGE OVERLAPPING CONTENT** — when two or more chapters across documents \
cover the SAME core concepts, MERGE them into a SINGLE unified module. \
Set `source_book_index` and `source_chapter_number` to the PRIMARY source \
(the one with the best or most complete explanation). List the other sources \
in `additional_source_chapters` (each entry: `{"book_index": N, "chapter_number": M}`). \
The module's sections should draw on BOTH sources — interleave the best explanations, \
examples, and perspectives from each book. The total number of modules should \
equal the number of UNIQUE TOPICS, NOT the number of source chapters.
3. **DETECT PREREQUISITES** — for each topic, determine what must be understood \
first. Build a DAG of prerequisites, then linearize it.
4. **DIFFICULTY RAMP** — the first modules should cover foundational vocabulary \
and concepts. Middle modules should build on these with applications. Final \
modules should require synthesis across topics.
5. **SOURCE ATTRIBUTION** — every module must set `source_book_index` to the \
index of the PRIMARY book it draws from (0-based). Every section must set \
`source_section_title` to the EXACT title of an extracted section (from any book). \
Sections within a merged module CAN reference sections from different books \
via their `source_book_index`. Multiple blueprint sections CAN share the same \
`source_section_title` when a source section is split into concept-focused units.
6. **source_chapter_number** must match the chapter number from the primary source book.
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
11. **SPLIT sections into atomic learning units** — when a source section \
covers 2+ distinct concepts, create MULTIPLE blueprint sections from that single \
source section. Each sub-section should focus on exactly 1 concept. \
Use `focus_concepts` to specify which concept name each sub-section covers. \
Only keep a section as a single unit when it covers just 1 concept.
12. **Set focus_concepts** — for each blueprint section, list the concept names \
that this section should teach and practice. Use EXACT concept names from the \
concept inventory. When a source section is split, each sub-section's \
focus_concepts should be a disjoint subset covering all the section's concepts.
13. **COVER ALL CONCEPTS** — you must cover ALL concepts from ALL documents. \
No concept should be dropped. However, you SHOULD merge chapters that cover \
the same concepts into unified modules. A single merged module can satisfy \
coverage for chapters from multiple books. Every source chapter must appear \
either as a module's primary source OR in some module's `additional_source_chapters`.

## OUTPUT FORMAT

Return ONLY a JSON object matching this schema (no markdown fences):

{
  "course_title": "Unified course title",
  "course_summary": "2-3 sentence overview covering all source documents",
  "learner_journey": "Module A → Module B → ...",
  "modules": [
    {
      "title": "Value at Risk: Concepts and Applications",
      "source_chapter_number": 3,
      "source_book_index": 0,
      "additional_source_chapters": [{"book_index": 1, "chapter_number": 5}],
      "summary": "Unified treatment of VaR drawing on both sources.",
      "sections": [
        {
          "title": "VaR Definition and Intuition",
          "source_section_title": "3.1 VaR Overview",
          "source_book_index": 0,
          "learning_objectives": ["Define VaR and explain its interpretation"],
          "template": "analogy_first",
          "bloom_target": "understand",
          "prerequisites": [],
          "rationale": "Foundation concept — combine both books' perspectives.",
          "focus_concepts": ["value at risk"]
        },
        {
          "title": "Confidence Levels in VaR",
          "source_section_title": "3.1 VaR Overview",
          "source_book_index": 0,
          "learning_objectives": ["Explain how confidence level affects VaR estimates"],
          "template": "analogy_first",
          "bloom_target": "understand",
          "prerequisites": ["VaR Definition and Intuition"],
          "rationale": "Confidence level is a separate concept that builds on VaR definition.",
          "focus_concepts": ["confidence level"]
        },
        {
          "title": "Parametric VaR Calculation",
          "source_section_title": "5.2 Parametric VaR",
          "source_book_index": 1,
          "learning_objectives": ["Calculate VaR using the parametric method"],
          "template": "worked_example",
          "bloom_target": "apply",
          "prerequisites": ["Confidence Levels in VaR"],
          "rationale": "Book 2 has better worked examples for calculations.",
          "focus_concepts": ["parametric VaR"]
        },
        {
          "title": "Historical Simulation for VaR",
          "source_section_title": "5.2 Parametric VaR",
          "source_book_index": 1,
          "learning_objectives": ["Calculate VaR using historical simulation"],
          "template": "compare_contrast",
          "bloom_target": "apply",
          "prerequisites": ["Parametric VaR Calculation"],
          "rationale": "Contrast with parametric approach to deepen understanding.",
          "focus_concepts": ["historical simulation"]
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

    # Fill in properly-designed modules for any chapters the LLM skipped
    blueprint = _ensure_all_chapters_covered_multi_doc(
        blueprint, books, client, chapter_analyses_per_book, document_type,
    )

    # Split sections that cover too many concepts into bite-size units
    blueprint = _split_overloaded_sections(
        blueprint, None, concept_graph,
        chapter_analyses_per_book=chapter_analyses_per_book,
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
            f"## Chapter {chapter.chapter_number} "
            f"(source_chapter_number={chapter.chapter_number}): "
            f"{chapter.title} "
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


def _format_section_characterization(
    sc: SectionCharacterization,
    analysis: ChapterAnalysis,
    concept_graph: ConceptGraph | None,
) -> list[str]:
    """Format a single section characterization into summary lines."""
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

    section_concepts = [
        concept_graph.resolve(c.name) if concept_graph else c.name
        for c in analysis.concepts if c.section_title.lower().strip() == sc.section_title.lower().strip()
    ]
    concepts_str = f" concepts=[{', '.join(section_concepts)}]" if section_concepts else ""

    lines = [
        f"  Section: {sc.section_title} — {sc.dominant_content_type}, "
        f"{sc.difficulty_estimate}{flags_str}{concepts_str}"
    ]
    if sc.summary:
        lines.append(f"    Summary: {sc.summary}")
    return lines


def _format_chapter_analysis(
    chapter: Chapter,
    analysis: ChapterAnalysis | None,
    concept_graph: ConceptGraph | None,
) -> list[str]:
    """Format a single chapter's analysis into summary lines."""
    lines: list[str] = [
        f"## Chapter {chapter.chapter_number} "
        f"(source_chapter_number={chapter.chapter_number}): "
        f"{chapter.title} "
        f"(pp. {chapter.start_page}-{chapter.end_page}, {len(chapter.sections)} sections)"
    ]

    if analysis:
        if analysis.concepts:
            concept_names = [
                concept_graph.resolve(c.name) if concept_graph else c.name
                for c in analysis.concepts
            ]
            lines.append(f"  Concepts ({len(concept_names)}): {', '.join(concept_names)}")
        if analysis.external_prerequisites:
            lines.append(f"  Prerequisites from prior chapters: {', '.join(analysis.external_prerequisites)}")
        if analysis.logical_flow:
            lines.append(f"  Logical flow: {analysis.logical_flow}")
        for sc in analysis.section_characterizations:
            lines.extend(_format_section_characterization(sc, analysis, concept_graph))
    else:
        for section in chapter.sections:
            snippet = section.text.strip()[:_SECTION_SNIPPET_LENGTH]
            if len(section.text.strip()) > _SECTION_SNIPPET_LENGTH:
                snippet += "..."
            lines.append(f"  - {section.title} (pp. {section.start_page}-{section.end_page})")
            if snippet:
                lines.append(f"    > {snippet}")

    lines.append("")
    return lines


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

    analyses_by_num = {a.chapter_number: a for a in chapter_analyses}
    for chapter in book.chapters:
        analysis = analyses_by_num.get(chapter.chapter_number)
        parts.extend(_format_chapter_analysis(chapter, analysis, concept_graph))

    result = "\n".join(parts)
    if len(result) > _MAX_SUMMARY_LENGTH * 4:
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

        rich = _build_rich_content_summary(book, analyses, concept_graph)
        # Skip the header lines (already added), take the chapter details
        for line in rich.split("\n"):
            if line.startswith("## Chapter") or line.startswith("  "):
                parts.append(line)

        parts.append("")

    result = "\n".join(parts)
    max_len = _MAX_MULTI_DOC_SUMMARY_LENGTH * max(len(books), 1)
    if len(result) > max_len:
        result = result[:max_len] + "\n\n[... truncated ...]"

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


_MISSING_CHAPTERS_PROMPT = """\
You are a curriculum architect. You previously designed a multi-document course \
but MISSED some chapters. Below is (A) your existing curriculum and (B) the \
chapters you missed. Design ADDITIONAL modules that integrate these missing \
chapters into a coherent extension of the existing course.

## RULES
1. Create one module per missing chapter (or group tightly related chapters).
2. Give each module a descriptive title — NOT just "Chapter N Title".
3. Each module must set `source_book_index` and `source_chapter_number` to \
match the source document.
4. Assign templates based on content type, not rotation.
5. Set bloom_target based on content complexity.
6. Add 1-3 learning objectives per section.
7. Map `source_section_title` to the EXACT extracted section titles.
8. Consider how these modules relate to the existing ones — set prerequisites \
when a missing module builds on an existing one.

## OUTPUT FORMAT
Return ONLY a JSON object:
{
  "modules": [ ... same format as before ... ]
}"""


def _ensure_all_chapters_covered_multi_doc(
    blueprint: CurriculumBlueprint,
    books: list[Book],
    client: LLMClient,
    chapter_analyses_per_book: list[list[ChapterAnalysis]] | None = None,
    document_type: DocumentType = "mixed",
) -> CurriculumBlueprint:
    """Design proper modules for any chapters the LLM skipped across all books.

    Multi-document version of _ensure_all_chapters_covered(). Checks every
    chapter in every book. For missing chapters, sends a follow-up LLM call
    to design properly-titled modules with learning objectives and template
    assignments — not raw chapter dumps.
    """
    # Build coverage set from module-level, section-level, AND additional sources
    covered: set[tuple[int, int]] = set()
    for m in blueprint.modules:
        if m.source_chapter_number is not None:
            book_idx = m.source_book_index or 0
            covered.add((book_idx, m.source_chapter_number))
        # Check section-level: a module from book 0 may reference
        # sections from book 1 via source_book_index on sections
        for s in m.sections:
            if s.source_book_index is not None and m.source_chapter_number is not None:
                covered.add((s.source_book_index, m.source_chapter_number))
        # Check additional_source_chapters (merged modules)
        for asc in m.additional_source_chapters:
            asc_book = asc.get("book_index")
            asc_ch = asc.get("chapter_number")
            if asc_book is not None and asc_ch is not None:
                covered.add((asc_book, asc_ch))

    # Find missing chapters across all books
    missing: list[tuple[int, Book, Chapter]] = []
    for book_idx, book in enumerate(books):
        for chapter in book.chapters:
            if (book_idx, chapter.chapter_number) not in covered:
                missing.append((book_idx, book, chapter))

    if not missing:
        return blueprint

    total_chapters = sum(len(b.chapters) for b in books)
    logger.warning(
        "Multi-doc blueprint covers %d/%d chapters. Designing modules for %d missing chapters via LLM.",
        len(covered), total_chapters, len(missing),
    )

    # Build context: existing modules + missing chapter summaries
    existing_summary = "## (A) EXISTING MODULES\n"
    for m in blueprint.modules:
        existing_summary += f"- {m.title} (book={m.source_book_index}, ch={m.source_chapter_number})\n"

    missing_summary = "\n## (B) MISSING CHAPTERS — design modules for these\n"
    for book_idx, book, chapter in missing:
        section_count = len(chapter.sections)
        missing_summary += (
            f"\n### Document {book_idx}: {book.title}\n"
            f"Chapter {chapter.chapter_number}: {chapter.title} "
            f"(pp. {chapter.start_page}-{chapter.end_page}, {section_count} sections)\n"
        )
        # Include section info and analysis if available
        analyses_by_num: dict[int, ChapterAnalysis] = {}
        if chapter_analyses_per_book and book_idx < len(chapter_analyses_per_book):
            analyses_by_num = {
                a.chapter_number: a for a in chapter_analyses_per_book[book_idx]
            }
        analysis = analyses_by_num.get(chapter.chapter_number)

        for section in chapter.sections:
            text_len = len(section.text.strip())
            missing_summary += f"  - {section.title} ({text_len} chars)\n"

        if analysis:
            concept_names = [c.name for c in analysis.concepts]
            if concept_names:
                missing_summary += f"  Concepts: {', '.join(concept_names[:15])}\n"
            for sc in analysis.section_characterizations:
                missing_summary += f"  Section '{sc.section_title}': {sc.dominant_content_type}, {sc.difficulty_estimate}\n"

    user_prompt = existing_summary + missing_summary

    # Use a structured response that just has "modules" list
    from pydantic import BaseModel

    class _SupplementBlueprint(BaseModel):
        modules: list[ModuleBlueprint]

    try:
        supplement = client.complete_structured(
            _MISSING_CHAPTERS_PROMPT, user_prompt, _SupplementBlueprint
        )
        new_modules = list(blueprint.modules) + supplement.modules

        logger.info(
            "LLM designed %d additional modules for missing chapters (total now: %d)",
            len(supplement.modules), len(new_modules),
        )
    except Exception:
        logger.warning(
            "LLM follow-up for missing chapters failed. Falling back to passthrough modules.",
        )
        # Fallback: deterministic passthrough (better than losing chapters)
        weights = DOCUMENT_TYPE_TEMPLATE_WEIGHTS.get(
            document_type, DOCUMENT_TYPE_TEMPLATE_WEIGHTS["mixed"]
        )
        templates_by_weight = sorted(weights.keys(), key=lambda t: -weights[t])

        new_modules = list(blueprint.modules)
        for book_idx, book, chapter in missing:
            title = re.sub(r"^CHAPTER\s+\d+[:\s]*", "", chapter.title).strip() or chapter.title
            sections: list[SectionBlueprint] = []
            for idx, section in enumerate(chapter.sections):
                sections.append(SectionBlueprint(
                    title=section.title,
                    source_section_title=section.title,
                    source_book_index=book_idx,
                    template=templates_by_weight[idx % len(templates_by_weight)],
                    bloom_target="understand",
                ))
            new_modules.append(ModuleBlueprint(
                title=title,
                source_chapter_number=chapter.chapter_number,
                source_book_index=book_idx,
                summary=f"Chapter {chapter.chapter_number}: {title}",
                sections=sections,
            ))

    # Sort: keep LLM-designed modules first, then supplementary by book order
    original_count = len(blueprint.modules)
    original = new_modules[:original_count]
    supplementary = new_modules[original_count:]
    supplementary.sort(
        key=lambda m: (m.source_book_index or 0, m.source_chapter_number or 0)
    )
    new_modules = original + supplementary

    return CurriculumBlueprint(
        course_title=blueprint.course_title,
        course_summary=blueprint.course_summary,
        learner_journey=blueprint.learner_journey,
        modules=new_modules,
    )


# ── Concept-level section splitting (deterministic safety net) ─────────────

# Minimum number of core/supporting concepts to trigger automatic splitting.
_MIN_CONCEPTS_TO_SPLIT = 2

# Maximum concepts per learning unit when auto-splitting.
_MAX_CONCEPTS_PER_UNIT = 1


def _split_overloaded_sections(
    blueprint: CurriculumBlueprint,
    chapter_analyses: list[ChapterAnalysis] | None,
    concept_graph: ConceptGraph | None = None,
    chapter_analyses_per_book: list[list[ChapterAnalysis]] | None = None,
) -> CurriculumBlueprint:
    """Deterministically split sections that cover too many concepts.

    Safety net for when the LLM planner doesn't split dense sections itself.
    Splits sections that:
    1. Have focus_concepts with 2+ items (LLM bundled multiple concepts), OR
    2. Have no focus_concepts but map to a source section with 2+ core/supporting
       concepts (requires deep reading analysis).

    Args:
        blueprint: The curriculum blueprint to post-process.
        chapter_analyses: Deep reading analyses per chapter (single-doc).
        concept_graph: Consolidated concept dependency graph.
        chapter_analyses_per_book: Per-book analyses (multi-doc). When provided,
            uses (book_index, chapter_number) keys to avoid collisions.

    Returns:
        Blueprint with overloaded sections split into focused units.
    """
    # Multi-doc: key by (book_index, chapter_number) to avoid collisions
    if chapter_analyses_per_book:
        analyses_by_book_ch: dict[tuple[int, int], ChapterAnalysis] = {}
        for book_idx, book_analyses in enumerate(chapter_analyses_per_book):
            for a in book_analyses:
                analyses_by_book_ch[(book_idx, a.chapter_number)] = a
    else:
        analyses_by_book_ch = None

    # Single-doc fallback: key by chapter_number only
    analyses_by_num = (
        {a.chapter_number: a for a in chapter_analyses}
        if chapter_analyses else {}
    )

    new_modules = []
    any_split = False

    for module in blueprint.modules:
        # Multi-doc: look up by (book_index, chapter_number) tuple
        if analyses_by_book_ch is not None and module.source_book_index is not None and module.source_chapter_number is not None:
            analysis = analyses_by_book_ch.get((module.source_book_index, module.source_chapter_number))
        elif module.source_chapter_number is not None:
            analysis = analyses_by_num.get(module.source_chapter_number)
        else:
            analysis = None

        new_sections: list[SectionBlueprint] = []
        for section_bp in module.sections:
            # Path A: already focused on exactly 1 concept — keep as-is
            if section_bp.focus_concepts and len(section_bp.focus_concepts) == 1:
                new_sections.append(section_bp)
                continue

            # Path B: LLM bundled 2+ concepts in focus_concepts — re-split
            if section_bp.focus_concepts and len(section_bp.focus_concepts) >= 2:
                any_split = True
                logger.info(
                    "Re-splitting section '%s' with %d focus_concepts into %d units",
                    section_bp.title, len(section_bp.focus_concepts),
                    len(section_bp.focus_concepts),
                )
                for i, concept_name in enumerate(section_bp.focus_concepts):
                    base_title = section_bp.source_section_title or section_bp.title
                    sub_title = f"{base_title}: {concept_name}"
                    objectives = [f"Explain {concept_name} and its role"]
                    prerequisites = (
                        list(section_bp.prerequisites) if i == 0
                        else [new_sections[-1].title]
                    )
                    new_sections.append(SectionBlueprint(
                        title=sub_title,
                        source_section_title=section_bp.source_section_title,
                        source_book_index=section_bp.source_book_index,
                        learning_objectives=objectives,
                        template=section_bp.template,
                        bloom_target=section_bp.bloom_target,
                        prerequisites=prerequisites,
                        rationale=f"Auto-split: focusing on {concept_name} from multi-concept section.",
                        focus_concepts=[concept_name],
                    ))
                continue

            # Path C: no focus_concepts — need analysis to determine concept count
            if not analysis:
                new_sections.append(section_bp)
                continue

            # Find concepts for this source section
            source_title = section_bp.source_section_title or section_bp.title
            section_concepts = [
                c for c in analysis.concepts
                if c.section_title.lower().strip() == source_title.lower().strip()
            ]

            # Filter to core + supporting (ignore peripheral)
            meaningful = [
                c for c in section_concepts if c.importance != "peripheral"
            ]

            if len(meaningful) < _MIN_CONCEPTS_TO_SPLIT:
                new_sections.append(section_bp)
                continue

            # Order by concept graph topology if available
            if concept_graph and concept_graph.topological_order:
                topo_index = {
                    name.lower().strip(): i
                    for i, name in enumerate(concept_graph.topological_order)
                }
                meaningful.sort(
                    key=lambda c: topo_index.get(
                        concept_graph.resolve(c.name).lower().strip(), 999
                    )
                )

            # Chunk into groups of _MAX_CONCEPTS_PER_UNIT
            chunks = [
                meaningful[i : i + _MAX_CONCEPTS_PER_UNIT]
                for i in range(0, len(meaningful), _MAX_CONCEPTS_PER_UNIT)
            ]

            any_split = True
            logger.info(
                "Auto-splitting section '%s' (%d concepts) into %d units",
                section_bp.title, len(meaningful), len(chunks),
            )

            for i, chunk in enumerate(chunks):
                concept_names = [c.name for c in chunk]
                title_parts = " and ".join(concept_names[:_MAX_CONCEPTS_PER_UNIT])
                sub_title = f"{section_bp.title}: {title_parts}"

                objectives = [
                    f"Explain {c.name} and its role" for c in chunk
                ]

                bloom = _bloom_from_concept_position(chunk, concept_graph)

                prerequisites = list(section_bp.prerequisites) if i == 0 else [new_sections[-1].title]

                new_sections.append(SectionBlueprint(
                    title=sub_title,
                    source_section_title=section_bp.source_section_title,
                    source_book_index=section_bp.source_book_index,
                    learning_objectives=objectives,
                    template=section_bp.template,
                    bloom_target=bloom or section_bp.bloom_target,
                    prerequisites=prerequisites,
                    rationale=f"Auto-split: focusing on {', '.join(concept_names)} from dense section.",
                    focus_concepts=concept_names,
                ))

        new_modules.append(ModuleBlueprint(
            title=module.title,
            source_chapter_number=module.source_chapter_number,
            source_book_index=module.source_book_index,
            summary=module.summary,
            sections=new_sections,
        ))

    if not any_split:
        return blueprint

    return CurriculumBlueprint(
        course_title=blueprint.course_title,
        course_summary=blueprint.course_summary,
        learner_journey=blueprint.learner_journey,
        modules=new_modules,
    )


def _bloom_from_concept_position(
    concepts: list,
    concept_graph: ConceptGraph | None,
) -> BloomLevel | None:
    """Determine bloom_target from concept position in dependency graph."""
    if not concept_graph:
        return None

    for c in concepts:
        canonical = concept_graph.resolve(c.name)
        if canonical in concept_graph.foundation_concepts:
            return "understand"
        if canonical in concept_graph.advanced_concepts:
            return "analyze"

    # Default based on how many prerequisites the concepts have
    max_deps = 0
    for c in concepts:
        canonical = concept_graph.resolve(c.name)
        deps = sum(1 for e in concept_graph.edges if e.source == canonical)
        max_deps = max(max_deps, deps)

    if max_deps == 0:
        return "understand"
    elif max_deps <= 2:
        return "apply"
    else:
        return "analyze"


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
    Allows Bloom resets when transitioning between different source sections
    (concept sub-units for a new topic naturally restart the progression).

    Args:
        blueprint: The curriculum blueprint to validate.

    Returns:
        List of warning strings. Empty if progression is valid.
    """
    warnings: list[str] = []

    for module in blueprint.modules:
        prev_level = 0
        prev_source = ""
        for section in module.sections:
            level = _BLOOM_ORDER.get(section.bloom_target, 2)
            source = section.source_section_title or section.title

            # Allow Bloom reset when switching to a new source section
            if source != prev_source:
                prev_level = 0

            if level < prev_level - 1:
                warnings.append(
                    f"Module '{module.title}': section '{section.title}' "
                    f"({section.bloom_target}, level {level}) drops significantly "
                    f"from previous section (level {prev_level})"
                )
            prev_level = level
            prev_source = source

    return warnings


# ── Internal: chapter/section matching ───────────────────────────────────────


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


_CHAPTER_NUM_RE = re.compile(r"(?:chapter|ch\.?)\s+(\d+)", re.IGNORECASE)


def _extract_chapter_number_from_title(title: str) -> int | None:
    """Extract a chapter number from a title like 'CHAPTER 3 Fundamentals'."""
    m = _CHAPTER_NUM_RE.search(title)
    return int(m.group(1)) if m else None


def _count_section_matches(
    chapter: Chapter, section_bps: list[SectionBlueprint],
) -> int:
    """Count how many blueprint sections match sections in *chapter*."""
    return sum(
        1 for sbp in section_bps
        if find_matching_section(chapter, sbp) is not None
    )


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
