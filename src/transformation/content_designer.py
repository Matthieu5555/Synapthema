"""LLM-powered content transformation — the deep module for Stage 2.

Single public entry point: transform_chapter(). Takes a Chapter from Stage 1
and an LLM client, sends each section through the LLM for transformation into
interactive training elements, and returns a complete TrainingModule.

Uses Instructor for schema-constrained LLM output — the model is forced to
return valid JSON matching our Pydantic models, eliminating JSON parse failures.
"""

from __future__ import annotations

import base64
import io
import logging
import random
import re
import string
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

from collections import Counter

from pydantic import BaseModel, Field, model_validator

from src.extraction.types import Chapter, ImageRef, Section
from src.transformation.content_pre_analyzer import (
    analyze_section as pre_analyze_section,
    classify_section_quality,
)
from src.transformation.analysis_types import (
    ChapterAnalysis,
    ConceptEntry,
    SectionCharacterization,
    resolve_concept,
)
from src.transformation.curriculum_planner import find_matching_section
from src.transformation.llm_client import LLMClient, LLMError
from src.transformation.content_designer_prompts import (
    BLOOM_PROMPT_SUPPLEMENTS,
    SYSTEM_PROMPT,
    TARGET_SELECTION_PROMPT,
    TEMPLATE_DESCRIPTIONS,
    build_section_prompt,
    build_target_selection_prompt,
)
from src.transformation.types import (
    ELEMENT_BLOOM_MAP,
    ELEMENT_ROLE,
    ModuleBlueprint,
    ReinforcementTarget,
    ReinforcementTargetSet,
    SectionBlueprint,
    SlideElement,
    Slide,
    TrainingElement,
    TrainingModule,
    TrainingSection,
)

logger = logging.getLogger(__name__)

# Minimum character count for a section's text to be worth sending to the LLM.
# Sections shorter than this are typically just headers or brief transitions
# that don't contain enough substance for interactive training content.
MIN_SECTION_TEXT_LENGTH = 100

# Minimum section text length for Phase 1 (target selection) to be worthwhile.
# Sections shorter than this have too little content for meaningful reinforcement
# target identification — skip the extra LLM call and go straight to Phase 2.
_MIN_TEXT_FOR_TARGETS = 500


def _filter_by_focus(
    concepts: list[ConceptEntry],
    focus_concepts: list[str] | None,
) -> list[ConceptEntry]:
    """Filter concepts to the focus set (case-insensitive). Passthrough when empty."""
    if not focus_concepts or not concepts:
        return concepts
    focus_lower = {n.lower().strip() for n in focus_concepts}
    return [c for c in concepts if c.name.lower().strip() in focus_lower]


# Content template rotation — cycled through to prevent monotony.
# Derived from TEMPLATE_DESCRIPTIONS (single source of truth), excluding
# special-purpose templates that require curriculum-planner context.
_ROTATION_EXCLUSIONS = {"visual_summary", "milestone_assessment"}
TEMPLATE_ROTATION = tuple(
    t for t in TEMPLATE_DESCRIPTIONS if t not in _ROTATION_EXCLUSIONS
)


class SectionResponse(BaseModel):
    """Pydantic model for the LLM's structured response.

    Instructor validates the LLM output against this schema automatically.
    If validation fails, Instructor re-prompts with the error for self-correction.
    """

    elements: list[TrainingElement] = Field(
        description="Ordered sequence of interactive training elements"
    )

    @model_validator(mode="after")
    def fix_bloom_levels(self) -> "SectionResponse":
        """Override LLM-chosen bloom levels with the canonical mapping."""
        for element in self.elements:
            correct = ELEMENT_BLOOM_MAP.get(element.element_type)
            if correct and element.bloom_level != correct:
                object.__setattr__(element, "bloom_level", correct)
        return self

    @model_validator(mode="after")
    def enforce_element_distribution(self) -> "SectionResponse":
        """Enforce 1-slide-then-drill section structure.

        Architecture: each section = 1 concept = 1 slide + 2-3 exercises + 2-3 flashcards.
        Hard errors catch truly bad outputs (triggers Instructor retry).
        Warnings flag aspirational targets that are not worth a retry.
        """
        counts = Counter(e.element_type for e in self.elements)

        teach_types = {"slide", "mermaid", "worked_example"}
        # Exercise types (excludes flashcard — flashcard is reinforcement, not practice)
        exercise_types = {
            "quiz", "fill_in_the_blank", "matching", "ordering",
            "categorization", "error_detection", "analogy",
        }

        teach_count = sum(counts.get(t, 0) for t in teach_types)
        exercise_count = sum(counts.get(t, 0) for t in exercise_types)
        flashcard_count = counts.get("flashcard", 0)

        # Count distinct exercise types used
        exercise_types_used = {
            t for t in exercise_types if counts.get(t, 0) > 0
        }

        # ── Hard constraints (trigger Instructor retry) ──────────────────

        if teach_count < 1:
            raise ValueError("Section must contain at least 1 teaching element (slide/mermaid)")

        # Max 1 slide per section (mermaid may accompany it for processes).
        # Allow slide + mermaid (2) but not slide + slide (2 slides).
        slide_count = counts.get("slide", 0) + counts.get("worked_example", 0)
        if slide_count > 1:
            raise ValueError(
                f"Section has {slide_count} slides/worked_examples (maximum: 1). "
                f"Each section teaches ONE concept with ONE slide. "
                f"The curriculum planner has already split concepts."
            )

        # Minimum 2 exercises per section
        if exercise_count < 2:
            raise ValueError(
                f"Section has {exercise_count} exercise(s) (minimum: 2). "
                f"Generate 2-3 exercises after the slide."
            )

        if counts.get("interactive_essay", 0) > 2:
            raise ValueError("Section must contain at most 2 interactive_essay elements")

        # Reject if all exercises are the same type (no variety)
        if exercise_count >= 2 and len(exercise_types_used) < 2:
            raise ValueError(
                f"All {exercise_count} exercises are the same type "
                f"({next(iter(exercise_types_used))}). Use at least 2 different exercise types."
            )

        # Reject if more than 1 quiz (MCQ) — overused, boring
        quiz_count = counts.get("quiz", 0)
        if quiz_count > 1:
            raise ValueError(
                f"Section has {quiz_count} quiz (MCQ) elements (maximum: 1). "
                f"Replace extra quizzes with: matching, ordering, fill_in_the_blank, "
                f"categorization, analogy, or error_detection."
            )

        # ── Warnings (aspirational targets, not worth a retry) ───────────

        # Target: 3 exercises for hard concepts
        if exercise_count < 3:
            logger.warning(
                "Section has %d exercises (target: 3 for difficult concepts)",
                exercise_count,
            )

        # Target: 3 different exercise types when 3+ exercises
        if exercise_count >= 3 and len(exercise_types_used) < 3:
            logger.warning(
                "Low exercise variety: %d types used (target: all different types)",
                len(exercise_types_used),
            )

        # Warn on flashcard count (target: 2-3 per section)
        if flashcard_count < 2:
            logger.warning(
                "Low flashcard count: %d (target: 2-3 per section)",
                flashcard_count,
            )
        elif flashcard_count > 4:
            logger.warning(
                "High flashcard count: %d (target: 2-3 per section)",
                flashcard_count,
            )

        return self

    @model_validator(mode="after")
    def enforce_interleaved_order(self) -> "SectionResponse":
        """Preserve LLM's teach-practice interleaving while anchoring bookend elements.

        Core elements (slides + practice) keep their LLM-generated order intact
        so teach-practice cycles are not destroyed. Only bookend elements are
        moved to fixed positions: section_intro first, concept_map/flashcard/
        interactive_essay at the end.
        """
        if not self.elements:
            return self

        anchor_end_roles = {"synthesis", "reinforce", "assess"}
        intro = [e for e in self.elements if e.element_type == "section_intro"]
        core = [
            e for e in self.elements
            if ELEMENT_ROLE.get(e.element_type) not in ({"intro"} | anchor_end_roles)
        ]
        synthesis = [e for e in self.elements if e.element_type == "concept_map"]
        flashcards = [e for e in self.elements if e.element_type == "flashcard"]
        essays = [e for e in self.elements if e.element_type == "interactive_essay"]

        self.elements = intro + core + synthesis + flashcards + essays
        return self


class _SectionInput(NamedTuple):
    """Prepared input for a single section transformation."""

    section: Section
    title: str
    template: str
    learning_objectives: list[str]
    bloom_target: str | None
    rationale: str | None = None
    focus_concepts: list[str] | None = None


def transform_chapter(
    chapter: Chapter,
    client: LLMClient,
    blueprint: ModuleBlueprint | None = None,
    chapter_analysis: ChapterAnalysis | None = None,
    prior_concepts: list[str | dict] | None = None,
    document_type: str | None = None,
    extracted_dir: Path | None = None,
    vision_enabled: bool = False,
    max_workers: int = 4,
    supplementary_context: str | None = None,
    additional_chapters: list[tuple[int, Chapter]] | None = None,
    additional_extracted_dirs: dict[int, Path] | None = None,
    canonical_map: dict[str, str] | None = None,
) -> TrainingModule:
    """Transform a book chapter into an interactive training module.

    When a blueprint is provided (from the curriculum planner), uses its
    section order, template assignments, learning objectives, and Bloom's
    targets. Otherwise falls back to template rotation.

    When deep reading analysis is available, passes concept context to each
    section transformation so the LLM knows exactly which concepts to teach
    and what the learner already knows.

    Args:
        chapter: A Chapter dataclass from Stage 1 extraction.
        client: An LLM client implementing the LLMClient protocol.
        blueprint: Optional module blueprint from the curriculum planner.
        chapter_analysis: Optional deep reading analysis for this chapter.
        prior_concepts: Concept names from previously transformed chapters.
        document_type: Document type hint for prompt tuning.
        extracted_dir: Path to the extraction output directory (for reading
            image files when vision is enabled).
        vision_enabled: If True and images exist, send them to the LLM as
            vision content blocks.
        max_workers: Maximum parallel LLM calls for section processing.
        additional_chapters: Extra chapters from other books merged into
            this module. Each entry is (book_index, Chapter).
        additional_extracted_dirs: Per-book extracted dirs for additional
            chapters' image paths. Maps book_index → Path.

    Returns:
        A TrainingModule with sections containing training elements.
    """
    module_title = blueprint.title if blueprint else chapter.title

    additional_count = len(additional_chapters) if additional_chapters else 0
    logger.info(
        "Transforming chapter %d: '%s' (%d sections%s%s%s)",
        chapter.chapter_number,
        module_title,
        len(chapter.sections),
        ", with blueprint" if blueprint else "",
        ", with analysis" if chapter_analysis else "",
        f", +{additional_count} additional chapters" if additional_count else "",
    )

    # Build a list of section inputs, filtering out unmatched/short sections
    section_inputs = _prepare_section_inputs(
        chapter, blueprint,
        additional_chapters=additional_chapters,
        additional_extracted_dirs=additional_extracted_dirs,
        primary_extracted_dir=extracted_dir,
    )

    # Transform each section (parallel when max_workers > 1)
    training_sections = _fold_transform_sections(
        section_inputs, module_title, client, chapter_analysis, prior_concepts,
        module_summary=blueprint.summary if blueprint else None,
        document_type=document_type,
        extracted_dir=extracted_dir,
        vision_enabled=vision_enabled,
        max_workers=max_workers,
        supplementary_context=supplementary_context,
        canonical_map=canonical_map,
    )

    total_elements = sum(len(s.elements) for s in training_sections)
    logger.info(
        "Chapter %d transformed: %d elements in %d sections",
        chapter.chapter_number,
        total_elements,
        len(training_sections),
    )

    return TrainingModule(
        chapter_number=chapter.chapter_number,
        title=module_title,
        sections=training_sections,
    )


def _prepare_section_inputs(
    chapter: Chapter,
    blueprint: ModuleBlueprint | None,
    additional_chapters: list[tuple[int, Chapter]] | None = None,
    additional_extracted_dirs: dict[int, Path] | None = None,
    primary_extracted_dir: Path | None = None,
) -> list[_SectionInput]:
    """Build the list of section inputs from blueprint or rotation fallback.

    When additional_chapters are provided (multi-doc merged modules), also
    searches them for matching sections and merges text from both sources.

    Filters out unmatched and short sections up front.
    """
    inputs: list[_SectionInput] = []

    if blueprint and blueprint.sections:
        for section_bp in blueprint.sections:
            # Try primary chapter first
            section = find_matching_section(chapter, section_bp)

            # If not found in primary, search additional chapters
            if section is None and additional_chapters:
                for _bk_idx, add_ch in additional_chapters:
                    section = find_matching_section(add_ch, section_bp)
                    if section is not None:
                        # Resolve image paths to absolute if from another book
                        section = _resolve_additional_images(
                            section, _bk_idx, additional_extracted_dirs,
                        )
                        break

            if section is None:
                logger.warning(
                    "Blueprint section '%s' not found in chapter or additional sources, skipping",
                    section_bp.source_section_title or section_bp.title,
                )
                continue

            # If found in primary, also check additional chapters for a
            # matching section to merge content from both sources
            if additional_chapters:
                section = _merge_additional_sections(
                    section, section_bp, additional_chapters,
                    additional_extracted_dirs,
                )

            if len(section.text.strip()) < MIN_SECTION_TEXT_LENGTH:
                logger.debug("Skipping short section: '%s' (%d chars)", section.title, len(section.text))
                continue
            quality = classify_section_quality(section.title, section.text)
            if quality != "content":
                logger.info("Skipping %s section: '%s'", quality, section.title)
                continue
            inputs.append(_SectionInput(
                section=section,
                title=section_bp.title,
                template=section_bp.template,
                learning_objectives=section_bp.learning_objectives,
                bloom_target=section_bp.bloom_target,
                rationale=section_bp.rationale or None,
                focus_concepts=section_bp.focus_concepts or None,
            ))

        if not inputs:
            logger.warning(
                "All %d blueprint sections failed to match chapter '%s', "
                "falling back to chapter's own sections",
                len(blueprint.sections), chapter.title,
            )
            # Fall through to template rotation below

    if not inputs:
        template_idx = 0
        for section in chapter.sections:
            if len(section.text.strip()) < MIN_SECTION_TEXT_LENGTH:
                logger.debug("Skipping short section: '%s' (%d chars)", section.title, len(section.text))
                continue
            quality = classify_section_quality(section.title, section.text)
            if quality != "content":
                logger.info("Skipping %s section: '%s'", quality, section.title)
                continue
            template = TEMPLATE_ROTATION[template_idx % len(TEMPLATE_ROTATION)]
            template_idx += 1
            inputs.append(_SectionInput(
                section=section,
                title=section.title,
                template=template,
                learning_objectives=[],
                bloom_target=None,
            ))

    return inputs


def _resolve_additional_images(
    section: Section,
    book_idx: int,
    additional_extracted_dirs: dict[int, Path] | None,
) -> Section:
    """Convert relative image paths to absolute for sections from additional books.

    When a section comes from an additional chapter (different book), its image
    paths are relative to that book's extraction directory. We convert them to
    absolute paths so they resolve correctly regardless of which extracted_dir
    is used for the module.
    """
    if not section.images or not additional_extracted_dirs:
        return section
    book_dir = additional_extracted_dirs.get(book_idx)
    if not book_dir:
        return section

    resolved_images = tuple(
        ImageRef(
            path=book_dir / img.path,
            page=img.page,
            caption=img.caption,
            bbox=img.bbox,
        )
        for img in section.images
    )
    return Section(
        title=section.title,
        level=section.level,
        start_page=section.start_page,
        end_page=section.end_page,
        text=section.text,
        images=resolved_images,
        tables=section.tables,
        subsections=section.subsections,
    )


def _merge_additional_sections(
    primary_section: Section,
    section_bp: SectionBlueprint,
    additional_chapters: list[tuple[int, Chapter]],
    additional_extracted_dirs: dict[int, Path] | None,
) -> Section:
    """Merge content from additional chapters into the primary section.

    Searches additional chapters for sections matching the same blueprint.
    If found, merges their text (with source attribution) and combines
    images and tables.
    """
    merged_texts: list[str] = []
    merged_images: list[ImageRef] = list(primary_section.images)
    merged_tables = list(primary_section.tables)
    found_additional = False

    for bk_idx, add_ch in additional_chapters:
        add_section = find_matching_section(add_ch, section_bp)
        if add_section is None:
            continue

        found_additional = True
        merged_texts.append(add_section.text)

        # Resolve additional images to absolute paths
        if add_section.images and additional_extracted_dirs:
            book_dir = additional_extracted_dirs.get(bk_idx)
            if book_dir:
                for img in add_section.images:
                    merged_images.append(ImageRef(
                        path=book_dir / img.path,
                        page=img.page,
                        caption=img.caption,
                        bbox=img.bbox,
                    ))
            else:
                merged_images.extend(add_section.images)
        elif add_section.images:
            merged_images.extend(add_section.images)

        merged_tables.extend(add_section.tables)

    if not found_additional:
        return primary_section

    # Build merged text with clear source attribution
    combined_text = (
        f"=== PRIMARY SOURCE ===\n{primary_section.text}\n\n"
        f"=== ADDITIONAL SOURCE(S) ===\n" + "\n\n".join(merged_texts)
    )

    logger.info(
        "Merged section '%s': primary (%d chars) + %d additional source(s) (%d chars total)",
        primary_section.title,
        len(primary_section.text),
        len(merged_texts),
        len(combined_text),
    )

    return Section(
        title=primary_section.title,
        level=primary_section.level,
        start_page=primary_section.start_page,
        end_page=primary_section.end_page,
        text=combined_text,
        images=tuple(merged_images),
        tables=tuple(merged_tables),
        subsections=primary_section.subsections,
    )


class _SectionContext(NamedTuple):
    """Precomputed context for a single section transformation."""

    prior_titles: list[str]
    cumulative_concepts: list[str | dict]
    section_concepts: list[ConceptEntry]
    section_char: SectionCharacterization | None


def _precompute_section_contexts(
    section_inputs: list[_SectionInput],
    chapter_analysis: ChapterAnalysis | None,
    prior_concepts: list[str | dict] | None,
    canonical_map: dict[str, str] | None = None,
) -> list[_SectionContext]:
    """Precompute all context needed for each section transformation.

    Looks up section analysis once per section and threads cumulative state
    through the list. Everything here is derived from the chapter analysis
    (available before any LLM call), so sections can then run in parallel.

    When canonical_map is provided, concept names are resolved to their
    canonical forms so that within-chapter cumulative concepts and LLM
    prompts use consistent naming.
    """
    contexts: list[_SectionContext] = []
    prior_titles: list[str] = []
    cumulative: list[str | dict] = list(prior_concepts or [])

    for inp in section_inputs:
        section_concepts, section_char = _lookup_section_analysis(
            inp.section.title, chapter_analysis,
            canonical_map=canonical_map,
        )
        filtered = _filter_by_focus(section_concepts, inp.focus_concepts)

        # Snapshot context for this section BEFORE adding its own data
        contexts.append(_SectionContext(
            prior_titles=list(prior_titles),
            cumulative_concepts=list(cumulative),
            section_concepts=filtered,
            section_char=section_char,
        ))
        # Advance for subsequent sections
        prior_titles.append(inp.title)
        cumulative = cumulative + [c.name for c in filtered]

    return contexts


def _parallel_target_selection(
    section_inputs: list[_SectionInput],
    contexts: list[_SectionContext],
    chapter_title: str,
    client: LLMClient,
    max_workers: int,
) -> list[list[ReinforcementTarget] | None]:
    """Run Phase 1 (target selection) in parallel for sections that need it.

    Sections with focus_concepts or short text skip Phase 1 entirely.
    Uses precomputed section_concepts from contexts (no redundant lookups).
    """
    n = len(section_inputs)
    targets: list[list[ReinforcementTarget] | None] = [None] * n

    # Identify which sections need target selection
    to_run: list[tuple[int, _SectionInput, list[ConceptEntry]]] = []
    for i, inp in enumerate(section_inputs):
        if inp.focus_concepts or len(inp.section.text.strip()) < _MIN_TEXT_FOR_TARGETS:
            continue  # Skip — targets[i] stays None
        to_run.append((i, inp, contexts[i].section_concepts))

    if not to_run:
        return targets

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _select_reinforcement_targets,
                inp.section, chapter_title, client, concepts,
                inp.bloom_target,
            ): i
            for i, inp, concepts in to_run
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                targets[idx] = future.result()
            except Exception as exc:
                logger.warning(
                    "Target selection failed for section '%s': %s",
                    section_inputs[idx].title, exc,
                )
                targets[idx] = None  # Non-fatal

    return targets


def _fold_transform_sections(
    section_inputs: list[_SectionInput],
    chapter_title: str,
    client: LLMClient,
    chapter_analysis: ChapterAnalysis | None,
    prior_concepts: list[str | dict] | None,
    module_summary: str | None = None,
    document_type: str | None = None,
    extracted_dir: Path | None = None,
    vision_enabled: bool = False,
    max_workers: int = 4,
    supplementary_context: str | None = None,
    canonical_map: dict[str, str] | None = None,
) -> list[TrainingSection]:
    """Transform sections with parallel LLM calls.

    Precomputes section contexts (prior_titles, cumulative_concepts) from the
    chapter analysis, then runs Phase 1 (target selection) and Phase 2 (content
    generation) in parallel using a thread pool.
    """
    if not section_inputs:
        return []

    # Precompute all section contexts from analysis (no LLM needed)
    contexts = _precompute_section_contexts(
        section_inputs, chapter_analysis, prior_concepts,
        canonical_map=canonical_map,
    )

    # Phase 1: Run target selection in parallel for sections that need it
    targets_list = _parallel_target_selection(
        section_inputs, contexts, chapter_title, client, max_workers,
    )

    # Phase 2: Run content generation in parallel
    def _generate_one(idx: int) -> TrainingSection:
        inp = section_inputs[idx]
        ctx = contexts[idx]
        section_concepts = ctx.section_concepts
        section_char = ctx.section_char

        targets = targets_list[idx]

        if inp.focus_concepts and len(inp.focus_concepts) > 1:
            logger.warning(
                "Section '%s' arrived at content designer with %d focus_concepts "
                "(%s) — expected max 1. The curriculum planner should have split this.",
                inp.title, len(inp.focus_concepts),
                ", ".join(inp.focus_concepts),
            )

        # Extract key terms via regex pre-analyzer (zero LLM cost)
        signals = pre_analyze_section(inp.section.title, inp.section.text)
        section_key_terms = list(signals.key_terms) if signals.key_terms else None

        elements = _transform_section(
            section=inp.section,
            chapter_title=chapter_title,
            client=client,
            template=inp.template,
            prior_sections=ctx.prior_titles,
            learning_objectives=inp.learning_objectives,
            bloom_target=inp.bloom_target,
            section_concepts=section_concepts,
            prior_concepts=ctx.cumulative_concepts,
            section_characterization=section_char,
            precomputed_targets=targets if targets is not None else [],
            module_summary=module_summary,
            section_rationale=inp.rationale,
            focus_concepts=inp.focus_concepts,
            document_type=document_type,
            extracted_dir=extracted_dir,
            vision_enabled=vision_enabled,
            supplementary_context=supplementary_context,
            key_terms=section_key_terms,
        )

        _shuffle_quiz_options(elements)
        verification_notes = _verify_elements(elements, inp.section.text)
        verification_notes.extend(
            _check_cross_references(elements, section_concepts, ctx.cumulative_concepts)
        )

        return TrainingSection(
            title=inp.title,
            source_section_title=inp.section.title,
            source_pages=f"pp. {inp.section.start_page}-{inp.section.end_page}",
            elements=elements,
            verification_notes=verification_notes,
            reinforcement_targets=[t.model_dump() for t in targets] if targets else [],
            learning_objectives=inp.learning_objectives or [],
        )

    n = len(section_inputs)
    results: list[TrainingSection | None] = [None] * n

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_generate_one, i): i for i in range(n)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.error(
                    "Section '%s' transformation failed, using fallback: %s",
                    section_inputs[idx].title, exc,
                )
                inp = section_inputs[idx]
                results[idx] = TrainingSection(
                    title=inp.title,
                    source_section_title=inp.section.title,
                    source_pages=f"pp. {inp.section.start_page}-{inp.section.end_page}",
                    elements=_fallback_section_elements(inp.section),
                    verification_notes=[f"[error] Generation failed: {exc}"],
                    learning_objectives=inp.learning_objectives or [],
                )

    return [r for r in results if r is not None]


def _fallback_section_elements(section: Section) -> list[TrainingElement]:
    """Create a minimal fallback section with one slide when generation fails."""
    truncated = section.text[:2000] + ("..." if len(section.text) > 2000 else "")
    slide = SlideElement(
        element_type="slide",
        bloom_level="understand",
        slide=Slide(
            title=section.title,
            content=f"*Content generation failed for this section. "
            f"The original source text is shown below for reference.*\n\n{truncated}",
            speaker_notes="This is a fallback slide created because LLM generation failed.",
            source_pages=f"pp. {section.start_page}-{section.end_page}",
        ),
    )
    return [slide]


# ── Internal: vision / multimodal helpers ─────────────────────────────────────

# Maximum images to send per section (controls token cost).
MAX_VISION_IMAGES = 5

# Maximum pixel dimension (longest side) for images sent to the LLM.
MAX_IMAGE_DIM = 1024


def _resize_and_encode_image(image_path: Path, max_dim: int = MAX_IMAGE_DIM) -> str | None:
    """Resize an image to fit within max_dim and return a data URI string.

    Returns a "data:image/png;base64,..." string, or None if the file is
    missing or unreadable.
    """
    try:
        from PIL import Image

        if not image_path.exists():
            logger.warning("Vision: image not found: %s", image_path)
            return None

        with Image.open(image_path) as img:
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

            # Resize if either dimension exceeds the limit
            if max(img.size) > max_dim:
                img.thumbnail((max_dim, max_dim), Image.LANCZOS)  # pyright: ignore[reportAttributeAccessIssue] -- Pillow stubs incomplete

            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/png;base64,{b64}"

    except Exception as exc:
        logger.warning("Vision: failed to encode image %s: %s", image_path, exc)
        return None


def _build_multimodal_prompt(
    text_prompt: str,
    images: list,
    extracted_dir: Path,
) -> list[dict]:
    """Build a multimodal content block list from a text prompt and section images.

    Prioritizes images with captions (more likely to be content-relevant),
    caps at MAX_VISION_IMAGES, and encodes each as a base64 data URI.

    Returns a list suitable for the OpenAI messages ``content`` field:
    [{"type": "text", "text": "..."}, {"type": "image_url", ...}, ...]
    """
    # Sort: images with captions first, then by page order
    sorted_images = sorted(
        images,
        key=lambda img: (not getattr(img, "caption", ""), getattr(img, "page", 0)),
    )
    selected = sorted_images[:MAX_VISION_IMAGES]

    blocks: list[dict] = [{"type": "text", "text": text_prompt}]

    for img in selected:
        img_path = extracted_dir / getattr(img, "path", "")
        data_uri = _resize_and_encode_image(img_path)
        if data_uri:
            blocks.append({
                "type": "image_url",
                "image_url": {"url": data_uri},
            })

    if len(blocks) == 1:
        # No images were successfully encoded, fall back to text-only
        return blocks

    logger.debug(
        "Vision: built multimodal prompt with %d image(s) for section",
        len(blocks) - 1,
    )
    return blocks


# ── Internal: section transformation ─────────────────────────────────────────


def _select_reinforcement_targets(
    section: Section,
    chapter_title: str,
    client: LLMClient,
    section_concepts: list[ConceptEntry] | None = None,
    bloom_target: str | None = None,
) -> list[ReinforcementTarget] | None:
    """Phase 1: Identify what's worth testing in this section.

    Returns a list of ReinforcementTarget objects, or None if the call fails.
    Failure is non-fatal — the pipeline falls back to single-phase generation.
    """
    target_prompt = build_target_selection_prompt(
        section_title=section.title,
        section_text=section.text,
        chapter_title=chapter_title,
        section_concepts=section_concepts,
        bloom_target=bloom_target,
    )

    try:
        target_set = client.complete_structured_light(
            TARGET_SELECTION_PROMPT, target_prompt, ReinforcementTargetSet
        )
        logger.debug(
            "Section '%s': %d reinforcement targets identified",
            section.title, len(target_set.targets),
        )
        return target_set.targets

    except (LLMError, ValueError, TypeError, KeyError) as exc:
        logger.warning(
            "Target selection failed for section '%s', falling back to single-phase: %s",
            section.title, exc,
        )
        return None


def _transform_section(
    section: Section,
    chapter_title: str,
    client: LLMClient,
    template: str = "analogy_first",
    prior_sections: list[str] | None = None,
    learning_objectives: list[str] | None = None,
    bloom_target: str | None = None,
    section_concepts: list[ConceptEntry] | None = None,
    prior_concepts: list[str | dict] | None = None,
    section_characterization: SectionCharacterization | None = None,
    precomputed_targets: list[ReinforcementTarget] | None = None,
    module_summary: str | None = None,
    section_rationale: str | None = None,
    focus_concepts: list[str] | None = None,
    document_type: str | None = None,
    extracted_dir: Path | None = None,
    vision_enabled: bool = False,
    supplementary_context: str | None = None,
    key_terms: list[str] | None = None,
) -> list[TrainingElement]:
    """Transform a single section into training elements via LLM.

    Uses a two-phase approach:
    1. Phase 1 — Identify reinforcement targets (what's worth testing).
    2. Phase 2 — Generate elements with explicit targets.

    If Phase 1 fails, falls back to single-phase generation (no targets).
    When precomputed_targets is provided, Phase 1 is skipped.
    """
    # Phase 1: Identify what's worth testing (cheap, short output)
    if precomputed_targets is not None:
        # precomputed_targets=[] means Phase 1 ran and found nothing — use None
        # for the prompt builder (no targets), but do NOT re-run Phase 1.
        targets = precomputed_targets if precomputed_targets else None
    else:
        targets = _select_reinforcement_targets(
            section, chapter_title, client, section_concepts,
        )

    # Phase 2: Generate elements with explicit targets
    user_prompt = build_section_prompt(
        section_title=section.title,
        section_text=section.text,
        chapter_title=chapter_title,
        image_count=len(section.images),
        table_count=len(section.tables),
        template=template,
        source_pages=(section.start_page, section.end_page),
        prior_sections=prior_sections or [],
        learning_objectives=learning_objectives,
        bloom_target=bloom_target,
        section_concepts=section_concepts,
        prior_concepts=prior_concepts,
        section_characterization=section_characterization,
        reinforcement_targets=targets,
        module_summary=module_summary,
        section_rationale=section_rationale,
        focus_concepts=focus_concepts,
        document_type=document_type,
        tables=list(section.tables) if section.tables else None,
        images=list(section.images) if section.images else None,
        supplementary_context=supplementary_context,
        key_terms=key_terms,
    )

    # If vision is enabled and the section has images, build a multimodal prompt
    effective_prompt: str | list[dict] = user_prompt
    if vision_enabled and extracted_dir and section.images:
        effective_prompt = _build_multimodal_prompt(
            user_prompt, list(section.images), extracted_dir,
        )

    # Build bloom-aware system prompt by appending level-specific supplement
    effective_system_prompt = SYSTEM_PROMPT + BLOOM_PROMPT_SUPPLEMENTS.get(
        bloom_target or "understand", ""
    )

    response = client.complete_structured(
        effective_system_prompt, effective_prompt, SectionResponse
    )
    logger.debug(
        "Section '%s': %d elements generated", section.title, len(response.elements)
    )
    return response.elements


def _lookup_section_analysis(
    section_title: str,
    chapter_analysis: ChapterAnalysis | None,
    canonical_map: dict[str, str] | None = None,
) -> tuple[list[ConceptEntry], SectionCharacterization | None]:
    """Look up concepts and characterization for a section from deep reading analysis.

    When canonical_map is provided, resolves each concept's name to its
    canonical form so downstream consumers (prompts, cumulative lists)
    use consistent naming.

    Returns:
        Tuple of (concepts for this section, section characterization or None).
    """
    if chapter_analysis is None:
        return [], None

    # Filter concepts belonging to this section
    section_concepts = [
        c for c in chapter_analysis.concepts
        if c.section_title.lower().strip() == section_title.lower().strip()
    ]

    # Resolve concept names to canonical forms
    if canonical_map and section_concepts:
        section_concepts = [
            c.model_copy(update={"name": resolve_concept(c.name, canonical_map)})
            for c in section_concepts
        ]

    # Find matching section characterization
    section_char = None
    for sc in chapter_analysis.section_characterizations:
        if sc.section_title.lower().strip() == section_title.lower().strip():
            section_char = sc
            break

    return section_concepts, section_char


# ── Internal: quiz answer shuffling ───────────────────────────────────────────
# LLMs overwhelmingly place the correct answer at index 0 ("option A").
# This deterministic post-generation shuffle randomizes option order while
# remapping correct_index and hint_eliminate_index to match.


_thread_local_rng = threading.local()


def _get_rng() -> random.Random:
    """Return a per-thread Random instance (avoids contention on global state)."""
    rng = getattr(_thread_local_rng, "rng", None)
    if rng is None:
        rng = random.Random()
        _thread_local_rng.rng = rng
    return rng


def _shuffle_quiz_options(elements: list[TrainingElement]) -> list[TrainingElement]:
    """Shuffle quiz option order so the correct answer isn't always A.

    Operates in-place on quiz elements. Remaps correct_index and
    hint_eliminate_index to follow the shuffled positions.
    """
    rng = _get_rng()
    for element in elements:
        if element.element_type != "quiz":
            continue
        quiz = getattr(element, "quiz", None)
        if quiz is None:
            continue
        for question in quiz.questions:
            n = len(question.options)
            if n < 2:
                continue

            # Build a shuffled index mapping: old_index -> new_index
            indices = list(range(n))
            rng.shuffle(indices)

            # Reorder options
            new_options = [""] * n
            for old_idx, new_idx in enumerate(indices):
                new_options[new_idx] = question.options[old_idx]

            # Remap correct_index
            new_correct = indices[question.correct_index]

            # Remap hint_eliminate_index
            new_eliminate = question.hint_eliminate_index
            if 0 <= question.hint_eliminate_index < n:
                new_eliminate = indices[question.hint_eliminate_index]

            # Apply (using object.__setattr__ since Pydantic models may be frozen)
            object.__setattr__(question, "options", new_options)
            object.__setattr__(question, "correct_index", new_correct)
            object.__setattr__(question, "hint_eliminate_index", new_eliminate)

    return elements


# ── Internal: post-generation source verification ────────────────────────────
# Rule-based claim extraction + string-similarity matching against source text.
# Zero LLM cost. Flags potential hallucinations as warnings in the intermediate JSON.

# Minimum Jaccard similarity for a claim to be considered "supported" by the source.
_JACCARD_THRESHOLD = 0.6

# Pre-compiled patterns for detecting cross-reference language in generated content.
# Used by _check_cross_references() to verify that sections referencing prior concepts
# actually include appropriate cross-reference phrasing.
_CROSS_REF_PATTERNS = [
    re.compile(p) for p in [
        r"\brecall\b",
        r"\bearlier\b",
        r"\bpreviously\b",
        r"\bas we saw\b",
        r"\bas we discussed\b",
        r"\bwe learned\b",
        r"\bwe covered\b",
        r"\byou already\b",
        r"\bremember\b.*\bfrom\b",
        r"\bjust as\b.*\bearlier\b",
        r"\bsimilar to\b",
        r"\blike the\b.*\bwe\b",
        r"\bbuilds on\b",
        r"\bextending\b.*\bconcept\b",
        r"\bfrom the previous\b",
        r"\bin the last section\b",
    ]
]


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _extract_formulas(text: str) -> list[str]:
    """Extract LaTeX formulas from $...$ and $$...$$ delimiters."""
    # Block math first, then inline
    block = re.findall(r"\$\$(.+?)\$\$", text, re.DOTALL)
    inline = re.findall(r"(?<!\$)\$([^$]+?)\$(?!\$)", text)
    return block + inline


def _extract_numeric_claims(text: str) -> list[str]:
    """Extract numeric assertions (e.g., '15%', '3.14', '100 basis points')."""
    patterns = [
        r"\d+(?:\.\d+)?%",           # percentages
        r"\d+(?:\.\d+)?\s*basis\s*points?",  # basis points
        r"(?:^|[^\w$\\])\d+(?:\.\d+)?(?=[^\w$\\]|$)",  # standalone numbers
    ]
    results = []
    for pattern in patterns:
        results.extend(re.findall(pattern, text))
    return [r.strip() for r in results if len(r.strip()) > 0]


def _extract_definitions(text: str) -> list[str]:
    """Extract definitional claims like 'X is defined as...' or 'X means...'."""
    patterns = [
        r"([A-Z][^.]*?\bis\s+defined\s+as\b[^.]*\.)",
        r"([A-Z][^.]*?\bmeans\b[^.]*\.)",
        r"([A-Z][^.]*?\brefers\s+to\b[^.]*\.)",
    ]
    results = []
    for pattern in patterns:
        results.extend(re.findall(pattern, text))
    return results


def _jaccard_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two normalized strings."""
    words_a = set(_normalize(a).split())
    words_b = set(_normalize(b).split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _check_claim_against_source(claim: str, source_text: str) -> bool:
    """Check if a claim is supported by the source text.

    Uses substring containment and word-level Jaccard similarity.
    For formulas, checks that variable names appear in source.
    """
    norm_claim = _normalize(claim)
    norm_source = _normalize(source_text)

    # Exact substring match
    if norm_claim in norm_source:
        return True

    # Jaccard similarity on word sets
    if _jaccard_similarity(claim, source_text) >= _JACCARD_THRESHOLD:
        return True

    # For formulas: check that variable letters appear in source
    if any(c in claim for c in "\\^_{}"):
        # Extract single-letter variables (not LaTeX commands)
        variables = set(re.findall(r"(?<!\\)\b([a-zA-Z])\b", claim))
        if variables and all(v.lower() in norm_source for v in variables):
            return True

    return False


def _extract_element_text(element: TrainingElement) -> str:
    """Extract verifiable text content from a training element."""
    match element.element_type:
        case "slide":
            s = element.slide  # type: ignore[union-attr]
            return f"{s.title} {s.content}"
        case "quiz":
            q = element.quiz  # type: ignore[union-attr]
            parts = []
            for qq in q.questions:
                parts.append(qq.explanation)
                parts.extend(qq.options)
            return " ".join(parts)
        case "flashcard":
            return element.flashcard.back  # type: ignore[union-attr]
        case "fill_in_the_blank":
            return " ".join(element.fill_in_the_blank.answers)  # type: ignore[union-attr]
        case "interactive_essay":
            return " ".join(element.interactive_essay.concepts_tested)  # type: ignore[union-attr]
        case _:
            return ""


def _check_cross_references(
    elements: list[TrainingElement],
    section_concepts: list[ConceptEntry],
    prior_concepts: list[str | dict] | None,
) -> list[str]:
    """Check whether generated content references prior concepts when it should.

    If this section's concepts overlap with or build on prior concepts, the
    generated text should contain cross-reference language. When it doesn't,
    emit a warning. Pure regex — zero LLM cost.

    Returns a list of warning strings (empty if cross-references are present
    or if there are no prior concepts to reference).
    """
    if not prior_concepts or not section_concepts:
        return []

    # Collect prior concept names (normalized to lowercase)
    prior_names: set[str] = set()
    for pc in prior_concepts:
        if isinstance(pc, dict):
            name = pc.get("name", "")
        else:
            name = str(pc)
        if name:
            prior_names.add(name.lower().strip())

    if not prior_names:
        return []

    # Collect current section concept names
    current_names = {c.name.lower().strip() for c in section_concepts if c.name}

    # Check if any prior concept is related (shared words or substring match)
    has_related_prior = False
    for current in current_names:
        current_words = set(current.split())
        for prior in prior_names:
            prior_words = set(prior.split())
            # Related if they share a meaningful word or one contains the other
            shared = current_words & prior_words - {"of", "the", "a", "an", "and", "in", "for", "to"}
            if shared or prior in current or current in prior:
                has_related_prior = True
                break
        if has_related_prior:
            break

    if not has_related_prior:
        return []

    # Collect all generated text
    all_text = ""
    for element in elements:
        all_text += " " + _extract_element_text(element)
    all_text_lower = all_text.lower()

    # Check for cross-reference language
    has_cross_ref = any(p.search(all_text_lower) for p in _CROSS_REF_PATTERNS)

    # Also check if any prior concept name is explicitly mentioned
    if not has_cross_ref:
        has_cross_ref = any(name in all_text_lower for name in prior_names)

    if not has_cross_ref:
        return [
            "[warning] cross-reference: this section has concepts related to "
            "previously taught material, but the generated content contains no "
            "cross-reference language (e.g., 'recall', 'as we saw', 'you already "
            "understand'). Consider regenerating with stronger cross-reference "
            "instructions."
        ]

    return []


def _verify_elements(
    elements: list[TrainingElement],
    source_text: str,
) -> list[str]:
    """Check generated elements against source material.

    Extracts verifiable claims from elements and checks each against the
    source text. Returns a list of warning strings. Does not modify or
    remove elements — the author makes the final call via the intermediate JSON.
    """
    warnings: list[str] = []

    for element in elements:
        element_text = _extract_element_text(element)
        if not element_text:
            continue

        element_type = getattr(element, "element_type", "unknown")

        # Check formulas
        for formula in _extract_formulas(element_text):
            if not _check_claim_against_source(formula, source_text):
                msg = f"[error] {element_type}: formula '{formula}' not found in source text"
                warnings.append(msg)
                logger.warning("Verification: %s", msg)

        # Check numeric claims
        for number in _extract_numeric_claims(element_text):
            if not _check_claim_against_source(number, source_text):
                msg = f"[warning] {element_type}: numeric claim '{number}' not found in source text"
                warnings.append(msg)
                logger.debug("Verification: %s", msg)

        # Check definitions
        for definition in _extract_definitions(element_text):
            if not _check_claim_against_source(definition, source_text):
                msg = f"[warning] {element_type}: definition claim not verified against source"
                warnings.append(msg)
                logger.debug("Verification: %s", msg)

    return warnings
