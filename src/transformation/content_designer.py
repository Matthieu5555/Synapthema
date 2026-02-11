"""LLM-powered content transformation — the deep module for Stage 2.

Single public entry point: transform_chapter(). Takes a Chapter from Stage 1
and an LLM client, sends each section through the LLM for transformation into
interactive training elements, and returns a complete TrainingModule.

Uses Instructor for schema-constrained LLM output — the model is forced to
return valid JSON matching our Pydantic models, eliminating JSON parse failures.
"""

from __future__ import annotations

import logging
import re
import string
from functools import reduce
from typing import NamedTuple

from collections import Counter

from pydantic import BaseModel, Field, model_validator

from src.extraction.types import Chapter, Section
from src.transformation.analysis_types import (
    ChapterAnalysis,
    ConceptEntry,
    SectionCharacterization,
)
from src.transformation.curriculum_planner import find_matching_section
from src.transformation.llm_client import LLMClient, LLMError
from src.transformation.prompts import (
    BLOOM_PROMPT_SUPPLEMENTS,
    SYSTEM_PROMPT,
    TARGET_SELECTION_PROMPT,
    TEMPLATE_DESCRIPTIONS,
    build_section_prompt,
    build_target_selection_prompt,
)
from src.transformation.types import (
    ELEMENT_BLOOM_MAP,
    ModuleBlueprint,
    ReinforcementTargetSet,
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
        """Enforce minimum/maximum element type counts per section."""
        counts = Counter(e.element_type for e in self.elements)

        if counts.get("slide", 0) < 1:
            raise ValueError("Section must contain at least 1 slide element")
        if counts.get("self_explain", 0) > 2:
            raise ValueError("Section must contain at most 2 self_explain elements")
        if counts.get("interactive_essay", 0) > 1:
            raise ValueError("Section must contain at most 1 interactive_essay element")

        assessment_types = {"quiz", "flashcard", "fill_in_the_blank", "matching", "self_explain"}
        assessment_count = sum(counts.get(t, 0) for t in assessment_types)
        if assessment_count < 1:
            raise ValueError("Section must contain at least 1 assessment element")

        return self


class _SectionInput(NamedTuple):
    """Prepared input for a single section transformation."""

    section: Section
    title: str
    template: str
    learning_objectives: list[str]
    bloom_target: str | None
    rationale: str | None = None


def transform_chapter(
    chapter: Chapter,
    client: LLMClient,
    blueprint: ModuleBlueprint | None = None,
    chapter_analysis: ChapterAnalysis | None = None,
    prior_concepts: list[str | dict] | None = None,
    document_type: str | None = None,
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

    Returns:
        A TrainingModule with sections containing training elements.
    """
    module_title = blueprint.title if blueprint else chapter.title

    logger.info(
        "Transforming chapter %d: '%s' (%d sections%s%s)",
        chapter.chapter_number,
        module_title,
        len(chapter.sections),
        ", with blueprint" if blueprint else "",
        ", with analysis" if chapter_analysis else "",
    )

    # Build a list of section inputs, filtering out unmatched/short sections
    section_inputs = _prepare_section_inputs(chapter, blueprint)

    # Fold: transform each section, threading cumulative concepts
    training_sections = _fold_transform_sections(
        section_inputs, module_title, client, chapter_analysis, prior_concepts,
        module_summary=blueprint.summary if blueprint else None,
        document_type=document_type,
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
) -> list[_SectionInput]:
    """Build the list of section inputs from blueprint or rotation fallback.

    Filters out unmatched and short sections up front.
    """
    inputs: list[_SectionInput] = []

    if blueprint and blueprint.sections:
        for section_bp in blueprint.sections:
            section = find_matching_section(chapter, section_bp)
            if section is None:
                logger.warning(
                    "Blueprint section '%s' not found in chapter, skipping",
                    section_bp.source_section_title or section_bp.title,
                )
                continue
            if len(section.text.strip()) < MIN_SECTION_TEXT_LENGTH:
                logger.debug("Skipping short section: '%s' (%d chars)", section.title, len(section.text))
                continue
            inputs.append(_SectionInput(
                section=section,
                title=section_bp.title,
                template=section_bp.template,
                learning_objectives=section_bp.learning_objectives,
                bloom_target=section_bp.bloom_target,
                rationale=section_bp.rationale or None,
            ))
    else:
        template_idx = 0
        for section in chapter.sections:
            if len(section.text.strip()) < MIN_SECTION_TEXT_LENGTH:
                logger.debug("Skipping short section: '%s' (%d chars)", section.title, len(section.text))
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


def _fold_transform_sections(
    section_inputs: list[_SectionInput],
    chapter_title: str,
    client: LLMClient,
    chapter_analysis: ChapterAnalysis | None,
    prior_concepts: list[str | dict] | None,
    module_summary: str | None = None,
    document_type: str | None = None,
) -> list[TrainingSection]:
    """Transform sections via fold, threading cumulative concepts and titles."""

    def fold(
        state: tuple[list[TrainingSection], list[str], list[str]],
        inp: _SectionInput,
    ) -> tuple[list[TrainingSection], list[str], list[str]]:
        results, prior_titles, cumulative = state

        try:
            section_concepts, section_char = _lookup_section_analysis(
                inp.section.title, chapter_analysis,
            )

            # Phase 1: Identify reinforcement targets before content generation
            targets = _select_reinforcement_targets(
                inp.section, chapter_title, client, section_concepts,
                bloom_target=inp.bloom_target,
            )

            # Pass targets (or empty list if Phase 1 failed) so _transform_section
            # does not re-run Phase 1
            elements = _transform_section(
                section=inp.section,
                chapter_title=chapter_title,
                client=client,
                template=inp.template,
                prior_sections=prior_titles,
                learning_objectives=inp.learning_objectives,
                bloom_target=inp.bloom_target,
                section_concepts=section_concepts,
                prior_concepts=cumulative,
                section_characterization=section_char,
                precomputed_targets=targets if targets is not None else [],
                module_summary=module_summary,
                section_rationale=inp.rationale,
                document_type=document_type,
            )

            # Post-generation verification: check claims against source text
            verification_notes = _verify_elements(elements, inp.section.text)

            new_section = TrainingSection(
                title=inp.title,
                source_pages=f"pp. {inp.section.start_page}-{inp.section.end_page}",
                elements=elements,
                verification_notes=verification_notes,
                reinforcement_targets=[t.model_dump() for t in targets] if targets else [],
                learning_objectives=inp.learning_objectives or [],
            )
        except Exception as exc:
            logger.error(
                "Section '%s' transformation failed, using fallback: %s",
                inp.title, exc,
            )
            elements = _fallback_section_elements(inp.section)
            section_concepts = []
            new_section = TrainingSection(
                title=inp.title,
                source_pages=f"pp. {inp.section.start_page}-{inp.section.end_page}",
                elements=elements,
                verification_notes=[f"[error] Generation failed: {exc}"],
                learning_objectives=inp.learning_objectives or [],
            )

        return (
            results + [new_section],
            prior_titles + [inp.title],
            cumulative + [c.name for c in section_concepts],
        )

    sections, _, _ = reduce(fold, section_inputs, ([], [], list(prior_concepts or [])))
    return sections


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


# ── Internal: section transformation ─────────────────────────────────────────


def _select_reinforcement_targets(
    section: Section,
    chapter_title: str,
    client: LLMClient,
    section_concepts: list[ConceptEntry] | None = None,
    bloom_target: str | None = None,
) -> list[object] | None:
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
    precomputed_targets: list[object] | None = None,
    module_summary: str | None = None,
    section_rationale: str | None = None,
    document_type: str | None = None,
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
        document_type=document_type,
        tables=list(section.tables) if section.tables else None,
        images=list(section.images) if section.images else None,
    )

    # Build bloom-aware system prompt by appending level-specific supplement
    effective_system_prompt = SYSTEM_PROMPT + BLOOM_PROMPT_SUPPLEMENTS.get(
        bloom_target or "understand", ""
    )

    response = client.complete_structured(
        effective_system_prompt, user_prompt, SectionResponse
    )
    logger.debug(
        "Section '%s': %d elements generated", section.title, len(response.elements)
    )
    return response.elements


def _lookup_section_analysis(
    section_title: str,
    chapter_analysis: ChapterAnalysis | None,
) -> tuple[list[ConceptEntry], SectionCharacterization | None]:
    """Look up concepts and characterization for a section from deep reading analysis.

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

    # Find matching section characterization
    section_char = None
    for sc in chapter_analysis.section_characterizations:
        if sc.section_title.lower().strip() == section_title.lower().strip():
            section_char = sc
            break

    return section_concepts, section_char


# ── Internal: post-generation source verification ────────────────────────────
# Rule-based claim extraction + string-similarity matching against source text.
# Zero LLM cost. Flags potential hallucinations as warnings in the intermediate JSON.

# Minimum Jaccard similarity for a claim to be considered "supported" by the source.
_JACCARD_THRESHOLD = 0.6


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


def _extract_element_text(element: TrainingElement) -> str:  # type: ignore[return]
    """Extract verifiable text content from a training element."""
    match element:
        case _ if hasattr(element, "slide"):
            slide = element.slide  # type: ignore[union-attr]
            return f"{slide.title} {slide.content}"
        case _ if hasattr(element, "quiz"):
            quiz = element.quiz  # type: ignore[union-attr]
            parts = []
            for q in quiz.questions:
                parts.append(q.explanation)
                parts.extend(q.options)
            return " ".join(parts)
        case _ if hasattr(element, "flashcard"):
            fc = element.flashcard  # type: ignore[union-attr]
            return fc.back
        case _ if hasattr(element, "fill_in_the_blank"):
            fib = element.fill_in_the_blank  # type: ignore[union-attr]
            return " ".join(fib.answers)
        case _ if hasattr(element, "self_explain"):
            se = element.self_explain  # type: ignore[union-attr]
            return " ".join(se.key_points)
        case _ if hasattr(element, "interactive_essay"):
            ie = element.interactive_essay  # type: ignore[union-attr]
            return " ".join(ie.concepts_tested)
        case _:
            return ""


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
