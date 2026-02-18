"""HTML rendering — the deep module for Stage 3.

Single public entry point: render_course(). Takes a sequence of TrainingModules
from Stage 2 and produces self-contained HTML files — one per chapter plus
an index page. All CSS and JS are inlined for zero-dependency output.

Supports all 13 element types: section intros, slides, mermaid diagrams, quizzes,
flashcards, fill-in-the-blank, matching, ordering, categorization, error detection,
analogy, concept maps, and interactive essays. Includes KaTeX for math rendering,
Bloom's Taxonomy badges, markdown rendering, and currency/LaTeX-aware fill-in-the-blank.
"""

from __future__ import annotations

import base64
import json
import logging
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import markdown as md
from jinja2 import Environment, FileSystemLoader

from functools import reduce

from src.transformation.analysis_types import ChapterAnalysis, ConceptGraph
from src.transformation.types import (
    AnalogyElement,
    CategorizationElement,
    ConceptMapElement,
    ErrorDetectionElement,
    FillInBlankElement,
    FlashcardElement,
    InteractiveEssayElement,
    MatchingElement,
    MermaidElement,
    OrderingElement,
    QuizElement,
    SectionIntroElement,
    SlideElement,
    TrainingModule,
    TrainingSection,
)

logger = logging.getLogger(__name__)

# Directory containing the Jinja2 HTML/CSS templates.
_TEMPLATES_DIR = Path(__file__).parent / "templates"

def render_course(
    modules: Sequence[TrainingModule],
    output_dir: Path,
    extracted_dir: Path | None = None,
    per_module_extracted_dirs: Sequence[Path] | None = None,
    embed_images: bool = True,
    concept_graph: ConceptGraph | None = None,
    chapter_analyses: list[ChapterAnalysis] | None = None,
    course_title: str | None = None,
    course_summary: str | None = None,
    learner_journey: str | None = None,
    source_book_titles: Sequence[str] | None = None,
    chapter_to_module: dict[int, int] | None = None,
) -> Path:
    """Render training modules as a self-contained interactive HTML course.

    Generates one HTML file per chapter module, plus an index.html landing
    page linking to all chapters. CSS and JS are embedded inline in each file.
    Chapters include cross-navigation (prev/next), theme toggle, progress
    tracking, concept graph visualization, and learner model integration.

    Args:
        modules: Sequence of TrainingModules from Stage 2 (one per chapter).
        output_dir: Directory to write HTML files into. Created if missing.
        extracted_dir: Directory containing extracted images (from Stage 1).
            Required if embed_images is True.
        embed_images: If True, images are base64-encoded inline in the HTML.
            If False, images are copied to output_dir and referenced by path.
        concept_graph: Consolidated concept dependency graph for visualization.
        chapter_analyses: Deep reading analyses for concept-to-element tagging.

    Returns:
        Path to the generated index.html file.
    """
    if not chapter_analyses:
        logger.warning(
            "render_course() called without chapter_analyses — "
            "learner model, mastery dashboard, and concept-based review will be inert"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )

    css = (_TEMPLATES_DIR / "styles.css").read_text(encoding="utf-8")
    graphlib_js = (_TEMPLATES_DIR / "graphlib.min.js").read_text(encoding="utf-8")
    dagre_js = (_TEMPLATES_DIR / "dagre.min.js").read_text(encoding="utf-8")

    course_title = course_title or _derive_course_title(modules)
    # Derive slug from dir name. If output_dir is "html" (nested under the
    # course root), use the parent directory name instead.
    course_slug = output_dir.parent.name if output_dir.name == "html" else output_dir.name

    # Load course_meta.json override (takes priority over blueprint values)
    meta_override = _load_course_meta(output_dir)
    if meta_override:
        course_title = meta_override.get("course_title", course_title)
        course_summary = meta_override.get("course_summary", course_summary)
        learner_journey = meta_override.get("learner_journey", learner_journey)
        logger.info("Applied course_meta.json overrides")

    # Build chapter analysis lookup by chapter number
    analyses_by_chapter: dict[int, ChapterAnalysis] = {}
    if chapter_analyses:
        for analysis in chapter_analyses:
            analyses_by_chapter[analysis.chapter_number] = analysis

    # Determine multi-doc status for source book attribution
    is_multi_doc = bool(source_book_titles and len(set(source_book_titles)) > 1)

    # First pass: build chapter metadata for cross-navigation.
    # Use sequential 1-based module index for file naming and identity —
    # source chapter_number is metadata, not identity (multiple modules
    # can share the same source chapter number in multi-book courses).
    chapter_info: list[dict] = [
        {
            "number": i + 1,
            "title": m.title,
            "filename": f"chapter_{i + 1:02d}.html",
            "element_count": len(m.all_elements),
            "section_count": len(m.sections),
            "source_book_title": source_book_titles[i] if source_book_titles and i < len(source_book_titles) else "",
        }
        for i, m in enumerate(modules)
    ]

    # Second pass: render chapters with full navigation context
    for i, module in enumerate(modules):
        html_path = output_dir / chapter_info[i]["filename"]
        prev_chapter = chapter_info[i - 1] if i > 0 else None
        next_chapter = chapter_info[i + 1] if i < len(modules) - 1 else None

        # Use per-module extracted dir (multi-doc) or fall back to global
        effective_extracted_dir = (
            per_module_extracted_dirs[i]
            if per_module_extracted_dirs and i < len(per_module_extracted_dirs)
            else extracted_dir
        )

        _render_chapter(
            module=module,
            env=env,
            css=css,
            graphlib_js=graphlib_js,
            dagre_js=dagre_js,
            output_path=html_path,
            extracted_dir=effective_extracted_dir,
            embed_images=embed_images,
            course_title=course_title,
            course_slug=course_slug,
            chapter_count=len(modules),
            prev_chapter=prev_chapter,
            next_chapter=next_chapter,
            chapter_analysis=analyses_by_chapter.get(module.chapter_number),
            concept_graph=concept_graph,
            source_book_title=chapter_info[i]["source_book_title"] if is_multi_doc else "",
            module_number=i + 1,
        )

        logger.info(
            "Rendered chapter %d: %s (%d elements, %d sections)",
            i + 1,
            chapter_info[i]["filename"],
            len(module.all_elements),
            len(module.sections),
        )

    # Prepare concept graph data for the index page
    graph_data = None
    mindmap_data = None
    if concept_graph and concept_graph.concepts:
        graph_data = _prepare_graph_data(concept_graph, modules, chapter_to_module)
        mindmap_data = _prepare_mindmap_data(concept_graph, modules, chapter_to_module)

    subtitle = f"Interactive Training Course \u2014 {len(modules)} Chapters"
    if meta_override and "subtitle" in meta_override:
        subtitle = meta_override["subtitle"]

    index_path = output_dir / "index.html"
    _render_index(
        env=env,
        css=css,
        chapters=chapter_info,
        course_title=course_title,
        course_slug=course_slug,
        output_path=index_path,
        graph_data=graph_data,
        mindmap_data=mindmap_data,
        course_summary=course_summary,
        learner_journey=learner_journey,
        subtitle=subtitle,
    )

    _render_review_pages(
        env=env,
        css=css,
        course_title=course_title,
        course_slug=course_slug,
        chapters=chapter_info,
        output_dir=output_dir,
    )

    # Write course_meta.json for user editing (only if it doesn't exist)
    _write_course_meta(output_dir, course_title, course_summary, learner_journey, subtitle)

    logger.info("Course rendered: %d chapters + index at %s", len(modules), output_dir)
    return index_path


# ── Internal: chapter rendering ──────────────────────────────────────────────


def _render_chapter(
    module: TrainingModule,
    env: Environment,
    css: str,
    graphlib_js: str,
    dagre_js: str,
    output_path: Path,
    extracted_dir: Path | None,
    embed_images: bool,
    course_title: str = "",
    course_slug: str = "",
    chapter_count: int = 1,
    prev_chapter: dict | None = None,
    next_chapter: dict | None = None,
    chapter_analysis: ChapterAnalysis | None = None,
    concept_graph: ConceptGraph | None = None,
    source_book_title: str = "",
    module_number: int | None = None,
) -> None:
    """Render a single chapter's training module to an HTML file."""
    template = env.get_template("base.html")

    # Use sequential module number for identity; fall back to source
    # chapter_number for backwards compatibility with direct callers.
    effective_number = module_number if module_number is not None else module.chapter_number

    # Build section-grouped data for the sidebar and content via fold
    sections_data, flat_elements = _build_sections_data(
        module, extracted_dir, embed_images,
        chapter_analysis=chapter_analysis,
        concept_graph=concept_graph,
        module_number=effective_number,
    )

    html = template.render(
        module_title=module.title,
        css=css,
        graphlib_js=graphlib_js,
        dagre_js=dagre_js,
        elements=flat_elements,
        sections=sections_data,
        course_title=course_title,
        course_slug=course_slug,
        chapter_number=effective_number,
        chapter_count=chapter_count,
        prev_chapter=prev_chapter,
        next_chapter=next_chapter,
        source_book_title=source_book_title,
    )

    output_path.write_text(html, encoding="utf-8")


def _build_sections_data(
    module: TrainingModule,
    extracted_dir: Path | None,
    embed_images: bool,
    chapter_analysis: ChapterAnalysis | None = None,
    concept_graph: ConceptGraph | None = None,
    module_number: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Build section metadata and flat element list via fold.

    Returns (sections_data, flat_elements) where each section tracks its
    start_index into the flat list. Each element is tagged with a
    deterministic element_id (for FSRS tracking) and concepts_tested
    (for the learner model).
    """
    # Use sequential module number for element IDs; fall back to source
    # chapter_number for backwards compatibility.
    effective_number = module_number if module_number is not None else module.chapter_number

    # Build section title → concept names mapping from deep reading analysis
    section_concepts: dict[str, list[str]] = {}
    if chapter_analysis:
        for concept in chapter_analysis.concepts:
            resolved_name = concept_graph.resolve(concept.name) if concept_graph else concept.name
            section_concepts.setdefault(concept.section_title, []).append(
                resolved_name
            )

    def fold(
        state: tuple[list[dict], list[dict], int, int],
        section: TrainingSection,
    ) -> tuple[list[dict], list[dict], int, int]:
        sections_so_far, flat_so_far, offset, section_index = state
        # Use source_section_title for concept lookup (handles split units
        # where title differs from the original extraction section title).
        lookup_title = getattr(section, "source_section_title", "") or section.title
        concepts = section_concepts.get(lookup_title, [])
        reinf_targets = section.reinforcement_targets if hasattr(section, "reinforcement_targets") else None
        prepared_elements = []
        for i, e in enumerate(section.elements):
            prepared = _prepare_element(e, extracted_dir, embed_images)
            element_dict = {
                **prepared,
                "element_id": _element_id(
                    effective_number, section_index, i,
                ),
                "concepts_tested": _tag_element_concepts(prepared, concepts, reinf_targets),
            }
            prepared_elements.append(element_dict)
        # Element ordering is owned by content_designer (SectionResponse validator).
        # The renderer trusts upstream ordering and does not re-sort.
        section_data = {
            "title": section.title,
            "source_pages": section.source_pages,
            "element_count": len(prepared_elements),
            "start_index": offset,
            "verification_notes": section.verification_notes,
            "learning_objectives": section.learning_objectives if hasattr(section, "learning_objectives") else [],
        }
        return (
            sections_so_far + [section_data],
            flat_so_far + prepared_elements,
            offset + len(prepared_elements),
            section_index + 1,
        )

    sections_data, flat_elements, _, _ = reduce(
        fold, module.sections, ([], [], 0, 0),
    )
    return sections_data, flat_elements


def _fix_unicode_escapes(text: str) -> str:
    r"""Decode literal \uXXXX sequences to actual Unicode characters.

    LLM responses sometimes contain raw escape sequences (e.g. ``\\u0394``)
    that survive JSON round-tripping as literal backslash-u text rather than
    the intended Unicode character (Δ).  Mermaid 11+ rejects these as syntax
    errors.
    """
    return re.sub(
        r"\\u([0-9a-fA-F]{4})",
        lambda m: chr(int(m.group(1), 16)),
        text,
    )


def _prep_section_intro(elem: SectionIntroElement, _ed: Path | None, _ei: bool) -> dict:
    si = elem.section_intro
    return {"section_intro": {
        "title": si.title,
        "content_html": _markdown_to_html(si.content),
        "source_pages": si.source_pages,
    }}


def _prep_slide(elem: SlideElement, extracted_dir: Path | None, embed_images: bool) -> dict:
    s = elem.slide
    image_data = None
    if s.image_path and extracted_dir and embed_images:
        image_data = _encode_image_base64(extracted_dir / Path(s.image_path))
    return {"slide": {
        "title": s.title,
        "content_html": _markdown_to_html(s.content),
        "speaker_notes": s.speaker_notes,
        "image_data": image_data,
        "source_pages": s.source_pages,
    }}


def _prep_quiz(elem: QuizElement, _ed: Path | None, _ei: bool) -> dict:
    q = elem.quiz
    return {"quiz": {
        "title": q.title,
        "questions": [
            {
                "question": _markdown_to_html_inline(qq.question),
                "options": [_markdown_to_html_inline(opt) for opt in qq.options],
                "correct_index": qq.correct_index,
                "explanation": _markdown_to_html(qq.explanation),
                "hint_metacognitive": _markdown_to_html_inline(qq.hint_metacognitive) if qq.hint_metacognitive else "",
                "hint_strategic": _markdown_to_html_inline(qq.hint_strategic) if qq.hint_strategic else "",
                "hint_eliminate_index": qq.hint_eliminate_index,
            }
            for qq in q.questions
        ],
    }}


def _prep_flashcard(elem: FlashcardElement, _ed: Path | None, _ei: bool) -> dict:
    f = elem.flashcard
    return {"flashcard": {
        "front": _markdown_to_html(f.front),
        "back": _markdown_to_html(f.back),
    }}


def _prep_fill_in_blank(elem: FillInBlankElement, _ed: Path | None, _ei: bool) -> dict:
    fitb = elem.fill_in_the_blank
    interactive_indices = _fitb_interactive_answer_indices(fitb.statement)
    interactive_answers = [
        fitb.answers[i] for i in interactive_indices if i < len(fitb.answers)
    ]
    return {"fill_in_the_blank": {
        "statement_html": _render_fitb_statement(fitb.statement, fitb.answers),
        "answers_json": json.dumps(interactive_answers),
        "hint": _markdown_to_html_inline(fitb.hint) if fitb.hint else "",
        "hint_first_letter": fitb.hint_first_letter,
    }}


def _prep_matching(elem: MatchingElement, _ed: Path | None, _ei: bool) -> dict:
    m = elem.matching
    left = list(m.left_items)
    right = list(m.right_items)
    rng = random.Random(m.title)
    shuffled_indices = list(range(len(right)))
    rng.shuffle(shuffled_indices)
    shuffled_right = [right[i] for i in shuffled_indices]
    correct_map = {orig: shuffled_indices.index(orig) for orig in range(len(left))}
    return {"matching": {
        "title": m.title,
        "left_items": [_markdown_to_html_inline(item) for item in left],
        "shuffled_right": [_markdown_to_html_inline(item) for item in shuffled_right],
        "correct_json": json.dumps(correct_map),
        "pair_explanations_json": json.dumps([_markdown_to_html_inline(e) if e else "" for e in m.pair_explanations]),
    }}


def _prep_ordering(elem: OrderingElement, _ed: Path | None, _ei: bool) -> dict:
    o = elem.ordering
    correct_order = list(o.items)
    rng = random.Random(o.title)
    shuffled = list(range(len(correct_order)))
    rng.shuffle(shuffled)
    return {"ordering": {
        "title": o.title,
        "instruction": _markdown_to_html_inline(o.instruction) if o.instruction else "",
        "shuffled_items": [_markdown_to_html_inline(correct_order[i]) for i in shuffled],
        "correct_order_json": json.dumps([shuffled.index(i) for i in range(len(correct_order))]),
        "explanation": _markdown_to_html(o.explanation),
        "hint": _markdown_to_html_inline(o.hint) if o.hint else "",
    }}


def _prep_categorization(elem: CategorizationElement, _ed: Path | None, _ei: bool) -> dict:
    cat = elem.categorization
    all_items = []
    for cat_idx, bucket in enumerate(cat.categories):
        for item in bucket.items:
            all_items.append({"text": _markdown_to_html_inline(item), "correct_cat": cat_idx})
    rng = random.Random(cat.title)
    rng.shuffle(all_items)
    return {"categorization": {
        "title": cat.title,
        "instruction": _markdown_to_html_inline(cat.instruction) if cat.instruction else "",
        "category_names": [_markdown_to_html_inline(b.name) if b.name else "" for b in cat.categories],
        "items_json": json.dumps(all_items),
        "explanation": _markdown_to_html(cat.explanation),
        "hint": _markdown_to_html_inline(cat.hint) if cat.hint else "",
    }}


def _prep_error_detection(elem: ErrorDetectionElement, _ed: Path | None, _ei: bool) -> dict:
    ed = elem.error_detection
    return {"error_detection": {
        "title": ed.title,
        "instruction": _markdown_to_html_inline(ed.instruction) if ed.instruction else "",
        "error_items": [
            {
                "statement": _markdown_to_html_inline(item.statement),
                "error_explanation": _markdown_to_html(item.error_explanation),
                "corrected_statement": _markdown_to_html_inline(item.corrected_statement),
            }
            for item in ed.items
        ],
        "context": _markdown_to_html_inline(ed.context) if ed.context else "",
    }}


def _prep_analogy(elem: AnalogyElement, _ed: Path | None, _ei: bool) -> dict:
    a = elem.analogy
    prepared_items = []
    for item in a.items:
        options = [item.answer] + list(item.distractors)
        rng = random.Random(item.stem)
        rng.shuffle(options)
        correct_idx = options.index(item.answer)
        prepared_items.append({
            "stem": _markdown_to_html_inline(item.stem),
            "options": [_markdown_to_html_inline(opt) for opt in options],
            "correct_index": correct_idx,
            "explanation": _markdown_to_html(item.explanation),
        })
    return {"analogy": {
        "title": a.title,
        "items_json": json.dumps(prepared_items),
    }}


def _prep_mermaid(elem: MermaidElement, _ed: Path | None, _ei: bool) -> dict:
    d = elem.mermaid
    return {"mermaid": {
        "title": d.title,
        "diagram_code": _fix_unicode_escapes(d.diagram_code),
        "caption": _markdown_to_html_inline(d.caption) if d.caption else "",
        "diagram_type": d.diagram_type,
    }}


def _prep_concept_map(elem: ConceptMapElement, _ed: Path | None, _ei: bool) -> dict:
    cmap = elem.concept_map
    return {"concept_map": {
        "title": cmap.title,
        "nodes_json": json.dumps([n.model_dump() for n in cmap.nodes]),
        "edges_json": json.dumps([e.model_dump() for e in cmap.edges]),
        "blank_indices_json": json.dumps(cmap.blank_edge_indices),
    }}


def _prep_interactive_essay(elem: InteractiveEssayElement, _ed: Path | None, _ei: bool) -> dict:
    ie = elem.interactive_essay
    is_static = len(ie.prompts) == 1 and not ie.tutor_system_prompt
    return {"interactive_essay": {
        "title": ie.title,
        "concepts_tested": ie.concepts_tested,
        "prompts": [
            {
                "prompt": _markdown_to_html_inline(p.prompt),
                "key_points_json": json.dumps([_markdown_to_html_inline(kp) if kp else "" for kp in p.key_points]),
                "example_response": _markdown_to_html(p.example_response),
                "minimum_words": p.minimum_words,
                "source_pages": getattr(p, "source_pages", ""),
            }
            for p in ie.prompts
        ],
        "passing_threshold": ie.passing_threshold,
        "tutor_system_prompt_json": json.dumps(ie.tutor_system_prompt),
        "is_static": is_static,
    }}


# Dispatch table: element_type → preparer function.
_ELEMENT_PREPARERS: dict[str, callable] = {
    "section_intro": _prep_section_intro,
    "slide": _prep_slide,
    "quiz": _prep_quiz,
    "flashcard": _prep_flashcard,
    "fill_in_the_blank": _prep_fill_in_blank,
    "matching": _prep_matching,
    "ordering": _prep_ordering,
    "categorization": _prep_categorization,
    "error_detection": _prep_error_detection,
    "analogy": _prep_analogy,
    "mermaid": _prep_mermaid,
    "concept_map": _prep_concept_map,
    "interactive_essay": _prep_interactive_essay,
}


def _prepare_element(
    elem: SectionIntroElement | SlideElement | QuizElement | FlashcardElement
    | FillInBlankElement | MatchingElement | OrderingElement | MermaidElement
    | ConceptMapElement | CategorizationElement | ErrorDetectionElement
    | AnalogyElement | InteractiveEssayElement,
    extracted_dir: Path | None,
    embed_images: bool,
) -> dict:
    """Convert a TrainingElement variant into a template-friendly dict.

    Dispatches to a type-specific preparer function via _ELEMENT_PREPARERS.
    """
    result: dict = {
        "element_type": elem.element_type,
        "bloom_level": elem.bloom_level,
    }
    preparer = _ELEMENT_PREPARERS.get(elem.element_type)
    if preparer:
        result.update(preparer(elem, extracted_dir, embed_images))
    return _fix_unicode_escapes_deep(result)  # type: ignore[return-value]


def _fix_unicode_escapes_deep(obj: object) -> object:
    """Recursively decode literal \\uXXXX in all string values."""
    if isinstance(obj, str):
        return _fix_unicode_escapes(obj)
    if isinstance(obj, list):
        return [_fix_unicode_escapes_deep(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _fix_unicode_escapes_deep(v) for k, v in obj.items()}
    return obj


# ── Internal: element-level concept tagging ───────────────────────────────────


def _extract_text_values(obj: dict | list | str) -> list[str]:
    """Recursively extract string values from nested dicts/lists."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, list):
        return [s for item in obj for s in _extract_text_values(item)]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _extract_text_values(v)]
    return []


def _tag_element_concepts(
    element: dict,
    section_concepts: list[str],
    reinforcement_targets: list[dict] | None = None,
) -> list[str]:
    """Tag a single element with specific concepts using keyword overlap."""
    if not section_concepts:
        return []

    # Extract text content from the element for matching
    text_parts: list[str] = []
    for key in ("section_intro", "slide", "quiz", "flashcard", "fill_in_the_blank",
                "matching", "concept_map", "interactive_essay", "mermaid"):
        sub = element.get(key)
        if sub:
            # Collect all string values recursively
            text_parts.extend(_extract_text_values(sub))

    element_text = " ".join(text_parts).lower()
    if not element_text:
        return section_concepts  # fallback to section-level

    matched = [c for c in section_concepts if c.lower() in element_text]
    return matched if matched else section_concepts  # fallback if no matches


# ── Internal: review page rendering ──────────────────────────────────────────


def _render_review_pages(
    env: Environment,
    css: str,
    course_title: str,
    course_slug: str,
    chapters: list[dict],
    output_dir: Path,
) -> None:
    """Render the FSRS review and mixed practice pages."""
    for template_name in ("review.html", "mixed_review.html"):
        tmpl = env.get_template(template_name)
        html = tmpl.render(
            course_title=course_title,
            course_slug=course_slug,
            css=css,
            chapters=chapters,
        )
        (output_dir / template_name).write_text(html, encoding="utf-8")

    logger.info("Rendered review.html and mixed_review.html")


# ── Internal: index rendering ────────────────────────────────────────────────


def _render_index(
    env: Environment,
    css: str,
    chapters: list[dict],
    course_title: str,
    course_slug: str,
    output_path: Path,
    graph_data: dict | None = None,
    mindmap_data: dict | None = None,
    course_summary: str | None = None,
    learner_journey: str | None = None,
    subtitle: str = "",
) -> None:
    """Render the course index/landing page."""
    template = env.get_template("index.html")
    html = template.render(
        course_title=course_title,
        course_slug=course_slug,
        css=css,
        chapters=chapters,
        graph_data=graph_data,
        mindmap_data=mindmap_data,
        course_summary=course_summary,
        learner_journey=learner_journey,
        subtitle=subtitle,
    )
    output_path.write_text(html, encoding="utf-8")


# ── Internal: course metadata override ──────────────────────────────────────


def _load_course_meta(output_dir: Path) -> dict[str, str] | None:
    """Load course_meta.json from output directory.  Returns None if missing."""
    meta_path = output_dir / "course_meta.json"
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        logger.warning("Invalid course_meta.json, ignoring")
        return None


def _write_course_meta(
    output_dir: Path,
    course_title: str,
    course_summary: str | None,
    learner_journey: str | None,
    subtitle: str,
) -> None:
    """Write course_meta.json to output directory (only if it doesn't exist)."""
    meta_path = output_dir / "course_meta.json"
    if meta_path.exists():
        return
    data = {
        "course_title": course_title,
        "course_summary": course_summary or "",
        "learner_journey": learner_journey or "",
        "subtitle": subtitle,
    }
    meta_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote course_meta.json to %s", meta_path)


# ── Internal: element ID generation ─────────────────────────────────────────


def _element_id(chapter_number: int, section_index: int, element_index: int) -> str:
    """Generate a deterministic, stable element ID for FSRS tracking.

    The ID encodes chapter, section, and element position so it survives
    re-renders as long as the source content order is unchanged.
    """
    return f"card_ch{chapter_number:02d}_s{section_index:02d}_e{element_index:02d}"


# ── Internal: concept graph visualization ───────────────────────────────────


# Curated palette — distinct, accessible colors for up to ~24 chapters.
_CHAPTER_PALETTE = [
    "#6366f1",  # indigo
    "#f43f5e",  # rose
    "#10b981",  # emerald
    "#f59e0b",  # amber
    "#3b82f6",  # blue
    "#ec4899",  # pink
    "#14b8a6",  # teal
    "#f97316",  # orange
    "#8b5cf6",  # violet
    "#22c55e",  # green
    "#ef4444",  # red
    "#06b6d4",  # cyan
    "#eab308",  # yellow
    "#a855f7",  # purple
    "#0ea5e9",  # sky
    "#d946ef",  # fuchsia
    "#84cc16",  # lime
    "#e11d48",  # crimson
    "#2dd4bf",  # aqua
    "#fb923c",  # tangerine
    "#818cf8",  # periwinkle
    "#4ade80",  # mint
    "#f472b6",  # bubblegum
    "#38bdf8",  # cornflower
]

# Maximum nodes to show in the knowledge map to keep it readable.
_MAX_GRAPH_NODES = 60


def _generate_chapter_colors(count: int) -> dict[int, str]:
    """Map chapter numbers to distinct colors from a curated palette."""
    return {
        i + 1: _CHAPTER_PALETTE[i % len(_CHAPTER_PALETTE)]
        for i in range(count)
    }


def _truncate_label(label: str, max_len: int = 28) -> str:
    """Shorten a node label for graph display."""
    if len(label) <= max_len:
        return label
    return label[: max_len - 1].rstrip() + "\u2026"


def _prepare_graph_data(
    concept_graph: ConceptGraph,
    modules: Sequence[TrainingModule],
    chapter_to_module: dict[int, int] | None = None,
) -> dict:
    """Convert ConceptGraph to vis-network compatible JSON.

    Selects the most cross-referenced concepts (up to _MAX_GRAPH_NODES),
    truncates long labels, and uses a curated color palette.

    When *chapter_to_module* is provided (multi-doc mode), concept chapter
    numbers are remapped to sequential module indices so colors and groups
    align with the rendered modules.
    """
    chapter_colors = _generate_chapter_colors(len(modules))

    def _remap(ch: int) -> int:
        if chapter_to_module is not None:
            return chapter_to_module.get(ch, 0)
        return ch

    # Rank concepts by how many chapters reference them (most connected first)
    ranked = sorted(
        concept_graph.concepts,
        key=lambda c: len(c.mentioned_in_chapters),
        reverse=True,
    )
    top_concepts = ranked[:_MAX_GRAPH_NODES]
    top_ids = {c.canonical_name for c in top_concepts}

    nodes = [
        {
            "id": c.canonical_name,
            "label": _truncate_label(c.canonical_name),
            "title": f"<b>{c.canonical_name}</b><br>{c.definition}" if c.definition else c.canonical_name,
            "group": _remap(c.first_introduced_chapter),
            "color": chapter_colors.get(_remap(c.first_introduced_chapter), "#999"),
            "size": 8 + 4 * len(c.mentioned_in_chapters),
        }
        for c in top_concepts
    ]

    edges = [
        {
            "from": e.source,
            "to": e.target,
            "arrows": "to",
        }
        for e in concept_graph.edges
        if e.source in top_ids and e.target in top_ids
    ]

    return {"nodes": nodes, "edges": edges}


# Maximum concepts shown per chapter in the mind-map tree.
_MAX_CONCEPTS_PER_CHAPTER = 8

# Edge types that imply a hierarchical (parent → child) relationship.
_HIERARCHICAL_EDGE_TYPES = {"builds_on", "requires"}


def _prepare_mindmap_data(
    concept_graph: ConceptGraph,
    modules: Sequence[TrainingModule],
    chapter_to_module: dict[int, int] | None = None,
) -> dict:
    """Build a chapter-grouped collapsible tree from the concept graph.

    First level: chapters (one node per module).  Second level: the most
    important concepts introduced in that chapter.  Third+ levels: concepts
    that ``builds_on`` / ``requires`` a parent concept (prerequisite
    nesting).  The result is rendered as a collapsible tree on the index
    page — similar to a NotebookLM mind map.

    When *chapter_to_module* is provided (multi-doc mode), concept chapter
    numbers are remapped to sequential module indices.  Concepts from
    source chapters that don't map to any module are excluded.
    """
    if not concept_graph.concepts:
        return {}

    chapter_colors = _generate_chapter_colors(len(modules))
    concept_lookup = {c.canonical_name: c for c in concept_graph.concepts}

    def _remap(ch: int) -> int:
        if chapter_to_module is not None:
            return chapter_to_module.get(ch, 0)
        return ch

    # Build prerequisite adjacency among ALL concepts (not just top N).
    # children_of[prerequisite] = [dependent, ...].
    children_of: dict[str, list[str]] = defaultdict(list)
    parent_of: dict[str, list[str]] = defaultdict(list)
    for e in concept_graph.edges:
        if e.relationship in _HIERARCHICAL_EDGE_TYPES:
            children_of[e.target].append(e.source)
            parent_of[e.source].append(e.target)

    # Group concepts by their remapped chapter (module index).
    # In multi-doc mode, concepts from unmapped chapters (remap → 0) are excluded.
    by_chapter: dict[int, list] = defaultdict(list)
    for c in concept_graph.concepts:
        mapped = _remap(c.first_introduced_chapter)
        if mapped == 0:
            continue
        by_chapter[mapped].append(c)

    # Sort each chapter's concepts by importance (cross-chapter mentions).
    for ch_concepts in by_chapter.values():
        ch_concepts.sort(
            key=lambda c: len(c.mentioned_in_chapters), reverse=True
        )

    # Track which concepts have been placed in the tree (DAG → tree).
    assigned: set[str] = set()

    def _build_concept_subtree(
        name: str, depth: int = 0,
    ) -> dict | None:
        if name in assigned or depth > 4:
            return None
        c = concept_lookup.get(name)
        if not c:
            return None
        assigned.add(name)

        kids: list[dict] = []
        for child_name in children_of.get(name, []):
            if child_name not in assigned:
                subtree = _build_concept_subtree(child_name, depth + 1)
                if subtree:
                    kids.append(subtree)
        kids.sort(
            key=lambda k: k.get("importance", 0), reverse=True
        )

        mod_idx = _remap(c.first_introduced_chapter)
        return {
            "name": c.canonical_name,
            "definition": c.definition or "",
            "chapter": mod_idx,
            "color": chapter_colors.get(mod_idx, "#999"),
            "importance": len(c.mentioned_in_chapters),
            "children": kids,
        }

    # Build the tree: one root per module that has concepts.
    roots: list[dict] = []
    # Use sequential module index for titles (works for both single- and multi-doc).
    if chapter_to_module is not None:
        module_titles = {i + 1: m.title for i, m in enumerate(modules)}
    else:
        module_titles = {m.chapter_number: m.title for m in modules}

    for ch_num in sorted(by_chapter.keys()):
        ch_concepts = by_chapter[ch_num]
        color = chapter_colors.get(ch_num, "#999")
        title = module_titles.get(ch_num, f"Chapter {ch_num}")

        # Pick the top concepts for this chapter (roots within the chapter).
        # Prefer concepts not yet assigned as children of another concept.
        concept_nodes: list[dict] = []
        for c in ch_concepts[:_MAX_CONCEPTS_PER_CHAPTER]:
            if c.canonical_name in assigned:
                continue
            subtree = _build_concept_subtree(c.canonical_name)
            if subtree:
                concept_nodes.append(subtree)

        if not concept_nodes:
            continue

        roots.append({
            "name": title,
            "definition": "",
            "chapter": ch_num,
            "color": color,
            "importance": 999,
            "children": concept_nodes,
            "is_chapter": True,
        })

    if not roots:
        return {}

    return {"roots": roots}


# ── Internal: helpers ────────────────────────────────────────────────────────


def _render_fill_blanks(statement: str) -> str:
    """Replace _____ placeholders with HTML input elements."""
    counter = [0]

    def replacer(match: re.Match) -> str:
        idx = counter[0]
        counter[0] += 1
        return f'<input type="text" class="fitb-blank" data-blank-index="{idx}" placeholder="...">'

    return re.sub(r"_{3,}", replacer, statement)


# Currency pattern: $ immediately followed by a digit (e.g. $90, $1,000, $3.14).
# Only the $ is matched (lookahead); the digits stay in the text.
# The negative lookahead (?![a-zA-Z_\\$]) after the digits ensures that
# math expressions like $3v$, $3\begin{...}$, or $0$ are NOT treated as currency.
# NOTE: Use [$] instead of \$ — Python 3.13+ treats \$ as end-of-string anchor.
_CURRENCY_RE = re.compile(r"[$](?=\d[\d,]*\.?\d*(?![a-zA-Z_\\$]))")

# LaTeX block patterns (display then inline).
# NOTE: Use [$] instead of \$ — Python 3.13+ treats \$ as end-of-string anchor.
_LATEX_DISPLAY_RE = re.compile(r"[$][$].+?[$][$]", re.DOTALL)
_LATEX_INLINE_RE = re.compile(r"[$](?![$]).+?[$]")

# LaTeX \(...\) and \[...\] delimiter patterns.
# The LLM is told to use $...$ but sometimes produces these instead.
# Must be protected from markdown just like $ delimiters.
_LATEX_PAREN_RE = re.compile(r"\\\(.+?\\\)")
_LATEX_BRACKET_RE = re.compile(r"\\\[.+?\\\]", re.DOTALL)

# Pattern for double-escaped LaTeX commands inside math spans.
# LLMs sometimes produce \\frac instead of \frac in JSON output, resulting
# in literal double-backslashes in the parsed string.  We normalize these
# back to single backslashes so KaTeX can render them.
_DOUBLE_ESCAPED_CMD_RE = re.compile(r"\\\\([a-zA-Z]+)")


def _fix_double_escaped_latex(math_span: str) -> str:
    """Normalize \\\\cmd → \\cmd inside a math span.

    Preserves intentional ``\\\\`` (LaTeX line break) by only replacing
    ``\\\\`` when followed by a LaTeX command name (alphabetic chars).
    """
    return _DOUBLE_ESCAPED_CMD_RE.sub(r"\\\1", math_span)

# ── Doubled-math deduplication ────────────────────────────────────────────────
# LLMs sometimes emit each math expression twice: once rendered as plain text
# and once in LaTeX delimiters, producing e.g. "p=0.12$p=0.12$" or
# "$f_X(x)$f_X(x)" in the raw output.  These patterns detect and fix that.

# Pattern 1: plain-text duplicate immediately after a closing $ delimiter.
# Matches $<math>$<same-math-as-plain-text> and keeps only the delimited form.
# E.g. "$p=0.12$p=0.12" → "$p=0.12$"
_MATH_ECHO_AFTER_RE = re.compile(
    r"([$][$]?)(.+?)\1"   # group(1)=delim, group(2)=math content
    r"(?=\2)",             # lookahead: the same content repeated immediately
)

# Pattern 2: plain-text duplicate immediately before an opening $ delimiter.
# Matches <plain-text>$<same-text>$ and keeps only the delimited form.
# E.g. "p=0.12$p=0.12$" → "$p=0.12$"


def _deduplicate_math(text: str) -> str:
    """Remove LLM-generated doubled math expressions.

    LLMs sometimes write a math expression twice: once as plain text and
    once inside $ delimiters, back-to-back.  This produces rendered output
    like ``p=0.12p=0.12`` (the plain text + the KaTeX-rendered version).

    Strategy: find each $...$ or $$...$$ span and check whether the plain
    text immediately before or after it duplicates the math content (after
    stripping LaTeX commands).  If so, remove the plain-text duplicate.
    """
    if "$" not in text:
        return text

    # Collect all math spans with their positions
    spans: list[tuple[int, int, str]] = []  # (start, end, inner_content)
    for pattern in (_LATEX_DISPLAY_RE, _LATEX_INLINE_RE):
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end(), m.group(0)))
    if not spans:
        return text

    # Sort by position (display math first at same position due to greedier match)
    spans.sort(key=lambda s: s[0])

    # For each math span, derive a plain-text version by stripping LaTeX commands
    result = text
    for start, end, full_match in reversed(spans):  # reverse to preserve indices
        # Strip $ delimiters to get inner content
        if full_match.startswith("$$"):
            inner = full_match[2:-2]
        else:
            inner = full_match[1:-1]

        # Build a plain-text version: strip common LaTeX commands
        plain = _latex_to_plain(inner)
        if len(plain) < 2:
            continue

        # Check for duplicate immediately AFTER the math span
        after = result[end:]
        if after.startswith(plain):
            result = result[:end] + result[end + len(plain):]
        # Check for duplicate immediately BEFORE the math span
        elif start >= len(plain) and result[start - len(plain):start] == plain:
            result = result[:start - len(plain)] + result[start:]

    return result


def _latex_to_plain(latex: str) -> str:
    """Convert LaTeX math to approximate plain-text for dedup matching.

    Strips common LaTeX commands (\\frac, \\text, \\cdot, etc.) and
    formatting characters ({, }, ^, _) to produce a plain-text version
    that can be compared against the LLM's plain-text echo.
    """
    s = latex
    # Remove common LaTeX commands (keep their arguments)
    s = re.sub(r"\\(?:frac|text|mathrm|mathbf|mathit|operatorname|sqrt|hat|bar|vec|dot|tilde)\s*", "", s)
    # Remove remaining backslash commands (\cdot, \times, \leq, etc.)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    # Remove braces, caret, underscore
    s = re.sub(r"[{}^_]", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", "", s)
    return s

# Escape unescaped % in LaTeX math — % is a comment character in LaTeX/KaTeX.
_UNESCAPED_PERCENT_RE = re.compile(r"(?<!\\)%")


def _escape_latex_percent(math_str: str) -> str:
    """Escape unescaped ``%`` inside a LaTeX math span.

    In LaTeX, ``%`` starts a comment that eats the rest of the line.
    KaTeX inherits this behavior, so ``$\\text{Up by 2%}$`` breaks
    rendering.  Escaping to ``\\%`` fixes it.
    """
    return _UNESCAPED_PERCENT_RE.sub(r"\\%", math_str)


# Dangerous HTML patterns stripped after markdown conversion.
_SCRIPT_RE = re.compile(r"<script[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
_STYLE_RE = re.compile(r"<style[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE)
_EVENT_HANDLER_RE = re.compile(r"""\s+on\w+\s*=\s*(?:"[^"]*"|'[^']*')""", re.IGNORECASE)


def _protect_currency(text: str) -> tuple[str, list[str]]:
    """Replace currency $DIGITS patterns with sentinels before LaTeX detection."""
    currency_spans: list[str] = []

    def _save(match: re.Match) -> str:
        idx = len(currency_spans)
        currency_spans.append(match.group(0))
        return f"\x00CURR{idx}\x00"

    return _CURRENCY_RE.sub(_save, text), currency_spans


def _restore_currency(html: str, currency_spans: list[str]) -> str:
    """Restore currency sentinels after markdown conversion.

    Uses the HTML entity ``&#36;`` instead of a literal ``$`` so that
    KaTeX auto-render does not treat currency amounts (e.g. $100) as
    LaTeX math delimiters on the client side.
    """
    for idx, original in enumerate(currency_spans):
        safe = original.replace("$", "&#36;")
        html = html.replace(f"\x00CURR{idx}\x00", safe)
    return html


def _sanitize_html(html: str) -> str:
    """Strip dangerous HTML from markdown output (defense in depth).

    Removes <script>, <style> tags and on* event handler attributes.
    Content is LLM-generated (low risk), but since we render with |safe
    in templates we add this safety layer.
    """
    html = _SCRIPT_RE.sub("", html)
    html = _STYLE_RE.sub("", html)
    html = _EVENT_HANDLER_RE.sub("", html)
    return html


def _markdown_to_html(text: str) -> str:
    """Convert markdown to HTML, preserving LaTeX math delimiters for KaTeX.

    Uses the Python-Markdown library for correct parsing of bold, italic,
    inline code, fenced code blocks, lists, and paragraphs. LaTeX math
    ($...$, $$...$$) is protected from markdown interpretation and restored
    after conversion so KaTeX can render it client-side.

    Currency dollar signs ($90, $1,000) are protected from being treated
    as LaTeX delimiters.  Dangerous HTML (script tags, event handlers) is
    stripped from the output.
    """
    # Fix LLM doubled-math before any other processing.
    text = _deduplicate_math(text)

    # Protect currency $ from being treated as LaTeX delimiters.
    text, currency_spans = _protect_currency(text)

    # Protect LaTeX math blocks from markdown interpretation.
    # Display math ($$...$$) must be matched before inline ($...$).
    # Also protect \(...\) and \[...\] delimiters (LLM sometimes uses these
    # despite being told to use $ delimiters).
    math_spans: list[str] = []

    def _save_math(match: re.Match) -> str:
        idx = len(math_spans)
        span = _fix_double_escaped_latex(match.group(0))
        math_spans.append(_escape_latex_percent(span))
        return f"\x00MATH{idx}\x00"

    protected = _LATEX_DISPLAY_RE.sub(_save_math, text)
    protected = _LATEX_BRACKET_RE.sub(_save_math, protected)
    protected = _LATEX_INLINE_RE.sub(_save_math, protected)
    protected = _LATEX_PAREN_RE.sub(_save_math, protected)

    html = md.markdown(protected, extensions=["fenced_code"])

    for idx, original in enumerate(math_spans):
        html = html.replace(f"\x00MATH{idx}\x00", original)

    html = _restore_currency(html, currency_spans)
    return _sanitize_html(html)


def _markdown_to_html_inline(text: str) -> str:
    """Convert markdown to HTML without the outer ``<p>`` wrapper.

    Use for inline contexts where content appears inside existing block
    elements (quiz questions, option labels, prompts inside ``<p>``).
    Single-paragraph content loses the wrapper; multi-paragraph keeps it.
    """
    html = _markdown_to_html(text)
    stripped = html.strip()
    if stripped.startswith("<p>") and stripped.endswith("</p>"):
        inner = stripped[3:-4]
        if "<p>" not in inner:
            return inner
    return html


def _render_fitb_statement(statement: str, answers: list[str] | None = None) -> str:
    """Process a FITB statement with LaTeX-aware blank handling and markdown.

    Blanks (``_{3,}``) inside LaTeX blocks are replaced with KaTeX-renderable
    ``\\underline{\\hspace{3em}}`` markers since ``<input>`` HTML cannot
    appear inside KaTeX math.  Blanks outside LaTeX become interactive
    ``<input>`` elements with per-blank hint buttons.  The whole statement
    then receives markdown conversion with LaTeX and currency protection.

    Args:
        statement: The FITB statement with ``_____`` marking each blank.
        answers: Optional list of correct answers (one per blank, in order).
            When provided, each ``<input>`` gets a ``data-answer`` attribute
            and a hint button for progressive letter reveal.
    """
    # Step 1: protect currency
    text, currency_spans = _protect_currency(statement)

    # Step 2: replace blanks inside LaTeX with KaTeX markers
    def _replace_blanks_in_latex(match: re.Match) -> str:
        return re.sub(r"_{3,}", r"\\underline{\\hspace{3em}}", match.group(0))

    text = _LATEX_DISPLAY_RE.sub(_replace_blanks_in_latex, text)
    text = _LATEX_INLINE_RE.sub(_replace_blanks_in_latex, text)

    # Step 3: replace remaining blanks (outside LaTeX) with <input> elements
    # Also compute which original blank indices map to interactive inputs
    interactive_indices = _fitb_interactive_answer_indices(statement)
    blank_counter = [0]

    def _input_replacer(match: re.Match) -> str:
        idx = blank_counter[0]
        blank_counter[0] += 1
        # Map interactive blank index to the original answer index
        answer_attr = ""
        if answers and idx < len(interactive_indices):
            orig_idx = interactive_indices[idx]
            if orig_idx < len(answers):
                escaped = answers[orig_idx].replace('"', "&quot;")
                answer_attr = f' data-answer="{escaped}"'
        hint_btn = (
            '<button class="hint-letter-btn fitb-hint-letter-btn" '
            'onclick="revealNextLetters(this.previousElementSibling)" '
            'title="Reveal a letter">?</button>'
        )
        return (
            f'<input type="text" class="fitb-blank" '
            f'data-blank-index="{idx}" placeholder="..."{answer_attr}>'
            f'{hint_btn}'
        )

    text = re.sub(r"_{3,}", _input_replacer, text)

    # Step 4: protect LaTeX from markdown
    math_spans: list[str] = []

    def _save_math(match: re.Match) -> str:
        idx = len(math_spans)
        span = _fix_double_escaped_latex(match.group(0))
        math_spans.append(_escape_latex_percent(span))
        return f"\x00MATH{idx}\x00"

    protected = _LATEX_DISPLAY_RE.sub(_save_math, text)
    protected = _LATEX_INLINE_RE.sub(_save_math, protected)

    # Also protect <input> tags from markdown
    input_spans: list[str] = []

    def _save_input(match: re.Match) -> str:
        idx = len(input_spans)
        input_spans.append(match.group(0))
        return f"\x00INPUT{idx}\x00"

    protected = re.sub(r"<input[^>]+>(?:<button[^>]+>[^<]*</button>)?", _save_input, protected)

    # Step 5: markdown conversion
    html = md.markdown(protected, extensions=["fenced_code"])

    # Step 6: restore everything
    for idx, original in enumerate(input_spans):
        html = html.replace(f"\x00INPUT{idx}\x00", original)
    for idx, original in enumerate(math_spans):
        html = html.replace(f"\x00MATH{idx}\x00", original)
    html = _restore_currency(html, currency_spans)
    return _sanitize_html(html)


def _fitb_interactive_answer_indices(statement: str) -> list[int]:
    """Return original blank indices that become interactive ``<input>`` fields.

    Blanks inside LaTeX become visual-only markers; blanks outside become
    interactive inputs.  Returns 0-based indices into the original answer
    list for the blanks that will be interactive.
    """
    text, _ = _protect_currency(statement)

    # Build LaTeX region spans
    latex_ranges: list[tuple[int, int]] = []
    for m in _LATEX_DISPLAY_RE.finditer(text):
        latex_ranges.append((m.start(), m.end()))
    for m in _LATEX_INLINE_RE.finditer(text):
        if not any(s <= m.start() < e for s, e in latex_ranges):
            latex_ranges.append((m.start(), m.end()))

    def _is_in_latex(pos: int) -> bool:
        return any(s <= pos < e for s, e in latex_ranges)

    return [
        i
        for i, m in enumerate(re.finditer(r"_{3,}", text))
        if not _is_in_latex(m.start())
    ]


def _encode_image_base64(image_path: Path) -> str | None:
    """Read an image file and return a base64 data URI string."""
    if not image_path.exists():
        logger.warning("Image not found: %s", image_path)
        return None

    try:
        data = image_path.read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        suffix = image_path.suffix.lower()
        mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}
        mime = mime_types.get(suffix, "image/png")
        return f"data:{mime};base64,{encoded}"
    except OSError:
        logger.warning("Failed to read image: %s", image_path, exc_info=True)
        return None


def _derive_course_title(modules: Sequence[TrainingModule]) -> str:
    """Derive a course title from the module titles.

    Uses the common prefix of chapter titles if one exists (e.g. if all
    chapters are "Quant Finance: ..."), otherwise falls back to a generic title.
    """
    if not modules:
        return "Interactive Training Course"

    titles = [m.title for m in modules]
    if len(titles) == 1:
        return titles[0]

    # Find common prefix across all chapter titles (split on word boundaries)
    prefix = titles[0]
    for title in titles[1:]:
        while prefix and not title.startswith(prefix):
            prefix = prefix.rsplit(" ", 1)[0] if " " in prefix else ""

    # Use the common prefix if it's meaningful (>5 chars), else generic
    prefix = prefix.rstrip(" :-–—")
    if len(prefix) > 5:
        return prefix

    return "Interactive Training Course"
