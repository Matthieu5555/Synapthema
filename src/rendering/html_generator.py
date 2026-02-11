"""HTML rendering — the deep module for Stage 3.

Single public entry point: render_course(). Takes a sequence of TrainingModules
from Stage 2 and produces self-contained HTML files — one per chapter plus
an index page. All CSS and JS are inlined for zero-dependency output.

Supports all 5 element types: slides, quizzes, flashcards, fill-in-the-blank,
and matching exercises. Includes KaTeX for math rendering and Bloom's Taxonomy
badges.
"""

from __future__ import annotations

import base64
import json
import logging
import random
import re
from collections.abc import Sequence
from pathlib import Path

import markdown as md
from jinja2 import Environment, FileSystemLoader

from functools import reduce

from src.transformation.analysis_types import ChapterAnalysis, ConceptGraph
from src.transformation.types import (
    ConceptMapElement,
    FillInBlankElement,
    FlashcardElement,
    InteractiveEssayElement,
    MatchingElement,
    MermaidElement,
    QuizElement,
    SelfExplainElement,
    SlideElement,
    TrainingElement,
    TrainingModule,
    TrainingSection,
)

logger = logging.getLogger(__name__)

# Directory containing the Jinja2 HTML/CSS templates.
_TEMPLATES_DIR = Path(__file__).parent / "templates"

# Bloom's Taxonomy sort order for difficulty progression within sections.
_BLOOM_SORT_ORDER: dict[str, int] = {
    "remember": 1, "understand": 2, "apply": 3,
    "analyze": 4, "evaluate": 5, "create": 6,
}


def render_course(
    modules: Sequence[TrainingModule],
    output_dir: Path,
    extracted_dir: Path | None = None,
    embed_images: bool = True,
    concept_graph: ConceptGraph | None = None,
    chapter_analyses: list[ChapterAnalysis] | None = None,
    course_title: str | None = None,
    course_summary: str | None = None,
    learner_journey: str | None = None,
    source_book_titles: dict[int, str] | None = None,
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

    course_title = course_title or _derive_course_title(modules)
    course_slug = output_dir.name  # Already a slug from config.py

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
    is_multi_doc = bool(source_book_titles and len(set(source_book_titles.values())) > 1)

    # First pass: build chapter metadata for cross-navigation
    chapter_info: list[dict] = [
        {
            "number": m.chapter_number,
            "title": m.title,
            "filename": f"chapter_{m.chapter_number:02d}.html",
            "element_count": len(m.all_elements),
            "section_count": len(m.sections),
            "source_book_title": source_book_titles.get(m.chapter_number, "") if source_book_titles else "",
        }
        for m in modules
    ]

    # Second pass: render chapters with full navigation context
    for i, module in enumerate(modules):
        html_path = output_dir / chapter_info[i]["filename"]
        prev_chapter = chapter_info[i - 1] if i > 0 else None
        next_chapter = chapter_info[i + 1] if i < len(modules) - 1 else None

        _render_chapter(
            module=module,
            env=env,
            css=css,
            output_path=html_path,
            extracted_dir=extracted_dir,
            embed_images=embed_images,
            course_title=course_title,
            course_slug=course_slug,
            chapter_count=len(modules),
            prev_chapter=prev_chapter,
            next_chapter=next_chapter,
            chapter_analysis=analyses_by_chapter.get(module.chapter_number),
            concept_graph=concept_graph,
            source_book_title=chapter_info[i]["source_book_title"] if is_multi_doc else "",
        )

        logger.info(
            "Rendered chapter %d: %s (%d elements, %d sections)",
            module.chapter_number,
            chapter_info[i]["filename"],
            len(module.all_elements),
            len(module.sections),
        )

    # Prepare concept graph data for the index page
    graph_data = None
    mindmap_data = None
    if concept_graph and concept_graph.concepts:
        graph_data = _prepare_graph_data(concept_graph, modules)
        mindmap_data = _prepare_mindmap_data(concept_graph, modules)

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
) -> None:
    """Render a single chapter's training module to an HTML file."""
    template = env.get_template("base.html")

    # Build section-grouped data for the sidebar and content via fold
    sections_data, flat_elements = _build_sections_data(
        module, extracted_dir, embed_images,
        chapter_analysis=chapter_analysis,
        concept_graph=concept_graph,
    )

    html = template.render(
        module_title=module.title,
        css=css,
        elements=flat_elements,
        sections=sections_data,
        course_title=course_title,
        course_slug=course_slug,
        chapter_number=module.chapter_number,
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
) -> tuple[list[dict], list[dict]]:
    """Build section metadata and flat element list via fold.

    Returns (sections_data, flat_elements) where each section tracks its
    start_index into the flat list. Each element is tagged with a
    deterministic element_id (for FSRS tracking) and concepts_tested
    (for the learner model).
    """
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
        concepts = section_concepts.get(section.title, [])
        reinf_targets = section.reinforcement_targets if hasattr(section, "reinforcement_targets") else None
        prepared_elements = []
        for i, e in enumerate(section.elements):
            prepared = _prepare_element(e, extracted_dir, embed_images)
            element_dict = {
                **prepared,
                "element_id": _element_id(
                    module.chapter_number, section_index, i,
                ),
                "concepts_tested": _tag_element_concepts(prepared, concepts, reinf_targets),
            }
            prepared_elements.append(element_dict)
        section_elements = sorted(
            prepared_elements,
            key=lambda e: _BLOOM_SORT_ORDER.get(e.get("bloom_level", "apply"), 3),
        )
        section_data = {
            "title": section.title,
            "source_pages": section.source_pages,
            "element_count": len(section_elements),
            "start_index": offset,
            "verification_notes": section.verification_notes,
            "learning_objectives": section.learning_objectives if hasattr(section, "learning_objectives") else [],
        }
        return (
            sections_so_far + [section_data],
            flat_so_far + section_elements,
            offset + len(section_elements),
            section_index + 1,
        )

    sections_data, flat_elements, _, _ = reduce(
        fold, module.sections, ([], [], 0, 0),
    )
    return sections_data, flat_elements


def _prepare_element(
    elem: SlideElement | QuizElement | FlashcardElement | FillInBlankElement
    | MatchingElement | MermaidElement | ConceptMapElement
    | SelfExplainElement | InteractiveEssayElement,
    extracted_dir: Path | None,
    embed_images: bool,
) -> dict:
    """Convert a TrainingElement variant into a template-friendly dict.

    Uses structural pattern matching to destructure each element type.
    """
    result: dict = {
        "element_type": elem.element_type,
        "bloom_level": elem.bloom_level,
    }

    match elem:
        case SlideElement(slide=s):
            image_data = None
            if s.image_path and extracted_dir and embed_images:
                image_data = _encode_image_base64(extracted_dir / Path(s.image_path))
            result["slide"] = {
                "title": s.title,
                "content_html": _markdown_to_html(s.content),
                "speaker_notes": s.speaker_notes,
                "image_data": image_data,
                "source_pages": s.source_pages,
            }

        case QuizElement(quiz=q):
            result["quiz"] = {
                "title": q.title,
                "questions": [
                    {
                        "question": qq.question,
                        "options": list(qq.options),
                        "correct_index": qq.correct_index,
                        "explanation": qq.explanation,
                        "hint_metacognitive": qq.hint_metacognitive,
                        "hint_strategic": qq.hint_strategic,
                        "hint_eliminate_index": qq.hint_eliminate_index,
                    }
                    for qq in q.questions
                ],
            }

        case FlashcardElement(flashcard=f):
            result["flashcard"] = {"front": f.front, "back": f.back}

        case FillInBlankElement(fill_in_the_blank=fitb):
            result["fill_in_the_blank"] = {
                "statement_html": _render_fill_blanks(fitb.statement),
                "answers_json": json.dumps(list(fitb.answers)),
                "hint": fitb.hint,
                "hint_first_letter": fitb.hint_first_letter,
            }

        case MatchingElement(matching=m):
            left = list(m.left_items)
            right = list(m.right_items)
            rng = random.Random(m.title)
            shuffled_indices = list(range(len(right)))
            rng.shuffle(shuffled_indices)
            shuffled_right = [right[i] for i in shuffled_indices]
            correct_map = {
                orig: shuffled_indices.index(orig)
                for orig in range(len(left))
            }
            result["matching"] = {
                "title": m.title,
                "left_items": left,
                "shuffled_right": shuffled_right,
                "correct_json": json.dumps(correct_map),
                "pair_explanations_json": json.dumps(list(m.pair_explanations)),
            }

        case MermaidElement(mermaid=d):
            result["mermaid"] = {
                "title": d.title,
                "diagram_code": d.diagram_code,
                "caption": d.caption,
                "diagram_type": d.diagram_type,
            }

        case ConceptMapElement(concept_map=cmap):
            result["concept_map"] = {
                "title": cmap.title,
                "nodes_json": json.dumps([n.model_dump() for n in cmap.nodes]),
                "edges_json": json.dumps([e.model_dump() for e in cmap.edges]),
                "blank_indices_json": json.dumps(cmap.blank_edge_indices),
            }

        case SelfExplainElement(self_explain=se):
            result["self_explain"] = {
                "prompt": se.prompt,
                "key_points_json": json.dumps(se.key_points),
                "example_response": _markdown_to_html(se.example_response),
                "minimum_words": se.minimum_words,
                "source_pages": se.source_pages,
            }

        case InteractiveEssayElement(interactive_essay=ie):
            result["interactive_essay"] = {
                "title": ie.title,
                "concepts_tested": ie.concepts_tested,
                "prompts": [
                    {
                        "prompt": p.prompt,
                        "key_points_json": json.dumps(p.key_points),
                        "example_response": _markdown_to_html(p.example_response),
                        "minimum_words": p.minimum_words,
                    }
                    for p in ie.prompts
                ],
                "passing_threshold": ie.passing_threshold,
                "tutor_system_prompt_json": json.dumps(ie.tutor_system_prompt),
            }

    return result


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
    for key in ("slide", "quiz", "flashcard", "fill_in_the_blank", "matching",
                "self_explain", "concept_map", "interactive_essay", "mermaid"):
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


def _load_course_meta(output_dir: Path) -> dict | None:
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
) -> dict:
    """Convert ConceptGraph to vis-network compatible JSON.

    Selects the most cross-referenced concepts (up to _MAX_GRAPH_NODES),
    truncates long labels, and uses a curated color palette.
    """
    chapter_colors = _generate_chapter_colors(len(modules))

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
            "group": c.first_introduced_chapter,
            "color": chapter_colors.get(c.first_introduced_chapter, "#999"),
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


# Maximum nodes in the hierarchical mind map.
_MAX_MINDMAP_NODES = 60

# Edge types that imply a hierarchical (parent → child) relationship.
_HIERARCHICAL_EDGE_TYPES = {"builds_on", "requires"}


def _prepare_mindmap_data(
    concept_graph: ConceptGraph,
    modules: Sequence[TrainingModule],
) -> dict:
    """Build a top-down hierarchical mind map from the concept graph.

    Foundation concepts sit at the top; concepts that build on them flow
    downward.  Only ``builds_on`` and ``requires`` edges are used (they
    imply a prerequisite hierarchy).  The vis-network hierarchical layout
    renders the result as a tree.

    Edge direction for display: prerequisite (target) → dependent (source),
    so foundations appear at the root and advanced concepts at the leaves.
    """
    chapter_colors = _generate_chapter_colors(len(modules))

    # Only keep hierarchical edges
    hier_edges = [
        e for e in concept_graph.edges
        if e.relationship in _HIERARCHICAL_EDGE_TYPES
    ]
    if not hier_edges:
        return {}

    # Concepts that participate in at least one hierarchical edge
    connected: set[str] = set()
    for e in hier_edges:
        connected.add(e.source)
        connected.add(e.target)

    # Rank connected concepts by cross-chapter importance
    concept_lookup = {c.canonical_name: c for c in concept_graph.concepts}
    connected_concepts = [
        concept_lookup[name]
        for name in connected
        if name in concept_lookup
    ]
    connected_concepts.sort(
        key=lambda c: len(c.mentioned_in_chapters), reverse=True
    )
    top_concepts = connected_concepts[:_MAX_MINDMAP_NODES]
    top_ids = {c.canonical_name for c in top_concepts}

    # Build nodes
    nodes = [
        {
            "id": c.canonical_name,
            "label": _truncate_label(c.canonical_name, max_len=24),
            "title": (
                f"<b>{c.canonical_name}</b><br>{c.definition}"
                if c.definition
                else c.canonical_name
            ),
            "group": c.first_introduced_chapter,
            "color": chapter_colors.get(c.first_introduced_chapter, "#999"),
            "size": 6 + 3 * len(c.mentioned_in_chapters),
        }
        for c in top_concepts
    ]

    # Edges: reverse direction so prerequisites point DOWN to dependents.
    # In the data, source depends on target (target is prerequisite).
    # For UD layout, from=prerequisite (top) → to=dependent (bottom).
    edges = [
        {
            "from": e.target,
            "to": e.source,
            "arrows": "to",
        }
        for e in hier_edges
        if e.source in top_ids and e.target in top_ids
    ]

    if not edges:
        return {}

    return {"nodes": nodes, "edges": edges}


# ── Internal: helpers ────────────────────────────────────────────────────────


def _render_fill_blanks(statement: str) -> str:
    """Replace _____ placeholders with HTML input elements."""
    counter = [0]

    def replacer(match: re.Match) -> str:
        idx = counter[0]
        counter[0] += 1
        return f'<input type="text" class="fitb-blank" data-blank-index="{idx}" placeholder="...">'

    return re.sub(r"_{3,}", replacer, statement)


def _markdown_to_html(text: str) -> str:
    """Convert markdown to HTML, preserving LaTeX math delimiters for KaTeX.

    Uses the Python-Markdown library for correct parsing of bold, italic,
    inline code, fenced code blocks, lists, and paragraphs. LaTeX math
    ($...$, $$...$$) is protected from markdown interpretation and restored
    after conversion so KaTeX can render it client-side.
    """
    # Protect LaTeX math blocks from markdown interpretation.
    # Display math ($$...$$) must be matched before inline ($...$).
    math_spans: list[str] = []

    def _save_math(match: re.Match) -> str:
        idx = len(math_spans)
        math_spans.append(match.group(0))
        return f"\x00MATH{idx}\x00"

    protected = re.sub(r"\$\$.+?\$\$", _save_math, text, flags=re.DOTALL)
    protected = re.sub(r"\$(?!\$).+?\$", _save_math, protected)

    html = md.markdown(protected, extensions=["fenced_code"])

    for idx, original in enumerate(math_spans):
        html = html.replace(f"\x00MATH{idx}\x00", original)

    return html


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
