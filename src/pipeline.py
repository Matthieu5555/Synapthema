"""Main pipeline orchestrator — wires Stages 1, 2, and 3 together.

Single public entry point: run_pipeline(). Extracts content from a PDF,
transforms it into interactive training elements via an LLM, and renders
the result as a self-contained HTML course.

Also provides main() as a CLI entry point.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel, TypeAdapter, ValidationError

from src.checkpoint import load_checkpoint, save_checkpoint, save_checkpoint_raw
from src.config import Config, InputSource
from src.extraction.multi_doc import extract_corpus, source_slug
from src.extraction.pdf_parser import extract_book
from src.extraction.types import Book, Chapter
from src.rendering.html_generator import RenderContext, render_course
from src.transformation.analysis_types import ChapterAnalysis, ConceptGraph
from src.transformation.concept_consolidator import consolidate_concepts
from src.profiles import ContentProfile, get_profile
from src.transformation.content_designer import TransformContext, transform_chapter
from src.transformation.content_pre_analyzer import detect_document_type
from src.transformation.curriculum_planner import (
    find_matching_chapter,
    plan_curriculum,
    plan_multi_document_curriculum,
)
from src.transformation.deep_reader import analyze_book
from src.rendering.mermaid_validator import validate_and_fix_mermaid_diagrams
from src.transformation.llm_client import LLMClient, create_llm_client
from src.transformation.types import (
    CourseCapabilities,
    CurriculumBlueprint,
    ModuleBlueprint,
    TrainingModule,
)

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _timed_stage(stage_name: str, **extra: object):
    """Log the duration of a pipeline stage."""
    logger.info("▶ %s started", stage_name)
    t0 = time.monotonic()
    yield
    elapsed = time.monotonic() - t0
    parts = [f"◀ {stage_name} completed in {elapsed:.1f}s"]
    for k, v in extra.items():
        parts.append(f"{k}={v}")
    logger.info(", ".join(parts))


class _AnalysesCheckpoint(BaseModel):
    """Composite checkpoint for deep reading analyses + concept graph."""

    chapter_analyses: list[ChapterAnalysis]
    concept_graph: ConceptGraph


def _book_extracted_dirs(
    extracted_dir: Path,
    input_sources: list[InputSource],
) -> list[Path]:
    """Return per-book extraction directories.

    Single-doc: returns [extracted_dir].
    Multi-doc: returns [extracted_dir/00_slug, extracted_dir/01_slug, ...].
    """
    if len(input_sources) <= 1:
        return [extracted_dir]
    return [
        extracted_dir / source_slug(src, idx)
        for idx, src in enumerate(input_sources)
    ]


def _maybe_create_llm_client(config: Config) -> LLMClient | None:
    """Create an LLM client if credentials are available, else return None."""
    if not config.llm_api_key:
        return None
    try:
        return create_llm_client(
            api_key=config.llm_api_key,
            model=config.llm_model,
            max_tokens=config.llm_max_tokens,
            temperature=config.llm_temperature,
            base_url=config.llm_base_url,
            model_light=config.llm_model_light,
            model_creative=config.llm_model_creative,
        )
    except Exception as exc:
        logger.warning("Failed to create LLM client for mermaid fixing: %s", exc)
        return None


def rerender_from_json(
    config: Config,
    exclude_element_types: set[str] | None = None,
) -> Path:
    """Re-render HTML from an existing training_modules.json.

    Skips extraction and LLM transformation — loads the intermediate JSON
    saved by a previous pipeline run and goes straight to Stage 3 rendering.
    If chapter_analyses.json exists, loads the concept graph and analyses
    for concept tagging and graph visualization.

    Args:
        config: Pipeline config (only extracted_dir, output_dir, and
            embed_images are used; LLM fields are ignored).
        exclude_element_types: If provided, remove elements of these types
            before rendering (e.g., {"mermaid", "concept_map"}).

    Returns:
        Path to the generated course index.html.
    """
    training_json = config.extracted_dir / "training_modules.json"
    logger.info("Render-only: loading %s", training_json)

    modules = _load_training_modules_checkpoint(training_json)
    if modules is None:
        msg = "Cannot load training modules from %s"
        logger.error(msg, training_json)
        raise FileNotFoundError(msg % training_json)
    logger.info("Loaded %d modules from JSON", len(modules))

    # Post-filter: remove excluded element types
    if exclude_element_types:
        for module in modules:
            for section in module.sections:
                section.elements = [
                    e for e in section.elements
                    if e.element_type not in exclude_element_types
                ]
        logger.info("Excluded element types: %s", ", ".join(sorted(exclude_element_types)))

    # Validate and fix mermaid diagrams (optional — requires LLM + Node.js)
    llm_client = _maybe_create_llm_client(config)
    total, fixed, _ = validate_and_fix_mermaid_diagrams(modules, llm_client)
    if fixed > 0:
        _save_training_json(modules, training_json)

    # Load concept graph and analyses if available (from a prior pipeline run)
    concept_graph = None
    chapter_analyses = None
    analyses_json = config.extracted_dir / "chapter_analyses.json"
    loaded_analyses = load_checkpoint(analyses_json, _AnalysesCheckpoint)
    if loaded_analyses is not None:
        chapter_analyses = loaded_analyses.chapter_analyses
        concept_graph = loaded_analyses.concept_graph
        logger.info(
            "Loaded concept graph (%d concepts) and %d chapter analyses",
            len(concept_graph.concepts),
            len(chapter_analyses),
        )

    # Load course metadata and blueprint for chapter mapping
    course_title = course_summary = learner_journey = None
    blueprint: CurriculumBlueprint | None = None

    meta_path = config.html_dir / "course_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            course_title = meta.get("course_title")
            course_summary = meta.get("course_summary")
            learner_journey = meta.get("learner_journey")
            logger.info("Loaded course metadata from %s", meta_path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load course_meta.json: %s", exc)

    blueprint_path = config.extracted_dir / "curriculum_blueprint.json"
    if blueprint_path.exists():
        try:
            blueprint = CurriculumBlueprint.model_validate(
                json.loads(blueprint_path.read_text(encoding="utf-8"))
            )
            if not course_title:
                course_title = blueprint.course_title
                course_summary = blueprint.course_summary
                learner_journey = blueprint.learner_journey
            logger.info("Loaded curriculum blueprint")
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Failed to load curriculum_blueprint.json: %s", exc)

    # Build chapter_to_module mapping for multi-doc courses
    chapter_to_module: dict[int, int] | None = None
    if blueprint and any(
        (bp.source_book_index or 0) > 0 for bp in blueprint.modules
    ):
        book_structure_path = config.extracted_dir / "book_structure.json"
        if book_structure_path.exists():
            try:
                books_data = json.loads(
                    book_structure_path.read_text(encoding="utf-8")
                )
                book_chapter_counts = [
                    len(b.get("chapters", [])) for b in books_data
                ]
                chapter_to_module = {}
                for mod_idx, bp in enumerate(blueprint.modules):
                    book_idx = bp.source_book_index or 0
                    ch_num = bp.source_chapter_number
                    if ch_num is not None and book_idx < len(book_chapter_counts):
                        book_offset = sum(book_chapter_counts[:book_idx])
                        global_ch = book_offset + ch_num
                        if global_ch not in chapter_to_module:
                            chapter_to_module[global_ch] = mod_idx + 1
                logger.info(
                    "Built chapter_to_module mapping: %s", chapter_to_module,
                )
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "Failed to load book_structure.json for chapter mapping: %s", exc,
                )

    capabilities = compute_capabilities(
        modules, concept_graph, chapter_analyses, course_title,
    )

    render_ctx = RenderContext(
        extracted_dir=config.extracted_dir,
        embed_images=config.embed_images,
        concept_graph=concept_graph,
        chapter_analyses=chapter_analyses,
        course_title=course_title,
        course_summary=course_summary,
        learner_journey=learner_journey,
        chapter_to_module=chapter_to_module,
        capabilities=capabilities,
    )
    index_path = render_course(modules, config.html_dir, render_ctx)

    logger.info("Render-only complete. Open %s in a browser.", index_path)
    return index_path


def run_pipeline(
    config: Config,
    chapter_number: int | None = None,
    resume: bool = False,
) -> Path:
    """Execute the full extraction → transformation → rendering pipeline.

    Supports both single-document and multi-document modes. When multiple
    input sources are provided, an LLM-powered planner analyzes all documents
    and constructs a unified linear course.

    When resume=True, each stage checks for existing checkpoint files and
    skips if the output is valid. This avoids re-running expensive LLM calls
    after a mid-pipeline failure.

    Args:
        config: Validated pipeline configuration.
        chapter_number: If set, only transform and render this chapter.
            Useful for testing LLM prompts without processing the entire book.
        resume: If True, load existing checkpoints instead of re-running
            completed stages.

    Returns:
        Path to the generated course index.html.
    """
    pipeline_t0 = time.monotonic()
    source_names = ", ".join(s.path.name for s in config.input_sources)
    mode = f"chapter {chapter_number}" if chapter_number else "full"
    resume_mode = ", resume" if resume else ""
    logger.info("Starting pipeline (%s%s): %s", mode, resume_mode, source_names)

    # Create LLM client first — needed for both TOC detection and content transformation
    client = create_llm_client(
        api_key=config.llm_api_key,
        model=config.llm_model,
        max_tokens=config.llm_max_tokens,
        temperature=config.llm_temperature,
        base_url=config.llm_base_url,
        model_light=config.llm_model_light,
        model_creative=config.llm_model_creative,
    )

    # Stage 1: Extraction
    with _timed_stage("Stage 1: Extraction"):
        books = _load_or_extract(config.extracted_dir, config.input_sources, client, resume)

    total_chapters = sum(len(b.chapters) for b in books)
    logger.info("Extracted %d book(s), %d chapters total", len(books), total_chapters)

    # Stage 1.25: Deep reading + Stage 1.3: Consolidation
    with _timed_stage("Stage 1.5: Deep reading & consolidation", chapters=total_chapters):
        all_chapter_analyses, concept_graph, global_flat = _load_or_analyze(
            config.extracted_dir, books, client, total_chapters, resume,
            max_workers=config.max_concurrent_llm,
        )

    # Document type detection (zero-cost heuristic, or manual override)
    if config.document_type == "auto":
        if len(books) > 1:
            # Per-book document type detection
            per_book_types = {i: detect_document_type(book) for i, book in enumerate(books)}
            # Use the most common type as the overall type
            from collections import Counter
            document_type = Counter(per_book_types.values()).most_common(1)[0][0]
            logger.info(
                "Per-book document types: %s",
                {i: t for i, t in per_book_types.items()},
            )
        else:
            document_type = detect_document_type(books[0])
    else:
        document_type = config.document_type  # type: ignore[assignment]
    logger.info("Document type: %s", document_type)

    # Stage 1.5: Curriculum planning
    with _timed_stage("Stage 1.75: Curriculum planning"):
        blueprint = _load_or_plan(
            config.extracted_dir, books, client, all_chapter_analyses, concept_graph, resume,
            document_type=document_type,
        )

    logger.info(
        "Curriculum planned: %d modules, journey: %s",
        len(blueprint.modules),
        blueprint.learner_journey[:100] if blueprint.learner_journey else "N/A",
    )

    # Stage 2: Transformation (with partial resume support)
    # Use per-book chapter numbers for transformation lookups
    analyses_by_book_chapter: dict[tuple[int, int], ChapterAnalysis] = {
        (book_idx, a.chapter_number): a
        for book_idx, book_analyses in enumerate(all_chapter_analyses)
        for a in book_analyses
    }

    # Compute per-book extraction directories for correct image path resolution
    book_dirs = _book_extracted_dirs(config.extracted_dir, config.input_sources)

    profile = get_profile(config.variant)

    with _timed_stage("Stage 2: Transformation", modules=len(blueprint.modules)):
        modules = _load_or_transform(
            config.extracted_dir, config.vision_enabled,
            blueprint, books, client, analyses_by_book_chapter,
            chapter_number, resume, concept_graph=concept_graph,
            document_type=document_type,
            max_workers=config.max_concurrent_llm,
            book_extracted_dirs=book_dirs,
            profile=profile,
            viz_enabled=config.viz_enabled,
        )

    # Build source book title list aligned with modules (index-based).
    # Uses blueprint module order (parallel to the modules list) so the
    # renderer can look up by sequential index rather than chapter_number,
    # which avoids mismatches when find_matching_chapter() resolves a
    # blueprint's source_chapter_number to a different extraction chapter.
    source_book_titles: list[str] | None = None
    if len(books) > 1:
        source_book_titles = [
            books[bp.source_book_index or 0].title
            if (bp.source_book_index or 0) < len(books)
            else ""
            for bp in blueprint.modules
        ]

    # Validate and fix mermaid diagrams before rendering
    total, fixed, _ = validate_and_fix_mermaid_diagrams(modules, client)
    if fixed > 0:
        _save_training_json(modules, config.extracted_dir / "training_modules.json")

    # Build mapping from global chapter number → sequential module index (1-based).
    # The concept graph uses global chapter numbers; the renderer needs to map
    # them to module indices for correct mindmap grouping and graph colors.
    chapter_to_module: dict[int, int] | None = None
    if len(books) > 1:
        chapter_to_module = {}
        for mod_idx, bp in enumerate(blueprint.modules):
            book_idx = bp.source_book_index or 0
            ch_num = bp.source_chapter_number
            if ch_num is not None and book_idx < len(books):
                book_offset = sum(
                    len(books[b].chapters) for b in range(book_idx)
                )
                global_ch = book_offset + ch_num
                # First module wins — concepts grouped under the first module
                # that covers this source chapter.
                if global_ch not in chapter_to_module:
                    chapter_to_module[global_ch] = mod_idx + 1
            # Also map additional source chapters to this module
            for asc in bp.additional_source_chapters:
                asc_bk = asc.get("book_index", 0)
                asc_ch = asc.get("chapter_number")
                if asc_ch is not None and asc_bk < len(books):
                    asc_offset = sum(len(books[b].chapters) for b in range(asc_bk))
                    asc_global = asc_offset + asc_ch
                    if asc_global not in chapter_to_module:
                        chapter_to_module[asc_global] = mod_idx + 1

    # Build per-module extracted dirs for correct image resolution in rendering
    per_module_dirs: list[Path] | None = None
    if len(books) > 1:
        per_module_dirs = [
            book_dirs[bp.source_book_index or 0]
            if (bp.source_book_index or 0) < len(book_dirs)
            else config.extracted_dir
            for bp in blueprint.modules
        ]

    # Compute course capabilities
    capabilities = compute_capabilities(
        modules, concept_graph, global_flat, blueprint.course_title,
    )
    logger.info("Course capabilities: %s", capabilities)

    # Stage 3: Render (always re-run — it's instant)
    with _timed_stage("Stage 3: Rendering", modules=len(modules)):
        render_ctx = RenderContext(
            extracted_dir=config.extracted_dir,
            per_module_extracted_dirs=per_module_dirs,
            embed_images=config.embed_images,
            concept_graph=concept_graph,
            chapter_analyses=global_flat,
            course_title=blueprint.course_title,
            course_summary=blueprint.course_summary,
            learner_journey=blueprint.learner_journey,
            source_book_titles=source_book_titles,
            chapter_to_module=chapter_to_module,
            capabilities=capabilities,
        )
        index_path = render_course(modules, config.html_dir, render_ctx)

    total_elapsed = time.monotonic() - pipeline_t0
    total_elements = sum(len(m.all_elements) for m in modules)
    logger.info(
        "Pipeline complete in %.1fs (%d modules, %d elements). Open %s in a browser.",
        total_elapsed, len(modules), total_elements, index_path,
    )
    return index_path


# ── Checkpoint helpers ────────────────────────────────────────────────────────


def _migrate_legacy_elements(data: list[dict]) -> None:
    """Migrate legacy element types in raw JSON data (in-place).

    - milestone → interactive_essay (old name)
    - self_explain → interactive_essay (merged into single-prompt static essay)
    """
    for module in data:
        for section in module.get("sections", []):
            for element in section.get("elements", []):
                if element.get("element_type") == "milestone":
                    element["element_type"] = "interactive_essay"
                    if "milestone" in element:
                        element["interactive_essay"] = element.pop("milestone")
                elif element.get("element_type") == "self_explain":
                    se = element.pop("self_explain", {})
                    element["element_type"] = "interactive_essay"
                    element["interactive_essay"] = {
                        "title": "",
                        "concepts_tested": [],
                        "prompts": [se],
                        "passing_threshold": 0.7,
                        "tutor_system_prompt": "",
                    }


def _load_training_modules_checkpoint(path: Path) -> list[TrainingModule] | None:
    """Load training modules with legacy migration. Returns None if invalid."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        _migrate_legacy_elements(data)
        adapter = TypeAdapter(list[TrainingModule])
        return adapter.validate_python(data)
    except Exception:
        logger.debug("Invalid training modules checkpoint %s, will re-run", path)
        return None


# ── Load-or-run wrappers ─────────────────────────────────────────────────────


def _load_or_extract(
    extracted_dir: Path,
    input_sources: list[InputSource],
    client: LLMClient,
    resume: bool,
) -> list[Book]:
    """Load extraction checkpoint or run extraction."""
    checkpoint_path = extracted_dir / "book_structure.json"

    if resume:
        books = load_checkpoint(checkpoint_path, list[Book])
        if books is not None and books and any(b.chapters for b in books):
            total = sum(len(b.chapters) for b in books)
            logger.info("Loaded extraction checkpoint (%d chapters)", total)
            return books
        if checkpoint_path.exists():
            logger.warning("Invalid extraction checkpoint, re-running extraction")

    if len(input_sources) == 1:
        books = [extract_book(
            input_sources[0].path, extracted_dir, llm_client=client,
        )]
    else:
        books = extract_corpus(input_sources, extracted_dir, client)

    total_chapters = sum(len(b.chapters) for b in books)
    logger.info(
        "Extraction complete: %d document(s), %d total chapters",
        len(books), total_chapters,
    )

    save_checkpoint_raw(checkpoint_path, [dataclasses.asdict(b) for b in books])

    return books


def _load_or_analyze(
    extracted_dir: Path,
    books: list[Book],
    client: LLMClient,
    total_chapters: int,
    resume: bool,
    max_workers: int = 4,
) -> tuple[list[list[ChapterAnalysis]], ConceptGraph, list[ChapterAnalysis]]:
    """Load deep reading checkpoint or run analysis + consolidation.

    Returns:
        Tuple of (per_book_analyses, concept_graph, global_flat_analyses).
        ``per_book_analyses`` retains per-book chapter numbers for transformation.
        ``global_flat_analyses`` has globally unique chapter numbers (1..N) that
        match the concept graph — used for rendering only.
    """
    checkpoint_path = extracted_dir / "chapter_analyses.json"

    if resume:
        loaded = load_checkpoint(checkpoint_path, _AnalysesCheckpoint)
        if loaded is not None and len(loaded.chapter_analyses) == total_chapters:
            global_flat = loaded.chapter_analyses
            concept_graph = loaded.concept_graph
            all_chapter_analyses = _regroup_analyses_by_book(global_flat, books)
            logger.info(
                "Loaded deep reading checkpoint (%d analyses, %d concepts)",
                len(global_flat), len(concept_graph.concepts),
            )
            return all_chapter_analyses, concept_graph, global_flat
        if checkpoint_path.exists():
            logger.warning("Invalid deep reading checkpoint, re-running analysis")

    # Run deep reading
    all_chapter_analyses = [
        analyze_book(book, client, max_workers=max_workers)
        for book in books
    ]

    total_concepts = sum(
        len(a.concepts) for book_analyses in all_chapter_analyses for a in book_analyses
    )
    logger.info(
        "Deep reading complete: %d concepts across %d chapters",
        total_concepts, total_chapters,
    )

    # Stage 1.3: Consolidation (always re-run — it's instant)
    # In multi-doc mode, per-book chapter numbers overlap (both books have
    # "chapter 1"). Create copies with globally unique chapter numbers so the
    # concept graph can distinguish chapters across books.
    flat_analyses = [a for book_analyses in all_chapter_analyses for a in book_analyses]
    if len(books) > 1:
        global_flat = [
            a.model_copy(update={"chapter_number": i + 1})
            for i, a in enumerate(flat_analyses)
        ]
    else:
        global_flat = flat_analyses

    concept_graph = consolidate_concepts(global_flat)
    logger.info(
        "Concept graph: %d unique concepts, %d edges, %d foundation, %d advanced",
        len(concept_graph.concepts),
        len(concept_graph.edges),
        len(concept_graph.foundation_concepts),
        len(concept_graph.advanced_concepts),
    )

    save_checkpoint(
        checkpoint_path,
        _AnalysesCheckpoint(
            chapter_analyses=global_flat, concept_graph=concept_graph,
        ),
    )

    return all_chapter_analyses, concept_graph, global_flat


def _load_or_plan(
    extracted_dir: Path,
    books: list[Book],
    client: LLMClient,
    all_chapter_analyses: list[list[ChapterAnalysis]],
    concept_graph: ConceptGraph,
    resume: bool,
    document_type: str = "mixed",
) -> CurriculumBlueprint:
    """Load curriculum blueprint checkpoint or run planner."""
    checkpoint_path = extracted_dir / "curriculum_blueprint.json"

    if resume:
        blueprint = load_checkpoint(checkpoint_path, CurriculumBlueprint)
        if blueprint is not None and blueprint.modules:
            logger.info(
                "Loaded curriculum checkpoint (%d modules)", len(blueprint.modules),
            )
            return blueprint
        if checkpoint_path.exists():
            logger.warning("Invalid curriculum checkpoint, re-running planner")

    if len(books) == 1:
        blueprint = plan_curriculum(
            books[0], client,
            chapter_analyses=all_chapter_analyses[0],
            concept_graph=concept_graph,
            document_type=document_type,  # type: ignore[arg-type]
        )
    else:
        blueprint = plan_multi_document_curriculum(
            books, client,
            chapter_analyses_per_book=all_chapter_analyses,
            concept_graph=concept_graph,
            document_type=document_type,  # type: ignore[arg-type]
        )

    save_checkpoint(checkpoint_path, blueprint)

    return blueprint


def _load_or_transform(
    extracted_dir: Path,
    vision_enabled: bool,
    blueprint: CurriculumBlueprint,
    books: list[Book],
    client: LLMClient,
    analyses_by_book_chapter: dict[tuple[int, int], ChapterAnalysis],
    chapter_number: int | None,
    resume: bool,
    concept_graph: ConceptGraph | None = None,
    document_type: str | None = None,
    max_workers: int = 4,
    book_extracted_dirs: list[Path] | None = None,
    profile: ContentProfile | None = None,
    viz_enabled: bool = False,
) -> list[TrainingModule]:
    """Load training modules checkpoint or run transformation.

    Supports partial resume: if some chapters are already transformed,
    only transforms the missing ones.
    """
    training_path = extracted_dir / "training_modules.json"

    if resume:
        existing = _load_training_modules_checkpoint(training_path)
        if existing is not None:
            existing_chapters = {m.chapter_number for m in existing}
            needed_chapters = {
                bp.source_chapter_number
                for bp in blueprint.modules
                if bp.source_chapter_number is not None
            }
            missing = needed_chapters - existing_chapters

            # Detect chapters with fallback sections that need retry
            for mod in existing:
                if mod.chapter_number in needed_chapters and _has_fallback_sections(mod):
                    missing.add(mod.chapter_number)

            if not missing:
                logger.info(
                    "All %d modules found in checkpoint, skipping transformation",
                    len(existing),
                )
                return existing

            # Remove modules we're about to re-run so they don't duplicate
            existing = [m for m in existing if m.chapter_number not in missing]

            good_chapters = existing_chapters - missing
            logger.info(
                "Resuming transformation: %d/%d modules complete, %d to (re)generate",
                len(good_chapters),
                len(needed_chapters),
                len(missing),
            )

            new_modules = _transform_modules(
                blueprint, books, client, analyses_by_book_chapter,
                chapter_number, skip_chapters=good_chapters,
                training_path=training_path,
                concept_graph=concept_graph,
                document_type=document_type,
                book_extracted_dirs=book_extracted_dirs,
                vision_enabled=vision_enabled,
                max_workers=max_workers,
                existing_modules=existing,
                profile=profile,
                viz_enabled=viz_enabled,
            )
            all_modules = sorted(
                existing + new_modules, key=lambda m: m.chapter_number,
            )
            _save_training_json(all_modules, training_path)
            return all_modules

        logger.warning("Invalid training checkpoint, re-running transformation")

    modules = _transform_modules(
        blueprint, books, client, analyses_by_book_chapter,
        chapter_number, training_path=training_path,
        concept_graph=concept_graph,
        document_type=document_type,
        book_extracted_dirs=book_extracted_dirs,
        vision_enabled=vision_enabled,
        max_workers=max_workers,
        profile=profile,
        viz_enabled=viz_enabled,
    )
    _save_training_json(modules, training_path)
    return modules


# ── Internal helpers ──────────────────────────────────────────────────────────


def _has_fallback_sections(module: TrainingModule) -> bool:
    """Check if a module contains sections that failed LLM generation."""
    for section in module.sections:
        for note in section.verification_notes:
            if note.startswith("[error] Generation failed:") or note.startswith("[fallback:"):
                return True
        # Also detect legacy fallback slides (pre-tagging)
        if len(section.elements) == 1 and section.elements[0].element_type == "slide":
            slide = getattr(section.elements[0], "slide", None)
            if slide and "Content generation failed" in slide.content:
                return True
    return False


def _regroup_analyses_by_book(
    flat_analyses: list[ChapterAnalysis], books: list[Book],
) -> list[list[ChapterAnalysis]]:
    """Re-group flat analyses into per-book lists matching book chapter counts."""
    result: list[list[ChapterAnalysis]] = []
    idx = 0
    for book in books:
        count = len(book.chapters)
        result.append(flat_analyses[idx : idx + count])
        idx += count
    return result




def _precompute_cumulative_concepts(
    blueprint: CurriculumBlueprint,
    books: list[Book],
    analyses_by_book_chapter: dict[tuple[int, int], ChapterAnalysis],
    concept_graph: ConceptGraph | None = None,
) -> dict[int, list[str | dict]]:
    """Precompute cumulative_concepts for each module index.

    Since cumulative_concepts is derived from chapter analyses (available
    before transformation), we can compute it upfront and parallelize
    chapter transformations.

    Returns:
        Mapping from module index → cumulative concepts from all prior modules.
    """
    result: dict[int, list[str | dict]] = {}
    cumulative: list[str | dict] = []
    for idx, module_bp in enumerate(blueprint.modules):
        result[idx] = list(cumulative)
        book_idx = module_bp.source_book_index or 0
        if book_idx >= len(books):
            continue
        book = books[book_idx]
        chapter = find_matching_chapter(book, module_bp)
        if chapter is None:
            continue
        analysis = analyses_by_book_chapter.get(
            (book_idx, chapter.chapter_number)
        )
        if analysis:
            cumulative = cumulative + [
                {"name": concept_graph.resolve(c.name) if concept_graph else c.name,
                 "type": c.concept_type,
                 "importance": c.importance}
                for c in analysis.concepts
            ]
    return result


# ── Cross-book enrichment ────────────────────────────────────────────────────

_MAX_SNIPPET_LENGTH = 800
_MAX_SUPPLEMENTARY_LENGTH = 3000


def _build_cross_book_index(
    books: list[Book],
    analyses_by_book_chapter: dict[tuple[int, int], ChapterAnalysis],
    concept_graph: ConceptGraph | None = None,
) -> dict[str, list[tuple[int, int, str, str]]]:
    """Map concept names to their source locations across all books.

    Returns:
        {canonical_concept_name: [(book_idx, ch_num, section_title, text_snippet), ...]}
    """
    # Build section text lookup: (book_idx, section_title) → text snippet
    section_text: dict[tuple[int, str], str] = {}
    for book_idx, book in enumerate(books):
        for chapter in book.chapters:
            for section in chapter.sections:
                section_text[(book_idx, section.title)] = section.text[:_MAX_SNIPPET_LENGTH]

    index: dict[str, list[tuple[int, int, str, str]]] = {}
    for (book_idx, ch_num), analysis in analyses_by_book_chapter.items():
        for concept in analysis.concepts:
            canonical = concept_graph.resolve(concept.name) if concept_graph else concept.name
            snippet = section_text.get((book_idx, concept.section_title), "")
            entry = (book_idx, ch_num, concept.section_title, snippet)
            index.setdefault(canonical, []).append(entry)

    return index


def _compute_supplementary_contexts(
    work_items: list[tuple[int, ModuleBlueprint, Chapter, ChapterAnalysis | None]],
    books: list[Book],
    analyses_by_book_chapter: dict[tuple[int, int], ChapterAnalysis],
    concept_graph: ConceptGraph | None = None,
) -> dict[int, str]:
    """Compute cross-book supplementary context for each work item.

    For each module, finds related sections from OTHER books that cover
    overlapping concepts and assembles text snippets.

    Returns:
        {work_item_index: supplementary_context_string}
    """
    if len(books) <= 1:
        return {}

    cross_index = _build_cross_book_index(books, analyses_by_book_chapter, concept_graph)
    if not cross_index:
        return {}

    supplementary_by_idx: dict[int, str] = {}

    for idx, module_bp, _chapter, chapter_analysis in work_items:
        if not chapter_analysis:
            continue

        this_book_idx = module_bp.source_book_index or 0

        # Resolve concept names for this module
        module_concepts = {
            concept_graph.resolve(c.name) if concept_graph else c.name
            for c in chapter_analysis.concepts
        }

        # Collect related snippets from other books
        related_snippets: list[str] = []
        seen_sections: set[tuple[int, str]] = set()
        total_len = 0

        for concept_name in sorted(module_concepts):
            if concept_name not in cross_index:
                continue
            for bk_idx, _ch_num, sec_title, snippet in cross_index[concept_name]:
                if bk_idx == this_book_idx:
                    continue  # Same book — skip
                key = (bk_idx, sec_title)
                if key in seen_sections:
                    continue
                seen_sections.add(key)

                book_title = books[bk_idx].title if bk_idx < len(books) else f"Book {bk_idx}"
                entry = f"**{book_title}, '{sec_title}'** (covers: {concept_name}):\n{snippet}"

                if total_len + len(entry) > _MAX_SUPPLEMENTARY_LENGTH:
                    break
                related_snippets.append(entry)
                total_len += len(entry)

        if related_snippets:
            supplementary_by_idx[idx] = "\n\n".join(related_snippets)

    logger.info(
        "Cross-book enrichment: %d/%d modules have supplementary context",
        len(supplementary_by_idx), len(work_items),
    )
    return supplementary_by_idx


def _transform_modules(
    blueprint: CurriculumBlueprint,
    books: list[Book],
    client: LLMClient,
    analyses_by_book_chapter: dict[tuple[int, int], ChapterAnalysis],
    chapter_number: int | None,
    skip_chapters: set[int] | None = None,
    training_path: Path | None = None,
    concept_graph: ConceptGraph | None = None,
    document_type: str | None = None,
    book_extracted_dirs: list[Path] | None = None,
    vision_enabled: bool = False,
    max_workers: int = 4,
    existing_modules: list[TrainingModule] | None = None,
    profile: ContentProfile | None = None,
    viz_enabled: bool = False,
) -> list[TrainingModule]:
    """Transform blueprint modules into TrainingModules in parallel.

    Precomputes cumulative_concepts from chapter analyses (available upfront),
    then runs chapter transformations concurrently using a thread pool.
    Incrementally saves checkpoints as each chapter completes.

    Args:
        skip_chapters: Chapter numbers to skip (already checkpointed).
        training_path: If provided, incrementally saves after each chapter.
        concept_graph: Optional concept graph for resolving canonical names.
        document_type: Optional document type for prompt hints.
        max_workers: Maximum parallel chapter transformations.
        existing_modules: Previously-checkpointed modules to include in
            incremental saves (ensures crash safety during resume).
    """
    # Precompute cumulative concepts for each module from analyses
    cumulative_by_idx = _precompute_cumulative_concepts(
        blueprint, books, analyses_by_book_chapter, concept_graph,
    )

    # Build list of work items (module_index, module_bp, chapter, analysis)
    work_items: list[tuple[int, ModuleBlueprint, Chapter, ChapterAnalysis | None]] = []
    for idx, module_bp in enumerate(blueprint.modules):
        book_idx = module_bp.source_book_index or 0
        if book_idx >= len(books):
            continue
        book = books[book_idx]
        chapter = find_matching_chapter(book, module_bp)
        if chapter is None:
            logger.warning(
                "No matching chapter for module '%s' (book_index=%s, chapter=%s), skipping",
                module_bp.title, module_bp.source_book_index, module_bp.source_chapter_number,
            )
            continue
        if chapter_number is not None and chapter.chapter_number != chapter_number:
            continue
        if skip_chapters and chapter.chapter_number in skip_chapters:
            continue
        chapter_analysis = analyses_by_book_chapter.get(
            (book_idx, chapter.chapter_number)
        )
        work_items.append((idx, module_bp, chapter, chapter_analysis))

    if not work_items:
        return []

    # Compute cross-book supplementary context for multi-doc enrichment
    supplementary_by_idx = _compute_supplementary_contexts(
        work_items, books, analyses_by_book_chapter, concept_graph,
    )

    # Thread-safe list + lock for incremental checkpoint saving
    completed_modules: list[TrainingModule] = []
    save_lock = threading.Lock()

    # When multiple chapters run in parallel, sections within each chapter
    # run sequentially to avoid thread pool nesting (max_workers² concurrency).
    # When only 1 chapter is being processed, sections get the full pool.
    section_workers = 1 if len(work_items) > 1 else max_workers

    def _transform_one(
        item: tuple[int, ModuleBlueprint, Chapter, ChapterAnalysis | None],
    ) -> tuple[int, TrainingModule]:
        idx, module_bp, chapter, chapter_analysis = item
        cumulative_concepts = cumulative_by_idx.get(idx, [])

        # Resolve per-book extracted_dir for correct image paths
        book_idx = module_bp.source_book_index or 0
        per_book_dir = (
            book_extracted_dirs[book_idx]
            if book_extracted_dirs and book_idx < len(book_extracted_dirs)
            else None
        )

        # Collect additional source chapters for merged modules (multi-doc)
        additional_chapters: list[tuple[int, Chapter]] = []
        for asc in module_bp.additional_source_chapters:
            asc_book_idx = asc.get("book_index", 0)
            asc_ch_num = asc.get("chapter_number")
            if asc_book_idx < len(books) and asc_ch_num is not None:
                for ch in books[asc_book_idx].chapters:
                    if ch.chapter_number == asc_ch_num:
                        additional_chapters.append((asc_book_idx, ch))
                        break

        # Build per-book extracted dirs for additional chapters' images
        additional_extracted_dirs: dict[int, Path] = {}
        if additional_chapters and book_extracted_dirs:
            for asc_bk_idx, _ in additional_chapters:
                if asc_bk_idx < len(book_extracted_dirs):
                    additional_extracted_dirs[asc_bk_idx] = book_extracted_dirs[asc_bk_idx]

        ctx = TransformContext(
            blueprint=module_bp,
            chapter_analysis=chapter_analysis,
            prior_concepts=cumulative_concepts,
            document_type=document_type,
            extracted_dir=per_book_dir,
            vision_enabled=vision_enabled,
            max_workers=section_workers,
            supplementary_context=supplementary_by_idx.get(idx),
            additional_chapters=additional_chapters or None,
            additional_extracted_dirs=additional_extracted_dirs or None,
            canonical_map=concept_graph.canonical_map if concept_graph else None,
            profile=profile,
            viz_enabled=viz_enabled,
        )
        module = transform_chapter(chapter, client, ctx)
        logger.info(
            "Chapter %d transformed: %d elements",
            chapter.chapter_number,
            len(module.all_elements),
        )
        return (idx, module)

    # Run chapter transformations in parallel
    results: dict[int, TrainingModule] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_transform_one, item): item for item in work_items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                idx, module = future.result()
            except Exception as exc:
                logger.error(
                    "Chapter '%s' transformation failed: %s", item[1].title, exc,
                )
                continue
            results[idx] = module

            # Incremental checkpoint: save existing + all completed modules
            if training_path is not None:
                with save_lock:
                    completed_modules.append(module)
                    # Merge existing (from prior resume) with newly completed
                    all_so_far = list(existing_modules or []) + completed_modules
                    sorted_so_far = sorted(
                        all_so_far, key=lambda m: m.chapter_number,
                    )
                    _save_training_json(sorted_so_far, training_path)

    # Return in original blueprint order
    return [results[idx] for idx, _, _, _ in work_items if idx in results]


def compute_capabilities(
    modules: list[TrainingModule],
    concept_graph: ConceptGraph | None,
    chapter_analyses: list[ChapterAnalysis] | None,
    course_title: str | None,
) -> CourseCapabilities:
    """Compute what features are available in this generated course.

    Called after all pipeline stages complete, before rendering.
    """
    has_analyses = bool(chapter_analyses)

    element_types: set[str] = set()
    has_objectives = False
    for m in modules:
        for s in m.sections:
            for e in s.elements:
                element_types.add(e.element_type)
            if s.learning_objectives:
                has_objectives = True

    return CourseCapabilities(
        has_concept_graph=bool(concept_graph and concept_graph.concepts),
        has_mastery_tracking=has_analyses,
        has_chapter_review=bool(modules),
        has_mixed_review=bool(modules),
        has_course_metadata=bool(course_title),
        has_learning_objectives=has_objectives,
        chapter_count=len(modules),
        element_types_present=sorted(element_types),
    )


def _save_training_json(modules: list[TrainingModule], output_path: Path) -> None:
    """Serialize training modules to JSON checkpoint."""
    save_checkpoint_raw(
        output_path, [m.model_dump(mode="json") for m in modules],
    )
