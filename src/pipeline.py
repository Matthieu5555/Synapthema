"""Main pipeline orchestrator — wires Stages 1, 2, and 3 together.

Single public entry point: run_pipeline(). Extracts content from a PDF,
transforms it into interactive training elements via an LLM, and renders
the result as a self-contained HTML course.

Also provides main() as a CLI entry point.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import ValidationError

from src.config import Config, InputSource
from src.extraction.multi_doc import extract_corpus
from src.extraction.pdf_parser import extract_book
from src.extraction.types import Book, Chapter, ImageRef, Section, Table
from src.rendering.html_generator import render_course
from src.transformation.analysis_types import ChapterAnalysis, ConceptGraph
from src.transformation.concept_consolidator import consolidate_concepts
from src.transformation.content_designer import transform_chapter
from src.transformation.content_pre_analyzer import detect_document_type
from src.transformation.curriculum_planner import (
    find_matching_chapter,
    plan_curriculum,
    plan_multi_document_curriculum,
)
from src.transformation.deep_reader import analyze_book
from src.rendering.mermaid_validator import validate_and_fix_mermaid_diagrams
from src.transformation.llm_client import LLMClient, create_llm_client
from src.transformation.types import CurriculumBlueprint, ModuleBlueprint, TrainingModule

logger = logging.getLogger(__name__)


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
        )
    except Exception as exc:
        logger.warning("Failed to create LLM client for mermaid fixing: %s", exc)
        return None


def rerender_from_json(config: Config) -> Path:
    """Re-render HTML from an existing training_modules.json.

    Skips extraction and LLM transformation — loads the intermediate JSON
    saved by a previous pipeline run and goes straight to Stage 3 rendering.
    If chapter_analyses.json exists, loads the concept graph and analyses
    for concept tagging and graph visualization.

    Args:
        config: Pipeline config (only extracted_dir, output_dir, and
            embed_images are used; LLM fields are ignored).

    Returns:
        Path to the generated course index.html.
    """
    training_json = config.extracted_dir / "training_modules.json"
    logger.info("Render-only: loading %s", training_json)

    data = json.loads(training_json.read_text(encoding="utf-8"))
    _migrate_legacy_elements(data)
    modules = [TrainingModule.model_validate(m) for m in data]
    logger.info("Loaded %d modules from JSON", len(modules))

    # Validate and fix mermaid diagrams (optional — requires LLM + Node.js)
    llm_client = _maybe_create_llm_client(config)
    total, fixed, _ = validate_and_fix_mermaid_diagrams(modules, llm_client)
    if fixed > 0:
        _save_training_json(modules, training_json)

    # Load concept graph and analyses if available (from a prior pipeline run)
    concept_graph = None
    chapter_analyses = None
    analyses_json = config.extracted_dir / "chapter_analyses.json"
    if analyses_json.exists():
        try:
            analyses_data = json.loads(analyses_json.read_text(encoding="utf-8"))
            chapter_analyses = [
                ChapterAnalysis.model_validate(a)
                for a in analyses_data.get("chapter_analyses", [])
            ]
            concept_graph = ConceptGraph.model_validate(
                analyses_data.get("concept_graph", {}),
            )
            logger.info(
                "Loaded concept graph (%d concepts) and %d chapter analyses",
                len(concept_graph.concepts),
                len(chapter_analyses),
            )
        except (OSError, json.JSONDecodeError, KeyError, ValueError, ValidationError) as exc:
            logger.warning("Failed to load chapter_analyses.json, skipping: %s", exc)

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

    index_path = render_course(
        modules=modules,
        output_dir=config.html_dir,
        extracted_dir=config.extracted_dir,
        embed_images=config.embed_images,
        concept_graph=concept_graph,
        chapter_analyses=chapter_analyses,
        course_title=course_title,
        course_summary=course_summary,
        learner_journey=learner_journey,
        chapter_to_module=chapter_to_module,
    )

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
    )

    # Stage 1: Extraction
    books = _load_or_extract(config.extracted_dir, config.input_sources, client, resume)

    total_chapters = sum(len(b.chapters) for b in books)

    # Stage 1.25: Deep reading + Stage 1.3: Consolidation
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

    modules = _load_or_transform(
        config.extracted_dir, config.vision_enabled,
        blueprint, books, client, analyses_by_book_chapter,
        chapter_number, resume, concept_graph=concept_graph,
        document_type=document_type,
        max_workers=config.max_concurrent_llm,
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

    # Stage 3: Render (always re-run — it's instant)
    index_path = render_course(
        modules=modules,
        output_dir=config.html_dir,
        extracted_dir=config.extracted_dir,
        embed_images=config.embed_images,
        concept_graph=concept_graph,
        chapter_analyses=global_flat,
        course_title=blueprint.course_title,
        course_summary=blueprint.course_summary,
        learner_journey=blueprint.learner_journey,
        source_book_titles=source_book_titles,
        chapter_to_module=chapter_to_module,
    )

    logger.info("Pipeline complete. Open %s in a browser.", index_path)
    return index_path


# ── Checkpoint loaders ────────────────────────────────────────────────────────
# Each loader returns None if the checkpoint is missing or invalid.
# Invalid JSON is treated as "no checkpoint" — the stage re-runs.


def _load_books_checkpoint(path: Path) -> list[Book] | None:
    """Load extraction checkpoint. Returns None if invalid."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        books = [_dict_to_book(b) for b in data]
        if not books or not any(b.chapters for b in books):
            return None
        return books
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.debug("Invalid extraction checkpoint: %s", exc)
        return None


def _load_analyses_checkpoint(
    path: Path, expected_count: int,
) -> tuple[list[ChapterAnalysis], ConceptGraph] | None:
    """Load deep reading + concept graph checkpoint. Returns None if invalid."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        analyses = [
            ChapterAnalysis.model_validate(a)
            for a in data.get("chapter_analyses", [])
        ]
        if len(analyses) != expected_count:
            return None
        if "concept_graph" not in data:
            return None
        concept_graph = ConceptGraph.model_validate(data["concept_graph"])
        return analyses, concept_graph
    except (OSError, json.JSONDecodeError, KeyError, ValueError, ValidationError) as exc:
        logger.debug("Invalid analyses checkpoint: %s", exc)
        return None


def _load_blueprint_checkpoint(path: Path) -> CurriculumBlueprint | None:
    """Load curriculum blueprint checkpoint. Returns None if invalid."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        blueprint = CurriculumBlueprint.model_validate(data)
        if not blueprint.modules:
            return None
        return blueprint
    except (OSError, json.JSONDecodeError, KeyError, ValueError, ValidationError) as exc:
        logger.debug("Invalid blueprint checkpoint: %s", exc)
        return None


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
    """Load training modules checkpoint. Returns None if invalid.

    Handles backward compatibility for legacy element types.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        _migrate_legacy_elements(data)
        return [TrainingModule.model_validate(m) for m in data]
    except (OSError, json.JSONDecodeError, KeyError, ValueError, ValidationError) as exc:
        logger.debug("Invalid training modules checkpoint: %s", exc)
        return None


# ── Load-or-run wrappers ─────────────────────────────────────────────────────


def _load_or_extract(
    extracted_dir: Path,
    input_sources: list[InputSource],
    client: LLMClient,
    resume: bool,
) -> list[Book]:
    """Load extraction checkpoint or run extraction."""
    book_structure_path = extracted_dir / "book_structure.json"

    if resume and book_structure_path.exists():
        books = _load_books_checkpoint(book_structure_path)
        if books is not None:
            total = sum(len(b.chapters) for b in books)
            logger.info(
                "Loaded extraction checkpoint (%d chapters)", total,
            )
            return books
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

    # Save extraction checkpoint
    _save_books_json(books, book_structure_path)

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
    analyses_path = extracted_dir / "chapter_analyses.json"

    if resume and analyses_path.exists():
        result = _load_analyses_checkpoint(analyses_path, total_chapters)
        if result is not None:
            global_flat, concept_graph = result
            # Re-group by book for downstream consumers
            all_chapter_analyses = _regroup_analyses_by_book(global_flat, books)
            logger.info(
                "Loaded deep reading checkpoint (%d analyses, %d concepts)",
                len(global_flat), len(concept_graph.concepts),
            )
            return all_chapter_analyses, concept_graph, global_flat
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

    _save_analyses_json(global_flat, concept_graph, analyses_path)

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
    blueprint_path = extracted_dir / "curriculum_blueprint.json"

    if resume and blueprint_path.exists():
        blueprint = _load_blueprint_checkpoint(blueprint_path)
        if blueprint is not None:
            logger.info(
                "Loaded curriculum checkpoint (%d modules)", len(blueprint.modules),
            )
            return blueprint
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

    _save_blueprint_json(blueprint, blueprint_path)

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
) -> list[TrainingModule]:
    """Load training modules checkpoint or run transformation.

    Supports partial resume: if some chapters are already transformed,
    only transforms the missing ones.
    """
    training_path = extracted_dir / "training_modules.json"

    if resume and training_path.exists():
        existing = _load_training_modules_checkpoint(training_path)
        if existing is not None:
            existing_chapters = {m.chapter_number for m in existing}
            needed_chapters = {
                bp.source_chapter_number
                for bp in blueprint.modules
                if bp.source_chapter_number is not None
            }
            missing = needed_chapters - existing_chapters

            if not missing:
                logger.info(
                    "All %d modules found in checkpoint, skipping transformation",
                    len(existing),
                )
                return existing

            logger.info(
                "Resuming transformation: %d/%d modules complete, %d remaining",
                len(existing_chapters & needed_chapters),
                len(needed_chapters),
                len(missing),
            )

            new_modules = _transform_modules(
                blueprint, books, client, analyses_by_book_chapter,
                chapter_number, skip_chapters=existing_chapters,
                training_path=training_path,
                concept_graph=concept_graph,
                document_type=document_type,
                extracted_dir=extracted_dir,
                vision_enabled=vision_enabled,
                max_workers=max_workers,
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
        extracted_dir=extracted_dir,
        vision_enabled=vision_enabled,
        max_workers=max_workers,
    )
    _save_training_json(modules, training_path)
    return modules


# ── Internal helpers ──────────────────────────────────────────────────────────


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


def _save_books_json(books: list[Book], output_path: Path) -> None:
    """Save extraction output as a checkpoint."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [dataclasses.asdict(b) for b in books]
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("Saved extraction checkpoint to %s", output_path)


def _dict_to_section(d: dict) -> Section:
    """Reconstruct a Section from a dictionary (checkpoint loading)."""
    return Section(
        title=d["title"],
        level=d["level"],
        start_page=d["start_page"],
        end_page=d["end_page"],
        text=d["text"],
        images=tuple(
            ImageRef(
                path=Path(img["path"]),
                page=img["page"],
                caption=img["caption"],
                bbox=tuple(img["bbox"]),
            )
            for img in d.get("images", ())
        ),
        tables=tuple(
            Table(
                page=t["page"],
                headers=tuple(t["headers"]),
                rows=tuple(tuple(r) for r in t["rows"]),
            )
            for t in d.get("tables", ())
        ),
        subsections=tuple(
            _dict_to_section(sub) for sub in d.get("subsections", ())
        ),
    )


def _dict_to_book(d: dict) -> Book:
    """Reconstruct a Book from a dictionary (checkpoint loading)."""
    return Book(
        title=d["title"],
        author=d["author"],
        total_pages=d["total_pages"],
        chapters=tuple(
            Chapter(
                chapter_number=ch["chapter_number"],
                title=ch["title"],
                start_page=ch["start_page"],
                end_page=ch["end_page"],
                sections=tuple(
                    _dict_to_section(s) for s in ch.get("sections", ())
                ),
            )
            for ch in d.get("chapters", ())
        ),
    )


def _save_analyses_json(
    analyses: list[ChapterAnalysis],
    concept_graph: ConceptGraph,
    output_path: Path,
) -> None:
    """Save deep reading analyses and concept graph to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "chapter_analyses": [a.model_dump(mode="json") for a in analyses],
        "concept_graph": concept_graph.model_dump(mode="json"),
    }
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved deep reading analyses to %s", output_path)


def _save_blueprint_json(blueprint: CurriculumBlueprint, output_path: Path) -> None:
    """Save curriculum blueprint as a checkpoint."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(blueprint.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved curriculum blueprint to %s", output_path)


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
    extracted_dir: Path | None = None,
    vision_enabled: bool = False,
    max_workers: int = 4,
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

        module = transform_chapter(
            chapter, client,
            blueprint=module_bp,
            chapter_analysis=chapter_analysis,
            prior_concepts=cumulative_concepts,
            document_type=document_type,
            extracted_dir=extracted_dir,
            vision_enabled=vision_enabled,
            max_workers=section_workers,
        )
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

            # Incremental checkpoint: save all completed modules so far
            if training_path is not None:
                with save_lock:
                    completed_modules.append(module)
                    # Save sorted by original blueprint order
                    sorted_so_far = sorted(
                        completed_modules, key=lambda m: m.chapter_number,
                    )
                    _save_training_json(sorted_so_far, training_path)

    # Return in original blueprint order
    return [results[idx] for idx, _, _, _ in work_items if idx in results]


def _save_training_json(modules: list[TrainingModule], output_path: Path) -> None:
    """Serialize training modules to JSON for inspection and debugging."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [m.model_dump(mode="json") for m in modules]
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved intermediate training JSON to %s", output_path)
