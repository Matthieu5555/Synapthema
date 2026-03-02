"""CLI entry point for the learningxp pipeline.

Usage patterns:
    uv run main.py doc1.pdf                          # Single PDF (backward compat)
    uv run main.py doc1.pdf doc2.pdf doc3.pdf        # Multiple PDFs
    uv run main.py --input-dir ./materials/           # Directory of PDFs
    uv run main.py doc1.pdf --chapter 3               # Single-doc, single chapter
    uv run main.py doc1.pdf doc2.pdf --render-only    # Re-render from existing JSON
    uv run main.py doc1.pdf --resume                  # Resume from last checkpoint
    uv run main.py doc1.pdf                            # Default: full run (no resume)
    uv run main.py doc1.pdf --exclude interactive_essay       # Skip interactive essays
    uv run main.py doc1.pdf --exclude interactive_essay self_explain  # Skip multiple types
"""

import argparse
import logging
import sys
from pathlib import Path

from src.config import load_config, load_render_config
from src.pipeline import rerender_from_json, run_pipeline

ELEMENT_TYPES = [
    "slide", "quiz", "flashcard", "fill_in_the_blank",
    "matching", "mermaid", "concept_map", "self_explain", "interactive_essay",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate interactive training from PDF documents.",
    )
    parser.add_argument(
        "pdfs", nargs="*", type=Path,
        help="Path(s) to PDF file(s). Multiple PDFs produce a unified course.",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Directory containing PDF files to process (alternative to listing PDFs)",
    )
    parser.add_argument(
        "--chapter", type=int, default=None,
        help="Only process this chapter number (single-doc mode only)",
    )
    parser.add_argument(
        "--render-only", action="store_true",
        help="Re-render HTML from existing training_modules.json (no LLM needed)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint (skip stages whose output already exists)",
    )
    parser.add_argument(
        "--exclude", nargs="+", metavar="TYPE", choices=ELEMENT_TYPES,
        help="Element types to exclude from output. "
        f"Choices: {', '.join(ELEMENT_TYPES)}",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.pdfs and not args.input_dir:
        parser.error("Provide at least one PDF file or use --input-dir")

    # Build config kwargs based on input mode
    config_kwargs: dict = {}
    if args.input_dir:
        config_kwargs["input_dir"] = args.input_dir.resolve()
    elif len(args.pdfs) == 1:
        config_kwargs["pdf_path"] = args.pdfs[0].resolve()
    else:
        config_kwargs["pdf_paths"] = [p.resolve() for p in args.pdfs]

    if args.render_only:
        config = load_render_config(**config_kwargs)
        index = rerender_from_json(config)
    else:
        config = load_config(**config_kwargs)
        index = run_pipeline(
            config,
            chapter_number=args.chapter,
            resume=args.resume,
        )

    # Post-filter: re-render with excluded element types removed
    if args.exclude:
        render_config = load_render_config(**config_kwargs)
        index = rerender_from_json(
            render_config,
            exclude_element_types=set(args.exclude),
        )

    print(f"\nDone. Open {index}")


if __name__ == "__main__":
    main()
