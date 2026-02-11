"""Streamlit GUI for the Learning Experience Generator.

Launch with: streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import webbrowser
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import streamlit as st
from dotenv import load_dotenv

from src.config import Config, ConfigError, InputSource, resolve_llm_provider
from src.pipeline import run_pipeline
from src.rendering.html_generator import render_course
from src.transformation.types import TrainingModule

# ── Load .env early ──────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent / ".env")

# ── Constants ────────────────────────────────────────────────────────────────

ELEMENT_TYPES = {
    "slide": "Slides (explanatory content)",
    "quiz": "Quizzes (multiple-choice)",
    "flashcard": "Flashcards (two-sided review)",
    "fill_in_the_blank": "Fill in the Blank",
    "matching": "Matching Exercises",
    "mermaid": "Diagrams (Mermaid.js)",
    "concept_map": "Concept Maps",
    "self_explain": "Self-Explanation",
    "interactive_essay": "Interactive Essays",
}


# ── Logging bridge ───────────────────────────────────────────────────────────


class StreamlitLogHandler(logging.Handler):
    """Routes pipeline log messages into a Streamlit status container."""

    def __init__(self, container: st.status) -> None:
        super().__init__(level=logging.INFO)
        self._container = container

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self._container.update(label=msg)
        self._container.write(msg)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _write_uploads_to_disk(
    uploaded_files: list[st.runtime.uploaded_file_manager.UploadedFile],
    dest_dir: Path,
) -> list[Path]:
    """Write Streamlit UploadedFile objects to real files on disk."""
    paths: list[Path] = []
    for uf in uploaded_files:
        target = dest_dir / uf.name
        target.write_bytes(uf.getbuffer())
        paths.append(target)
    return paths


def _filter_modules(
    modules: list[TrainingModule],
    enabled_types: set[str],
) -> list[TrainingModule]:
    """Remove training elements whose type is not in the enabled set."""
    for module in modules:
        for section in module.sections:
            section.elements = [
                e for e in section.elements if e.element_type in enabled_types
            ]
    return modules


def _deterministic_temp_dir(pdf_paths: list[Path]) -> Path:
    """Create a deterministic temp directory based on input file names."""
    names = sorted(p.name for p in pdf_paths)
    digest = hashlib.md5("_".join(names).encode()).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"lxp_{digest}"


def _zip_directory(directory: Path) -> bytes:
    """Create an in-memory ZIP of a directory's contents."""
    buf = BytesIO()
    with ZipFile(buf, "w") as zf:
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(directory))
    return buf.getvalue()


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Learning XP Generator",
    page_icon="📚",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")

    # -- PDF Upload --
    st.subheader("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Drag and drop one or more PDF files",
    )

    # -- Exercise Types --
    st.subheader("Exercise Types")
    col_a, col_b = st.columns(2)
    with col_a:
        select_all = st.button("Select All", use_container_width=True)
    with col_b:
        deselect_all = st.button("Deselect All", use_container_width=True)

    if select_all:
        for key in ELEMENT_TYPES:
            st.session_state[f"ex_{key}"] = True
    if deselect_all:
        for key in ELEMENT_TYPES:
            st.session_state[f"ex_{key}"] = False

    enabled_types: set[str] = set()
    for key, label in ELEMENT_TYPES.items():
        if st.checkbox(label, value=st.session_state.get(f"ex_{key}", True), key=f"ex_{key}"):
            enabled_types.add(key)

    # -- Settings --
    st.subheader("Settings")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Lower = more consistent, higher = more creative",
    )

    chapter_number = st.number_input(
        "Single Chapter (0 = all)",
        min_value=0,
        value=0,
        help="Process only this chapter number (0 to process all)",
    )

    document_type = st.selectbox(
        "Document type",
        options=["auto", "quantitative", "narrative", "procedural", "analytical", "regulatory", "mixed"],
        index=0,
        help="Auto-detect or manually specify the document type for template selection",
    )

    force_rerun = st.checkbox("Force re-run (ignore checkpoints)", value=False)

# ── Resolve LLM provider from .env ──────────────────────────────────────────

try:
    api_key, base_url, default_model = resolve_llm_provider()
    llm_configured = True
except ConfigError:
    api_key, base_url, default_model = "", "", ""
    llm_configured = False


# ── Main area ────────────────────────────────────────────────────────────────

st.title("Learning XP Generator")
st.markdown("Generate interactive training courses from PDF documents.")

can_generate = bool(uploaded_files) and llm_configured

if not llm_configured:
    st.error(
        "No LLM API key found. Add `OPENAI_KEY` or `OPENROUTER_KEY` to your `.env` file."
    )
elif not uploaded_files:
    st.info("Upload one or more PDF files in the sidebar to get started.")

if st.button(
    "Generate Course",
    type="primary",
    disabled=not can_generate,
    use_container_width=True,
):
    # Use deterministic temp dir so we can resume from checkpoints
    _upload_paths = [Path(uf.name) for uf in uploaded_files]
    tmp_dir = _deterministic_temp_dir(_upload_paths)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = tmp_dir / "pdfs"
    pdf_dir.mkdir(exist_ok=True)
    extracted_dir = tmp_dir / "extracted"
    output_dir = tmp_dir / "output"

    try:
        # Write uploaded PDFs to disk
        pdf_paths = _write_uploads_to_disk(uploaded_files, pdf_dir)

        # Build Config using .env provider settings
        model = os.getenv("LLM_MODEL", default_model)
        model_light = os.getenv("LLM_MODEL_LIGHT", model)
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "16384"))

        config = Config(
            input_sources=[InputSource(path=p) for p in pdf_paths],
            extracted_dir=extracted_dir,
            output_dir=output_dir,
            llm_api_key=api_key,
            llm_base_url=base_url,
            llm_model=model,
            llm_model_light=model_light,
            llm_temperature=temperature,
            llm_max_tokens=max_tokens,
            embed_images=os.getenv("EMBED_IMAGES", "true").lower() in ("true", "1", "yes"),
            document_type=document_type,
        )

        ch_num = int(chapter_number) if chapter_number > 0 else None
        resume = not force_rerun

        # Run pipeline with live log output
        with st.status("Starting pipeline...", expanded=True) as status:
            handler = StreamlitLogHandler(status)
            handler.setFormatter(logging.Formatter("%(message)s"))
            src_logger = logging.getLogger("src")
            src_logger.addHandler(handler)
            src_logger.setLevel(logging.INFO)

            try:
                index_path = run_pipeline(config, chapter_number=ch_num, resume=resume)
            finally:
                src_logger.removeHandler(handler)

            status.update(label="Pipeline complete!", state="complete")

        # Post-filter: remove disabled exercise types
        all_types_enabled = enabled_types == set(ELEMENT_TYPES.keys())
        if not all_types_enabled:
            training_json = extracted_dir / "training_modules.json"
            data = json.loads(training_json.read_text(encoding="utf-8"))
            modules = [TrainingModule.model_validate(m) for m in data]
            modules = _filter_modules(modules, enabled_types)

            index_path = render_course(
                modules=modules,
                output_dir=output_dir,
                extracted_dir=extracted_dir,
                embed_images=config.embed_images,
            )

        # Success output
        st.success(f"Course generated at: {index_path}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Open in Browser"):
                webbrowser.open(index_path.as_uri())

        with col2:
            zip_bytes = _zip_directory(output_dir)
            st.download_button(
                label="Download Course (ZIP)",
                data=zip_bytes,
                file_name="course.zip",
                mime="application/zip",
            )

    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        logging.getLogger(__name__).exception("Pipeline error")
