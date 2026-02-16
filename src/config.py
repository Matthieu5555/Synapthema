"""Centralized, validated configuration for the learningxp pipeline.

Loads settings from environment variables (via .env) with sensible defaults.
All configuration is validated at startup — misconfiguration fails fast with
clear error messages rather than causing subtle bugs deep in execution.

Supports two LLM providers:
- OpenAI directly (set OPENAI_KEY)
- OpenRouter (set OPENROUTER_KEY)
If both are set, OpenAI takes precedence.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import os
import logging
import re

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Base directory — all relative paths resolve from the project root
# (one level up from src/).
_PROJECT_ROOT = Path(__file__).parent.parent


@dataclass(frozen=True)
class InputSource:
    """A single input document."""

    path: Path
    source_type: Literal["pdf"] = "pdf"


@dataclass(frozen=True)
class Config:
    """Immutable pipeline configuration.

    Fields:
        input_sources: List of input documents to process.
        extracted_dir: Directory for intermediate extraction output (JSON + images).
            Defaults to output_dir / "json".
        output_dir: Root output directory. Contains json/ and html/ subdirectories.
        llm_api_key: API key for the LLM provider (OpenAI or OpenRouter).
        llm_base_url: Base URL for the LLM API endpoint.
            "https://api.openai.com/v1" for OpenAI,
            "https://openrouter.ai/api/v1" for OpenRouter.
        llm_model: Model identifier (e.g. "gpt-5.2", "openai/gpt-oss-120b").
            Changing this affects cost, speed, and output quality of Stage 2.
        llm_model_light: Secondary model for low-complexity tasks (TOC detection,
            target selection). Falls back to llm_model if empty.
        llm_temperature: Sampling temperature for LLM responses (0.0 = deterministic,
            1.0 = creative). Lower values produce more consistent training content.
        llm_max_tokens: Maximum tokens in LLM response. Must be large enough to hold
            a full chapter's worth of training elements as JSON (~4000 for short chapters,
            ~8000 for long ones).
        embed_images: If True, images are base64-encoded inline in HTML output.
            If False, images are referenced as relative file paths (requires serving
            from the output directory).
        vision_enabled: If True, extracted images are sent to the LLM as vision
            content during transformation (Stage 2). Requires a vision-capable model.
            If False, the LLM receives only text metadata about images.
        max_concurrent_llm: Maximum number of parallel LLM calls. Controls
            thread pool size for deep reading, section transformation, and
            chapter processing. Set to 1 for sequential execution.
    """

    input_sources: list[InputSource]
    extracted_dir: Path
    output_dir: Path
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_model_light: str
    llm_temperature: float
    llm_max_tokens: int
    embed_images: bool
    vision_enabled: bool
    document_type: str
    max_concurrent_llm: int

    @property
    def pdf_path(self) -> Path:
        """Backward-compatible access to the first input PDF path."""
        return self.input_sources[0].path

    @property
    def html_dir(self) -> Path:
        """Directory for rendered HTML output (output_dir / 'html')."""
        return self.output_dir / "html"


class ConfigError(Exception):
    """Raised when configuration is invalid or incomplete."""


def load_config(
    pdf_path: Path | None = None,
    pdf_paths: list[Path] | None = None,
    input_dir: Path | None = None,
    extracted_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Config:
    """Load and validate pipeline configuration from environment and defaults.

    Detects the LLM provider from environment variables:
    - OPENAI_KEY → OpenAI direct (default model: gpt-5.2)
    - OPENROUTER_KEY → OpenRouter (default model: openai/gpt-5.2)

    Args:
        pdf_path: Single PDF path (backward compat). Wrapped into a list.
        pdf_paths: Multiple PDF paths. Takes precedence over pdf_path.
        input_dir: Directory to scan for PDF files.
        extracted_dir: Override for extraction output directory.
        output_dir: Override for HTML output directory.

    Returns:
        Validated, frozen Config instance.

    Raises:
        ConfigError: If required configuration is missing or invalid.
    """
    load_dotenv(_PROJECT_ROOT / ".env")

    api_key, base_url, default_model = resolve_llm_provider()

    sources = _resolve_input_sources(pdf_path, pdf_paths, input_dir)

    slug = _slugify_multi_source(sources)
    resolved_output = output_dir or _PROJECT_ROOT / "output" / slug
    resolved_extracted = extracted_dir or resolved_output / "json"

    llm_model = os.getenv("LLM_MODEL", default_model)
    llm_model_light = os.getenv("LLM_MODEL_LIGHT", llm_model)

    # Temperature — 0.3 balances consistency with some creativity for training content
    llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    # Max tokens — 16384 accommodates most chapter transformations
    llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "16384"))

    # Image embedding — default True for fully self-contained HTML output
    embed_images = os.getenv("EMBED_IMAGES", "true").lower() in ("true", "1", "yes")

    # Vision — send extracted images to the LLM as vision content during Stage 2
    vision_enabled = os.getenv("VISION_ENABLED", "true").lower() in ("true", "1", "yes")

    # Document type override — "auto" means detect from content at runtime
    document_type = os.getenv("DOCUMENT_TYPE", "auto").lower()

    # Concurrency — max parallel LLM calls (deep reading, section transformation)
    max_concurrent_llm = int(os.getenv("MAX_CONCURRENT_LLM", "4"))

    config = Config(
        input_sources=sources,
        extracted_dir=resolved_extracted,
        output_dir=resolved_output,
        llm_api_key=api_key,
        llm_base_url=base_url,
        llm_model=llm_model,
        llm_model_light=llm_model_light,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        embed_images=embed_images,
        vision_enabled=vision_enabled,
        document_type=document_type,
        max_concurrent_llm=max_concurrent_llm,
    )

    source_names = ", ".join(s.path.name for s in sources)
    logger.info(
        "Configuration loaded: provider=%s, model=%s, model_light=%s, sources=[%s]",
        "openai" if "api.openai.com" in base_url else "openrouter",
        config.llm_model,
        config.llm_model_light,
        source_names,
    )
    return config


def load_render_config(
    pdf_path: Path | None = None,
    pdf_paths: list[Path] | None = None,
    input_dir: Path | None = None,
    extracted_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Config:
    """Load configuration for render-only mode (no LLM key required).

    Re-renders HTML from an existing training_modules.json without running
    extraction or LLM transformation. Only needs directory paths and the
    embed_images setting.

    Raises:
        ConfigError: If the training_modules.json file is not found.
    """
    load_dotenv(_PROJECT_ROOT / ".env")

    sources = _resolve_input_sources(pdf_path, pdf_paths, input_dir)

    slug = _slugify_multi_source(sources)
    resolved_output = output_dir or _PROJECT_ROOT / "output" / slug
    resolved_extracted = extracted_dir or resolved_output / "json"

    training_json = resolved_extracted / "training_modules.json"
    if not training_json.exists():
        raise ConfigError(
            f"No training_modules.json found at {training_json}. "
            "Run the full pipeline first before using --render-only."
        )

    embed_images = os.getenv("EMBED_IMAGES", "true").lower() in ("true", "1", "yes")

    # Optionally load LLM credentials for render-time features (mermaid fixing).
    # If no API key is found, mermaid auto-fixing is silently disabled.
    llm_api_key = llm_base_url = llm_model = llm_model_light = ""
    llm_temperature = 0.3
    llm_max_tokens = 4096
    try:
        api_key, base_url, default_model = resolve_llm_provider()
        llm_api_key = api_key
        llm_base_url = base_url
        llm_model = os.getenv("LLM_MODEL", default_model)
        llm_model_light = os.getenv("LLM_MODEL_LIGHT", llm_model)
        logger.info("LLM credentials found — mermaid auto-fixing enabled")
    except ConfigError:
        logger.info("No LLM credentials — mermaid auto-fixing disabled")

    config = Config(
        input_sources=sources,
        extracted_dir=resolved_extracted,
        output_dir=resolved_output,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        llm_model_light=llm_model_light,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        embed_images=embed_images,
        vision_enabled=False,
        document_type="auto",
        max_concurrent_llm=1,
    )

    logger.info("Render-only config loaded: extracted=%s, output=%s", resolved_extracted, resolved_output)
    return config


def _resolve_input_sources(
    pdf_path: Path | None,
    pdf_paths: list[Path] | None,
    input_dir: Path | None,
) -> list[InputSource]:
    """Build the list of InputSource from the various input options.

    Priority: pdf_paths > input_dir > pdf_path > auto-detect in project root.

    Raises:
        ConfigError: If no valid input sources are found.
    """
    paths: list[Path] = []

    if pdf_paths:
        paths = list(pdf_paths)
    elif input_dir:
        if not input_dir.is_dir():
            raise ConfigError(f"Input directory not found: {input_dir}")
        paths = sorted(
            list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.PDF")),
            key=lambda p: p.name.lower(),
        )
        if not paths:
            raise ConfigError(f"No PDF files found in {input_dir}")
    elif pdf_path:
        paths = [pdf_path]
    else:
        paths = [_find_pdf_in_project_root()]

    sources: list[InputSource] = []
    for p in paths:
        resolved = p.resolve() if not p.is_absolute() else p
        if not resolved.exists():
            raise ConfigError(f"PDF not found at {resolved}")
        sources.append(InputSource(path=resolved))

    return sources


def resolve_llm_provider() -> tuple[str, str, str]:
    """Detect which LLM provider to use from environment variables.

    Returns:
        Tuple of (api_key, base_url, default_model).

    Raises:
        ConfigError: If no API key is found.
    """
    openai_key = os.getenv("OPENAI_KEY", "")
    openrouter_key = os.getenv("OPENROUTER_KEY", "")

    if openai_key:
        return openai_key, "https://api.openai.com/v1", "gpt-5.2"

    if openrouter_key:
        return openrouter_key, "https://openrouter.ai/api/v1", "openai/gpt-5.2"

    raise ConfigError(
        "No LLM API key found. Set OPENAI_KEY or OPENROUTER_KEY in .env.\n"
        "  OpenAI: https://platform.openai.com/api-keys\n"
        "  OpenRouter: https://openrouter.ai/keys"
    )


def _slugify_pdf_name(pdf_path: Path) -> str:
    """Derive a filesystem-safe slug from a PDF filename.

    Examples:
        "cfa-program2026L2V6.PDF" → "cfa-program2026l2v6"
        "Quant Finance With Python.pdf" → "quant-finance-with-python"
    """
    stem = pdf_path.stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return slug or "unknown"


def _slugify_multi_source(sources: list[InputSource]) -> str:
    """Build a slug from one or more input sources.

    Single source: uses the PDF filename slug (e.g. "quant-finance-with-python").
    Multiple sources: joins slugs with "+" and truncates to keep filesystem paths
    reasonable (max 80 chars).

    Examples:
        [quant-finance.pdf] → "quant-finance"
        [frm-handbook.pdf, garp-quant.pdf] → "frm-handbook+garp-quant"
    """
    slugs = [_slugify_pdf_name(s.path) for s in sources]
    combined = "+".join(slugs)
    if len(combined) > 80:
        combined = combined[:80].rsplit("+", 1)[0]
    return combined or "unknown"


def _find_pdf_in_project_root() -> Path:
    """Find the first .pdf file in the project root directory.

    Raises:
        ConfigError: If no PDF files are found.
    """
    pdf_files = sorted(
        list(_PROJECT_ROOT.glob("*.pdf")) + list(_PROJECT_ROOT.glob("*.PDF")),
        key=lambda p: p.name.lower(),
    )
    if not pdf_files:
        raise ConfigError(
            f"No PDF files found in {_PROJECT_ROOT}. "
            "Place a PDF in the project root or pass pdf_path explicitly."
        )
    if len(pdf_files) > 1:
        logger.warning(
            "Multiple PDFs found in project root, using first: %s", pdf_files[0].name
        )
    return pdf_files[0]
