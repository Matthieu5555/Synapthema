"""Mermaid diagram validation and LLM-based auto-fixing.

Validates mermaid diagram syntax using Node.js + mermaid's parse() API.
When a diagram has syntax errors, calls the LLM (light model) to fix it
based on the exact error message.  Falls back gracefully if Node.js or
LLM credentials are unavailable.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from src.rendering.html_generator import _fix_unicode_escapes

if TYPE_CHECKING:
    from src.protocols import LLMClient

logger = logging.getLogger(__name__)

_VALIDATOR_SCRIPT = Path(__file__).parent / "mermaid_validate.mjs"

MAX_FIX_ATTEMPTS = 2

NODE_TIMEOUT = 15


# ── Structured output model for LLM mermaid fixing ──────────────────────────


class MermaidFixResult(BaseModel):
    """LLM response when fixing a mermaid diagram."""

    fixed_diagram_code: str = Field(
        description="The corrected mermaid diagram code with valid syntax",
    )
    explanation: str = Field(
        default="",
        description="Brief explanation of what was fixed",
    )


# ── Validation ───────────────────────────────────────────────────────────────


class ValidationResult:
    """Result of mermaid syntax validation."""

    __slots__ = ("valid", "error_message")

    def __init__(self, valid: bool, error_message: str = "") -> None:
        self.valid = valid
        self.error_message = error_message


def _node_available() -> bool:
    """Check if Node.js is available on the system PATH."""
    return shutil.which("node") is not None


def _validator_ready() -> bool:
    """Check if the Node.js mermaid validation script and node_modules exist."""
    if not _VALIDATOR_SCRIPT.exists():
        return False
    node_modules = _VALIDATOR_SCRIPT.parent.parent.parent / "node_modules" / "mermaid"
    return node_modules.exists()


def validate_mermaid(diagram_code: str) -> ValidationResult:
    """Validate mermaid diagram syntax using Node.js.

    Returns ValidationResult with valid=True if syntax is correct,
    or valid=False with the error_message if syntax is invalid.
    Returns valid=True if Node.js or mermaid is unavailable (optimistic fallback).
    """
    if not _node_available() or not _validator_ready():
        return ValidationResult(valid=True)

    try:
        result = subprocess.run(
            ["node", str(_VALIDATOR_SCRIPT)],
            input=diagram_code,
            capture_output=True,
            text=True,
            timeout=NODE_TIMEOUT,
        )

        if result.returncode == 0:
            return ValidationResult(valid=True)

        error_msg = result.stderr.strip() or result.stdout.strip()
        return ValidationResult(valid=False, error_message=error_msg)

    except subprocess.TimeoutExpired:
        logger.warning("Mermaid validation timed out after %ds", NODE_TIMEOUT)
        return ValidationResult(valid=True)
    except OSError as exc:
        logger.warning("Failed to run mermaid validator: %s", exc)
        return ValidationResult(valid=True)


# ── LLM fixing ───────────────────────────────────────────────────────────────

_MERMAID_FIX_SYSTEM = """\
You are a Mermaid.js syntax expert.  Fix the syntax error in the provided
mermaid diagram code.

Rules:
- Return ONLY the corrected diagram code in the fixed_diagram_code field.
- Preserve the diagram's meaning and structure as closely as possible.
- Common issues to fix:
  - Unescaped special characters inside node labels — wrap the label text
    in double quotes, e.g. A["label with (parens) or special chars"].
  - Missing or mismatched brackets / parentheses.
  - Invalid arrow syntax (e.g. ->- instead of -->).
  - Unicode characters that mermaid cannot parse — replace with ASCII
    equivalents or wrap the label in quotes.
  - Malformed subgraph or participant declarations.
  - Newline issues within node labels.
- Keep the diagram type (flowchart, sequence, state, mindmap, etc.) the same.
- Keep diagrams simple and readable.
- Do NOT add nodes, edges, or features that were not in the original."""


def _build_fix_prompt(diagram_code: str, error_message: str) -> str:
    """Build the user prompt for the LLM mermaid fixer."""
    return (
        f"Fix this mermaid diagram.  The parser reported this error:\n\n"
        f"ERROR:\n{error_message}\n\n"
        f"ORIGINAL DIAGRAM:\n```mermaid\n{diagram_code}\n```\n\n"
        f"Return the fixed diagram code."
    )


def _build_fix_prompt_no_error(diagram_code: str) -> str:
    """Build the user prompt when no parser error is available (LLM-only mode)."""
    return (
        f"Check this mermaid diagram for syntax errors.  If it is valid, "
        f"return it EXACTLY as-is.  If there are any syntax errors, fix them.\n\n"
        f"Common issues: unescaped special characters in node labels (wrap in "
        f'quotes), parentheses inside [] brackets, invalid arrow syntax.\n\n'
        f"DIAGRAM:\n```mermaid\n{diagram_code}\n```\n\n"
        f"Return the fixed (or unchanged) diagram code."
    )


def fix_mermaid_with_llm(
    diagram_code: str,
    error_message: str | None,
    llm_client: LLMClient,
) -> str | None:
    """Ask the LLM to fix a mermaid diagram.

    Uses the light model and structured output for reliable extraction.
    If error_message is provided, includes it for context.

    Returns the fixed diagram code, or None if the LLM call fails.
    """
    prompt = (
        _build_fix_prompt(diagram_code, error_message)
        if error_message
        else _build_fix_prompt_no_error(diagram_code)
    )
    try:
        result = llm_client.complete_structured_light(
            system_prompt=_MERMAID_FIX_SYSTEM,
            user_prompt=prompt,
            response_model=MermaidFixResult,
        )
        return result.fixed_diagram_code
    except Exception as exc:
        logger.warning("LLM mermaid fix failed: %s", exc)
        return None


# ── Orchestrator ─────────────────────────────────────────────────────────────


def validate_and_fix_mermaid_diagrams(
    modules: list,  # list[TrainingModule] — avoids circular import
    llm_client: LLMClient | None = None,
) -> tuple[int, int, int]:
    """Validate all mermaid diagrams in modules and fix invalid ones via LLM.

    Modifies ``diagram.diagram_code`` in-place on MermaidElement objects.

    Args:
        modules: List of TrainingModule objects (modified in place).
        llm_client: Optional LLM client for fixing invalid diagrams.
            If None, invalid diagrams are logged but not fixed.

    Returns:
        Tuple of (total_diagrams, diagrams_fixed, diagrams_unfixable).
    """
    node_ok = _node_available() and _validator_ready()
    if not node_ok:
        logger.info("Node.js mermaid validator unavailable, using LLM-only mode")

    total = 0
    fixed = 0
    unfixable = 0

    for module in modules:
        for section in module.sections:
            for element in section.elements:
                if element.element_type != "mermaid":
                    continue

                total += 1
                diagram = element.mermaid

                # Apply unicode escape fix first
                cleaned = _fix_unicode_escapes(diagram.diagram_code)

                # Validate
                vr = validate_mermaid(cleaned)

                if vr.valid:
                    diagram.diagram_code = cleaned
                    continue

                logger.warning(
                    "Invalid mermaid '%s': %s",
                    diagram.title,
                    vr.error_message[:200],
                )

                if llm_client is None:
                    logger.info("No LLM client — skipping fix for '%s'", diagram.title)
                    diagram.diagram_code = cleaned
                    unfixable += 1
                    continue

                # Try to fix with LLM
                current_code = cleaned
                current_error: str | None = vr.error_message
                fix_succeeded = False

                for attempt in range(MAX_FIX_ATTEMPTS):
                    logger.info(
                        "LLM fix attempt %d/%d for '%s'",
                        attempt + 1,
                        MAX_FIX_ATTEMPTS,
                        diagram.title,
                    )

                    fixed_code = fix_mermaid_with_llm(
                        current_code, current_error, llm_client,
                    )

                    if fixed_code is None:
                        break

                    # Re-validate
                    fix_vr = validate_mermaid(fixed_code)
                    if fix_vr.valid:
                        diagram.diagram_code = fixed_code
                        fixed += 1
                        fix_succeeded = True
                        logger.info("Fixed mermaid '%s' on attempt %d", diagram.title, attempt + 1)
                        break

                    current_code = fixed_code
                    current_error = fix_vr.error_message

                if not fix_succeeded:
                    logger.warning(
                        "Could not fix mermaid '%s' after %d attempts",
                        diagram.title,
                        MAX_FIX_ATTEMPTS,
                    )
                    diagram.diagram_code = cleaned
                    unfixable += 1

    if total > 0:
        logger.info(
            "Mermaid validation: %d total, %d fixed, %d unfixable",
            total, fixed, unfixable,
        )

    return total, fixed, unfixable
