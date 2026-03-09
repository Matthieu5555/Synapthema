"""ContentProfile dataclass — the variant abstraction.

All variant-specific pedagogical decisions are captured here so that
the rest of the codebase is variant-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ContentProfile:
    """Bundles all variant-specific configuration for content generation.

    The pipeline constructs a profile from the config variant, then
    threads it through every stage that needs to know "how should I
    teach this content?"
    """

    # ── Identity ─────────────────────────────────────────────────────────
    name: str

    # ── Prompt composition ───────────────────────────────────────────────
    # Domain-specific rules appended to the shared base system prompt.
    # Example: scientific adds "Always show derivation steps" etc.
    domain_rules: str = ""

    # Per-Bloom-level prompt supplements (overrides the defaults).
    bloom_prompt_supplements: dict[str, str] = field(default_factory=dict)

    # Template description overrides.  Keys present here replace the
    # corresponding entry in the shared TEMPLATE_DESCRIPTIONS dict.
    template_description_overrides: dict[str, str] = field(default_factory=dict)

    # ── Template selection ───────────────────────────────────────────────
    # Per-document-type template weight overrides.  Keys present here
    # replace the corresponding entry in DOCUMENT_TYPE_TEMPLATE_WEIGHTS.
    template_weight_overrides: dict[str, dict[str, float]] = field(
        default_factory=dict
    )

    # ── Element/Bloom mapping ────────────────────────────────────────────
    # Overrides for the default ELEMENT_BLOOM_MAP.
    element_bloom_overrides: dict[str, str] = field(default_factory=dict)

    # Subset of element types available in this variant.
    # None means "all types" (the default).
    available_element_types: frozenset[str] | None = None

    # ── SectionResponse validation thresholds ────────────────────────────
    min_exercises: int = 4
    min_exercise_types: int = 3
    max_quizzes: int = 1
    max_interactive_essays: int = 2

    # ── Concept vocabulary ───────────────────────────────────────────────
    # Extra concept types beyond the base set (defined in analysis_types.py).
    extra_concept_types: tuple[str, ...] = ()
