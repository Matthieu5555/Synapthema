"""General-purpose content profile — the default variant.

Captures the current behaviour as an explicit profile.
All values here were previously hardcoded across content_designer.py,
content_designer_prompts.py, content_pre_analyzer.py, and types.py.
"""

from src.profiles.base import ContentProfile

GENERAL_PURPOSE_PROFILE = ContentProfile(
    name="general_purpose",
    # No domain-specific rules — the base system prompt is sufficient.
    domain_rules="",
    # Default Bloom supplements, template weights, and element mappings
    # are used as-is (no overrides needed).
)
