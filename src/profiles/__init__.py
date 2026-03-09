"""Content profiles — variant-specific pedagogical configuration.

A ContentProfile bundles all the settings that differ between product
variants (general-purpose vs scientific/math/physics):
- Prompt text and composition rules
- Template weights and selection
- Bloom-level mappings and constraints
- SectionResponse validation thresholds
- Concept type vocabulary

Public API:
- ContentProfile: The dataclass itself
- get_profile(variant): Factory that returns the right profile
"""

from src.profiles.base import ContentProfile
from src.profiles.general import GENERAL_PURPOSE_PROFILE
from src.profiles.scientific import SCIENTIFIC_PROFILE

_PROFILES: dict[str, ContentProfile] = {
    "general_purpose": GENERAL_PURPOSE_PROFILE,
    "scientific": SCIENTIFIC_PROFILE,
}


def get_profile(variant: str = "general_purpose") -> ContentProfile:
    """Return a ContentProfile for the given variant name.

    Args:
        variant: One of "general_purpose", "scientific", or "auto".
            "auto" currently maps to "general_purpose" — future work will
            detect the variant from the document content.

    Raises:
        ValueError: If the variant name is not recognised.
    """
    if variant == "auto":
        variant = "general_purpose"
    profile = _PROFILES.get(variant)
    if profile is None:
        raise ValueError(
            f"Unknown variant {variant!r}. "
            f"Available: {', '.join(sorted(_PROFILES))}"
        )
    return profile


__all__ = ["ContentProfile", "get_profile"]
