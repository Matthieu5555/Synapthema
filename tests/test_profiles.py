"""Tests for the content profiles system (variant abstraction)."""

import pytest

from src.profiles import ContentProfile, get_profile
from src.profiles.general import GENERAL_PURPOSE_PROFILE
from src.profiles.scientific import SCIENTIFIC_PROFILE


# ── get_profile factory ───────────────────────────────────────────────────────


class TestGetProfile:
    """Test profile factory lookup."""

    def test_general_purpose(self):
        profile = get_profile("general_purpose")
        assert profile is GENERAL_PURPOSE_PROFILE
        assert profile.name == "general_purpose"

    def test_scientific(self):
        profile = get_profile("scientific")
        assert profile is SCIENTIFIC_PROFILE
        assert profile.name == "scientific"

    def test_auto_maps_to_general(self):
        profile = get_profile("auto")
        assert profile is GENERAL_PURPOSE_PROFILE

    def test_default_is_general(self):
        profile = get_profile()
        assert profile is GENERAL_PURPOSE_PROFILE

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            get_profile("nonexistent")


# ── General-purpose profile ──────────────────────────────────────────────────


class TestGeneralPurposeProfile:
    """Test that the general-purpose profile preserves current defaults."""

    def test_no_domain_rules(self):
        assert GENERAL_PURPOSE_PROFILE.domain_rules == ""

    def test_no_bloom_supplements(self):
        assert GENERAL_PURPOSE_PROFILE.bloom_prompt_supplements == {}

    def test_no_template_weight_overrides(self):
        assert GENERAL_PURPOSE_PROFILE.template_weight_overrides == {}

    def test_default_validation_thresholds(self):
        p = GENERAL_PURPOSE_PROFILE
        assert p.min_exercises == 4
        assert p.min_exercise_types == 3
        assert p.max_quizzes == 1
        assert p.max_interactive_essays == 2


# ── Scientific profile ──────────────────────────────────────────────────────


class TestScientificProfile:
    """Test that the scientific profile has expected overrides."""

    def test_has_domain_rules(self):
        assert "Derivation" in SCIENTIFIC_PROFILE.domain_rules or \
               "derivation" in SCIENTIFIC_PROFILE.domain_rules.lower()

    def test_has_bloom_supplements(self):
        assert "apply" in SCIENTIFIC_PROFILE.bloom_prompt_supplements
        assert "analyze" in SCIENTIFIC_PROFILE.bloom_prompt_supplements

    def test_has_template_weight_overrides(self):
        assert "quantitative" in SCIENTIFIC_PROFILE.template_weight_overrides

    def test_extra_concept_types(self):
        assert "derivation" in SCIENTIFIC_PROFILE.extra_concept_types
        assert "law" in SCIENTIFIC_PROFILE.extra_concept_types

    def test_validation_thresholds_are_defaults(self):
        """Scientific profile uses same thresholds as general (for now)."""
        p = SCIENTIFIC_PROFILE
        assert p.min_exercises == 4
        assert p.min_exercise_types == 3


# ── ContentProfile dataclass ────────────────────────────────────────────────


class TestContentProfile:
    """Test ContentProfile construction and immutability."""

    def test_frozen(self):
        profile = ContentProfile(name="test")
        with pytest.raises(AttributeError):
            profile.name = "changed"  # type: ignore[misc]

    def test_custom_thresholds(self):
        profile = ContentProfile(
            name="custom",
            min_exercises=6,
            min_exercise_types=4,
            max_quizzes=2,
            max_interactive_essays=1,
        )
        assert profile.min_exercises == 6
        assert profile.min_exercise_types == 4
        assert profile.max_quizzes == 2
        assert profile.max_interactive_essays == 1

    def test_default_thresholds(self):
        profile = ContentProfile(name="bare")
        assert profile.min_exercises == 4
        assert profile.min_exercise_types == 3
        assert profile.max_quizzes == 1
        assert profile.max_interactive_essays == 2

    def test_domain_rules_default_empty(self):
        profile = ContentProfile(name="bare")
        assert profile.domain_rules == ""

    def test_bloom_supplements_default_empty(self):
        profile = ContentProfile(name="bare")
        assert profile.bloom_prompt_supplements == {}


# ── SectionResponse threshold integration ───────────────────────────────────


class TestSectionResponseThresholds:
    """Test that SectionResponse validators use configurable thresholds."""

    def test_default_thresholds(self):
        from src.transformation.types import SectionResponse
        assert SectionResponse._min_exercises == 4
        assert SectionResponse._min_exercise_types == 3
        assert SectionResponse._max_quizzes == 1
        assert SectionResponse._max_interactive_essays == 2

    def test_thresholds_are_class_level(self):
        """Thresholds can be temporarily overridden at class level."""
        from src.transformation.types import SectionResponse
        original = SectionResponse._min_exercises
        try:
            SectionResponse._min_exercises = 2
            assert SectionResponse._min_exercises == 2
        finally:
            SectionResponse._min_exercises = original
