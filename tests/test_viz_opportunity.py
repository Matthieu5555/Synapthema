"""Tests for visualization opportunity detection."""

import pytest
from pydantic import TypeAdapter, ValidationError

from src.transformation.viz_opportunity import (
    NoVisualization,
    VisualizationOpportunity,
    VizTriageResult,
    viz_prefilter,
    build_viz_triage_prompt,
    build_viz_generation_prompt,
)


class TestVizPrefilter:
    """Tests for the rule-based pre-filter (Layer 1)."""

    def test_short_text_rejected(self) -> None:
        """Sections under 200 words should always be rejected."""
        short = "This is a very short section about bonds."
        assert viz_prefilter(short) is False

    def test_definition_only_rejected(self) -> None:
        """Pure definitions without parameter relationships should fail."""
        text = " ".join(["A bond is a debt instrument."] * 50)
        assert viz_prefilter(text) is False

    def test_parameter_relationship_passes(self) -> None:
        """Text with formulas and parameter relationships should pass."""
        text = (
            "The bond price equation shows the relationship between yield and price. "
            "As yield increases, the price decreases. The formula for bond pricing is "
            "P = C/(1+r) + C/(1+r)^2 + ... + (C+F)/(1+r)^n. This trade-off between "
            "yield and price is fundamental to fixed income investing. "
            "The curve shows an inverse relationship. "
        ) * 10  # Make it long enough
        assert viz_prefilter(text) is True

    def test_process_steps_passes(self) -> None:
        """Text describing a multi-step process should pass."""
        text = (
            "The settlement process involves several steps. "
            "Step 1: The trade is executed on the exchange. "
            "Step 2: The clearing house confirms the trade details. "
            "Step 3: Securities are transferred from seller to buyer. "
            "The equilibrium between supply and demand determines the price. "
        ) * 10
        assert viz_prefilter(text) is True

    def test_comparison_language_passes(self) -> None:
        """Text with comparison language and distributions should pass."""
        text = (
            "Compared to equity markets, bond markets exhibit different characteristics. "
            "The distribution of returns shows a negative skew. "
            "The graph illustrates how credit spreads widen versus treasury yields "
            "during periods of market stress. The curve is exponential in nature. "
        ) * 10
        assert viz_prefilter(text) is True


class TestVizTriageModels:
    """Tests for the Pydantic triage models (Layer 2)."""

    _adapter = TypeAdapter(VizTriageResult)

    def test_skip_decision(self) -> None:
        result = self._adapter.validate_python({
            "decision": "skip",
            "reason": "Simple definition, no interactive parameters",
        })
        assert isinstance(result, NoVisualization)

    def test_visualize_decision(self) -> None:
        result = self._adapter.validate_python({
            "decision": "visualize",
            "viz_type": "parameter_explorer",
            "concept": "Bond duration",
            "variables": ["yield", "maturity"],
            "learning_goal": "Understand how duration changes with yield and maturity",
            "confidence": 0.85,
        })
        assert isinstance(result, VisualizationOpportunity)
        assert result.viz_type == "parameter_explorer"
        assert len(result.variables) == 2

    def test_rejects_single_variable(self) -> None:
        """Must have at least 2 variables."""
        with pytest.raises(ValidationError):
            VisualizationOpportunity(
                viz_type="parameter_explorer",
                concept="Test",
                variables=["only_one"],
                learning_goal="Test",
                confidence=0.8,
            )

    def test_rejects_invalid_confidence(self) -> None:
        """Confidence must be 0-1."""
        with pytest.raises(ValidationError):
            VisualizationOpportunity(
                viz_type="parameter_explorer",
                concept="Test",
                variables=["a", "b"],
                learning_goal="Test",
                confidence=1.5,
            )

    def test_all_viz_types_accepted(self) -> None:
        for vtype in ("parameter_explorer", "process_stepper", "comparison",
                       "system_dynamics", "data_explorer"):
            opp = VisualizationOpportunity(
                viz_type=vtype,
                concept="Test",
                variables=["a", "b"],
                learning_goal="Test",
                confidence=0.8,
            )
            assert opp.viz_type == vtype


class TestPromptBuilders:
    """Tests for triage and generation prompt construction."""

    def test_triage_prompt_includes_title(self) -> None:
        prompt = build_viz_triage_prompt(
            section_title="Bond Pricing",
            section_text="Some content about bonds.",
        )
        assert "Bond Pricing" in prompt

    def test_triage_prompt_includes_concepts(self) -> None:
        prompt = build_viz_triage_prompt(
            section_title="Test",
            section_text="Content.",
            concepts=["duration", "convexity"],
        )
        assert "duration" in prompt
        assert "convexity" in prompt

    def test_triage_prompt_truncates_long_text(self) -> None:
        long_text = "x" * 5000
        prompt = build_viz_triage_prompt("Title", long_text)
        assert "[...truncated...]" in prompt

    def test_generation_prompt_includes_opportunity(self) -> None:
        opp = VisualizationOpportunity(
            viz_type="parameter_explorer",
            concept="Bond Duration",
            variables=["yield", "maturity"],
            learning_goal="See how duration responds to yield changes",
            confidence=0.9,
        )
        prompt = build_viz_generation_prompt(opp, "Bonds 101", "Some text about bonds.")
        assert "Bond Duration" in prompt
        assert "parameter_explorer" in prompt
        assert "yield" in prompt
        assert "maturity" in prompt

    def test_generation_prompt_includes_few_shot(self) -> None:
        opp = VisualizationOpportunity(
            viz_type="comparison",
            concept="Test",
            variables=["a", "b"],
            learning_goal="Compare A and B",
            confidence=0.8,
        )
        prompt = build_viz_generation_prompt(opp, "Title", "Text")
        assert "p5.js" in prompt  # Few-shot uses p5.js
        assert "window.onerror" in prompt  # Error handling in few-shot
