"""Tests for deep reading and concept consolidation Pydantic models."""

import pytest
from pydantic import ValidationError

from src.transformation.analysis_types import (
    ChapterAnalysis,
    ConceptEdge,
    ConceptEntry,
    ConceptGraph,
    PrerequisiteLink,
    ResolvedConcept,
    SectionCharacterization,
)


class TestConceptEntry:
    """ConceptEntry model validation."""

    def test_minimal_valid(self) -> None:
        entry = ConceptEntry(
            name="Sharpe ratio",
            definition="Risk-adjusted return measure.",
            concept_type="formula",
            section_title="Risk Metrics",
        )
        assert entry.name == "Sharpe ratio"
        assert entry.importance == "supporting"  # default
        assert entry.key_terms == []
        assert entry.formulas == []

    def test_full_entry(self) -> None:
        entry = ConceptEntry(
            name="Present value",
            definition="The current worth of future cash flows.",
            concept_type="formula",
            section_title="Time Value of Money",
            key_terms=["discount rate", "cash flow"],
            formulas=["PV = FV / (1 + r)^n"],
            importance="core",
        )
        assert entry.importance == "core"
        assert len(entry.key_terms) == 2
        assert len(entry.formulas) == 1

    def test_rejects_invalid_concept_type(self) -> None:
        with pytest.raises(ValidationError):
            ConceptEntry(
                name="X",
                definition="Y",
                concept_type="invalid_type",  # type: ignore[arg-type]
                section_title="S",
            )

    def test_rejects_invalid_importance(self) -> None:
        with pytest.raises(ValidationError):
            ConceptEntry(
                name="X",
                definition="Y",
                concept_type="definition",
                section_title="S",
                importance="critical",  # type: ignore[arg-type]
            )


class TestPrerequisiteLink:
    """PrerequisiteLink model validation."""

    def test_minimal(self) -> None:
        link = PrerequisiteLink(
            source_concept="Bond pricing",
            target_concept="Present value",
            relationship="requires",
        )
        assert link.reasoning == ""

    def test_with_reasoning(self) -> None:
        link = PrerequisiteLink(
            source_concept="Sharpe ratio",
            target_concept="Standard deviation",
            relationship="builds_on",
            reasoning="Sharpe ratio uses std dev in the denominator",
        )
        assert link.reasoning != ""

    def test_rejects_invalid_relationship(self) -> None:
        with pytest.raises(ValidationError):
            PrerequisiteLink(
                source_concept="A",
                target_concept="B",
                relationship="depends_on",  # type: ignore[arg-type]
            )


class TestSectionCharacterization:
    """SectionCharacterization model validation."""

    def test_minimal(self) -> None:
        sc = SectionCharacterization(
            section_title="Intro",
            dominant_content_type="conceptual",
        )
        assert sc.has_formulas is False
        assert sc.difficulty_estimate == "intermediate"
        assert sc.summary == ""

    def test_full(self) -> None:
        sc = SectionCharacterization(
            section_title="Derivatives Pricing",
            dominant_content_type="procedural",
            has_formulas=True,
            has_procedures=True,
            has_comparisons=False,
            has_definitions=True,
            has_examples=True,
            difficulty_estimate="advanced",
            summary="Covers Black-Scholes model step by step.",
        )
        assert sc.has_formulas is True
        assert sc.difficulty_estimate == "advanced"

    def test_rejects_invalid_content_type(self) -> None:
        with pytest.raises(ValidationError):
            SectionCharacterization(
                section_title="X",
                dominant_content_type="narrative",  # type: ignore[arg-type]
            )


class TestChapterAnalysis:
    """ChapterAnalysis model validation."""

    def test_empty_analysis(self) -> None:
        """A minimal analysis with no concepts (e.g., fallback case)."""
        analysis = ChapterAnalysis(
            chapter_number=1,
            chapter_title="Introduction",
        )
        assert analysis.concepts == []
        assert analysis.prerequisites == []
        assert analysis.section_characterizations == []
        assert analysis.external_prerequisites == []

    def test_full_analysis(self) -> None:
        concept = ConceptEntry(
            name="Bond",
            definition="A fixed-income instrument.",
            concept_type="definition",
            section_title="Fixed Income Basics",
            importance="core",
        )
        prereq = PrerequisiteLink(
            source_concept="Bond pricing",
            target_concept="Present value",
            relationship="requires",
        )
        section_char = SectionCharacterization(
            section_title="Fixed Income Basics",
            dominant_content_type="conceptual",
            has_definitions=True,
        )
        analysis = ChapterAnalysis(
            chapter_number=1,
            chapter_title="Fixed Income",
            concepts=[concept],
            prerequisites=[prereq],
            section_characterizations=[section_char],
            logical_flow="Starts with definitions, builds to pricing.",
            core_learning_outcome="Understand bond pricing fundamentals.",
            external_prerequisites=["Time value of money"],
            difficulty_progression="introductory to intermediate",
        )
        assert len(analysis.concepts) == 1
        assert analysis.concepts[0].name == "Bond"
        assert len(analysis.external_prerequisites) == 1


class TestResolvedConcept:
    """ResolvedConcept model validation."""

    def test_minimal(self) -> None:
        rc = ResolvedConcept(
            canonical_name="Sharpe ratio",
            definition="Risk-adjusted return measure.",
            first_introduced_chapter=2,
        )
        assert rc.aliases == []
        assert rc.mentioned_in_chapters == []

    def test_with_aliases(self) -> None:
        rc = ResolvedConcept(
            canonical_name="Standard deviation",
            aliases=["std dev", "volatility"],
            definition="Measure of dispersion.",
            first_introduced_chapter=1,
            mentioned_in_chapters=[1, 3, 5],
        )
        assert len(rc.aliases) == 2
        assert 3 in rc.mentioned_in_chapters


class TestConceptEdge:
    """ConceptEdge model validation."""

    def test_valid(self) -> None:
        edge = ConceptEdge(
            source="Bond pricing",
            target="Present value",
            relationship="requires",
        )
        assert edge.source == "Bond pricing"

    def test_rejects_invalid_relationship(self) -> None:
        with pytest.raises(ValidationError):
            ConceptEdge(
                source="A",
                target="B",
                relationship="needs",  # type: ignore[arg-type]
            )


class TestConceptGraph:
    """ConceptGraph model validation."""

    def test_empty_graph(self) -> None:
        graph = ConceptGraph()
        assert graph.concepts == []
        assert graph.edges == []
        assert graph.topological_order == []
        assert graph.foundation_concepts == []
        assert graph.advanced_concepts == []

    def test_populated_graph(self) -> None:
        concepts = [
            ResolvedConcept(
                canonical_name="Present value",
                definition="Current worth of future cash flows.",
                first_introduced_chapter=1,
                mentioned_in_chapters=[1, 2],
            ),
            ResolvedConcept(
                canonical_name="Bond pricing",
                definition="Determining the fair value of a bond.",
                first_introduced_chapter=2,
                mentioned_in_chapters=[2],
            ),
        ]
        edges = [
            ConceptEdge(
                source="Bond pricing",
                target="Present value",
                relationship="requires",
            ),
        ]
        graph = ConceptGraph(
            concepts=concepts,
            edges=edges,
            topological_order=["Present value", "Bond pricing"],
            foundation_concepts=["Present value"],
            advanced_concepts=["Bond pricing"],
        )
        assert len(graph.concepts) == 2
        assert len(graph.edges) == 1
        assert graph.topological_order[0] == "Present value"
        assert "Present value" in graph.foundation_concepts
