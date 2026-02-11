"""Structured analysis models for the deep reading stage.

Pydantic models representing the output of the deep reader (Stage 1.25)
and the concept consolidator (Stage 1.3). These are the "structured notes"
that flow from deep reading through planning to content design.

Hierarchy:
- Deep reader produces ChapterAnalysis per chapter (concepts, prerequisites,
  section characterizations, logical flow).
- Concept consolidator deduplicates across chapters into a ConceptGraph
  with ResolvedConcepts, directed edges, and topological order.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ── Concept types ────────────────────────────────────────────────────────────

ConceptType = Literal[
    "definition",
    "formula",
    "process",
    "comparison",
    "principle",
    "example",
    "theorem",
    "heuristic",
]

ConceptImportance = Literal["core", "supporting", "peripheral"]

DominantContentType = Literal[
    "conceptual",
    "procedural",
    "comparative",
    "theoretical",
    "applied",
    "mixed",
]

PrerequisiteRelationship = Literal[
    "requires",
    "builds_on",
    "contrasts_with",
    "applies",
]


# ── Deep reader output models ────────────────────────────────────────────────


class ConceptEntry(BaseModel):
    """A concept found during deep reading of a chapter."""

    name: str = Field(description="Concept name (e.g., 'Sharpe ratio')")
    definition: str = Field(description="One-sentence definition")
    concept_type: ConceptType = Field(description="Classification of this concept")
    section_title: str = Field(description="Section where this concept appears")
    key_terms: list[str] = Field(
        default_factory=list,
        description="Related terms and vocabulary",
    )
    formulas: list[str] = Field(
        default_factory=list,
        description="Mathematical formulas associated with this concept",
    )
    importance: ConceptImportance = Field(
        default="supporting",
        description="How central this concept is to the chapter",
    )


class PrerequisiteLink(BaseModel):
    """A dependency relationship between two concepts."""

    source_concept: str = Field(description="The concept that depends on another")
    target_concept: str = Field(description="The concept being depended upon")
    relationship: PrerequisiteRelationship = Field(
        description="Type of dependency"
    )
    reasoning: str = Field(
        default="",
        description="Why this dependency exists",
    )


class SectionCharacterization(BaseModel):
    """Content analysis of a single section from deep reading."""

    section_title: str = Field(description="Title of the analyzed section")
    dominant_content_type: DominantContentType = Field(
        description="Primary content type in this section"
    )
    has_formulas: bool = Field(default=False)
    has_procedures: bool = Field(default=False)
    has_comparisons: bool = Field(default=False)
    has_definitions: bool = Field(default=False)
    has_examples: bool = Field(default=False)
    difficulty_estimate: Literal["introductory", "intermediate", "advanced"] = Field(
        default="intermediate",
        description="Estimated difficulty level",
    )
    summary: str = Field(
        default="",
        description="2-3 sentence summary of the section's content",
    )


class ChapterAnalysis(BaseModel):
    """Complete structured analysis of one chapter from the deep reader."""

    chapter_number: int = Field(description="Chapter number (1-indexed)")
    chapter_title: str = Field(description="Chapter title")
    concepts: list[ConceptEntry] = Field(
        default_factory=list,
        description="All concepts identified in this chapter",
    )
    prerequisites: list[PrerequisiteLink] = Field(
        default_factory=list,
        description="Dependencies between concepts",
    )
    section_characterizations: list[SectionCharacterization] = Field(
        default_factory=list,
        description="Content analysis per section",
    )
    logical_flow: str = Field(
        default="",
        description="How sections connect logically (narrative description)",
    )
    core_learning_outcome: str = Field(
        default="",
        description="The single most important thing a learner should take away",
    )
    external_prerequisites: list[str] = Field(
        default_factory=list,
        description="Concepts assumed from prior chapters",
    )
    difficulty_progression: str = Field(
        default="",
        description="How difficulty changes across the chapter",
    )


# ── Concept consolidator output models ───────────────────────────────────────


class ResolvedConcept(BaseModel):
    """A concept after cross-chapter deduplication."""

    canonical_name: str = Field(description="Primary name for this concept")
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternate names found across chapters",
    )
    definition: str = Field(description="Best available definition")
    first_introduced_chapter: int = Field(
        description="Chapter number where first introduced"
    )
    mentioned_in_chapters: list[int] = Field(
        default_factory=list,
        description="All chapter numbers mentioning this concept",
    )


class ConceptEdge(BaseModel):
    """A directed dependency in the concept graph."""

    source: str = Field(description="Concept that depends on another (canonical name)")
    target: str = Field(description="Concept being depended upon (canonical name)")
    relationship: PrerequisiteRelationship = Field(
        description="Type of dependency"
    )


class ConceptGraph(BaseModel):
    """The consolidated concept dependency graph across all chapters."""

    concepts: list[ResolvedConcept] = Field(
        default_factory=list,
        description="All unique concepts after deduplication",
    )
    edges: list[ConceptEdge] = Field(
        default_factory=list,
        description="Directed dependency edges",
    )
    topological_order: list[str] = Field(
        default_factory=list,
        description="Concepts in natural learning order (prerequisites first)",
    )
    foundation_concepts: list[str] = Field(
        default_factory=list,
        description="Concepts with no prerequisites (starting points)",
    )
    advanced_concepts: list[str] = Field(
        default_factory=list,
        description="Concepts with the most prerequisites (endpoints)",
    )
    canonical_map: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from any raw concept name (lowercased) to its canonical name",
    )

    def resolve(self, name: str) -> str:
        """Resolve a raw concept name to its canonical form."""
        return self.canonical_map.get(name.lower().strip(), name)
