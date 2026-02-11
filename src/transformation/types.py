"""Data types for LLM-generated interactive training content.

Pydantic models for structured LLM output (used by Instructor for
schema-constrained generation). Includes all interactive element types:
slides, quizzes, flashcards, fill-in-the-blank, matching exercises,
self-explanation, and interactive essays.

Each element type has a deterministic Bloom's Taxonomy cognitive level
(see ELEMENT_BLOOM_MAP). The LLM does not choose bloom levels — they
are assigned automatically based on element type.

TrainingElement is a Pydantic v2 discriminated union — each variant is
a concrete class, and Pydantic auto-dispatches on the element_type field.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Discriminator, Field


# ── Bloom's Taxonomy cognitive levels ────────────────────────────────────────
# Used to tag each training element with its target cognitive level.
# Displayed as a badge in the HTML output for pedagogical transparency.
BloomLevel = Literal[
    "remember",     # Recall facts, terms, definitions
    "understand",   # Explain ideas, paraphrase, summarize
    "apply",        # Use knowledge in new situations
    "analyze",      # Break down, compare, distinguish
    "evaluate",     # Judge, justify, defend a position
    "create",       # Synthesize, design, construct
]

# ── Canonical element → Bloom mapping (source of truth) ─────────────────────
# Element type determines bloom level deterministically. The LLM does NOT
# choose bloom levels — this mapping overrides whatever it outputs.
ELEMENT_BLOOM_MAP: dict[str, BloomLevel] = {
    "flashcard": "remember",
    "slide": "understand",
    "mermaid": "understand",
    "quiz": "apply",
    "matching": "apply",
    "fill_in_the_blank": "apply",
    "concept_map": "analyze",
    "self_explain": "evaluate",
    "interactive_essay": "evaluate",
}

# Angles for reinforcement target selection (Phase 1 of two-phase generation).
ReinforcementAngle = Literal[
    "mechanism",      # How does X work?
    "connection",     # How does X relate to Y?
    "application",    # When would you use X?
    "edge_case",      # When does X fail?
    "contrast",       # How is X different from Y?
    "consequence",    # What follows from X?
]

# Element types suitable for assessment (used in reinforcement target suggestions).
AssessmentElementType = Literal[
    "quiz", "flashcard", "fill_in_the_blank", "matching", "self_explain",
]


# ── Reinforcement targets (Phase 1 output) ──────────────────────────────────


class ReinforcementTarget(BaseModel):
    """A specific insight identified as worth testing in a section."""

    concept_name: str = Field(description="The concept this target relates to")
    target_insight: str = Field(
        description="The specific insight to reinforce — NOT a definition, "
        "but a mechanism, connection, or application"
    )
    angle: ReinforcementAngle = Field(description="What angle to test from")
    bloom_level: BloomLevel = Field(
        description="The cognitive level this target naturally maps to"
    )
    suggested_element_type: AssessmentElementType = Field(
        description="Which element type best tests this insight"
    )


class ReinforcementTargetSet(BaseModel):
    """Output of Phase 1: what's worth testing in this section."""

    targets: list[ReinforcementTarget] = Field(
        min_length=3,
        description="Ordered list of insights worth reinforcing, from foundational to advanced",
    )


# ── Element types ────────────────────────────────────────────────────────────


class Slide(BaseModel):
    """An explanatory slide presenting information to the learner."""

    title: str = Field(description="Slide heading displayed prominently")
    content: str = Field(description="Body text, may contain markdown and LaTeX math ($...$ or $$...$$)")
    speaker_notes: str = Field(default="", description="Additional context not shown to the learner")
    image_path: str | None = Field(default=None, description="Path to an associated image, or null")
    source_pages: str = Field(default="", description="Source attribution, e.g. 'pp. 42-43'")


class QuizQuestion(BaseModel):
    """A single multiple-choice question."""

    question: str = Field(description="The question text")
    options: list[str] = Field(description="Answer choices, at least 2", min_length=2)
    correct_index: int = Field(description="0-based index of the correct answer")
    explanation: str = Field(default="", description="Explanation shown after answering")
    # Graduated hints (3 levels for multi-attempt feedback)
    hint_metacognitive: str = Field(
        default="",
        description="Level 1 hint: metacognitive prompt asking which concept applies",
    )
    hint_strategic: str = Field(
        default="",
        description="Level 2 hint: strategic clue narrowing the options",
    )
    hint_eliminate_index: int = Field(
        default=-1,
        description="Index of a distractor to visually eliminate on hint level 2 (-1 = none)",
    )


class Quiz(BaseModel):
    """A set of related quiz questions testing a section's content."""

    title: str = Field(description="Quiz heading")
    questions: list[QuizQuestion] = Field(description="Ordered sequence of questions", min_length=1)


class Flashcard(BaseModel):
    """A two-sided flashcard for key concept review."""

    front: str = Field(description="The prompt or question side")
    back: str = Field(description="The answer or definition side")


class FillInTheBlank(BaseModel):
    """A fill-in-the-blank exercise testing recall in context."""

    statement: str = Field(description="The statement with _____ marking each blank")
    answers: list[str] = Field(description="Correct answers for each blank, in order", min_length=1)
    hint: str = Field(default="", description="Optional hint shown to the learner")
    hint_first_letter: bool = Field(
        default=True,
        description="On second failure, reveal first letter of each answer",
    )


class MatchingExercise(BaseModel):
    """A matching exercise where learners pair related items."""

    title: str = Field(description="Exercise heading")
    left_items: list[str] = Field(description="Items on the left (e.g., terms)", min_length=2)
    right_items: list[str] = Field(description="Items on the right in correct order matching left_items", min_length=2)
    pair_explanations: list[str] = Field(
        default_factory=list,
        description="Explanation for each correct pair (same order as left/right items)",
    )


class MermaidDiagram(BaseModel):
    """A diagram rendered via Mermaid.js syntax."""

    title: str = Field(description="Diagram title displayed above")
    diagram_code: str = Field(
        description="Valid Mermaid syntax (flowchart, sequence, state, etc.)"
    )
    caption: str = Field(default="", description="Explanatory caption below the diagram")
    diagram_type: Literal[
        "flowchart", "sequence", "state", "class", "er", "gantt", "pie", "mindmap"
    ] = Field(default="flowchart", description="Mermaid diagram type")


class ConceptMapNode(BaseModel):
    """A single node in a concept map."""

    id: str = Field(description="Unique node identifier")
    label: str = Field(description="Display text for this concept")


class ConceptMapEdge(BaseModel):
    """A labeled relationship between two concept map nodes."""

    source: str = Field(description="Source node id")
    target: str = Field(description="Target node id")
    label: str = Field(
        description="Relationship description (e.g., 'is a type of', 'depends on')"
    )


class ConceptMap(BaseModel):
    """An interactive concept map showing relationships between concepts."""

    title: str = Field(description="Map title")
    nodes: list[ConceptMapNode] = Field(
        min_length=3, description="Concept nodes (3-12 recommended)"
    )
    edges: list[ConceptMapEdge] = Field(
        min_length=2, description="Labeled relationships between nodes"
    )
    blank_edge_indices: list[int] = Field(
        default_factory=list,
        description="Indices into edges list where label should be hidden for learner to fill in",
    )


class SelfExplain(BaseModel):
    """A self-explanation exercise where the learner explains a concept in their own words."""

    prompt: str = Field(
        description="The question prompting the learner to explain "
        "(e.g., 'Explain in your own words why diversification reduces portfolio risk')"
    )
    key_points: list[str] = Field(
        description="Key points that a good explanation should cover (shown as checklist after submission)",
        min_length=2,
    )
    example_response: str = Field(
        description="A model response shown after the learner self-assesses"
    )
    minimum_words: int = Field(
        default=50,
        description="Minimum word count before the learner can submit",
    )
    source_pages: str = Field(default="", description="Source attribution")


class InteractiveEssay(BaseModel):
    """An interactive essay checkpoint combining self-explanation with an LLM evaluator rubric."""

    title: str = Field(description="Interactive essay title (e.g., 'Chapter 3 Checkpoint')")
    concepts_tested: list[str] = Field(
        description="List of key concepts this essay tests",
        min_length=1,
    )
    prompts: list[SelfExplain] = Field(
        description="Ordered explanation prompts for this essay",
        min_length=1,
    )
    passing_threshold: float = Field(
        default=0.7,
        description="Fraction of key points the learner must self-check to pass (0.0-1.0)",
    )
    tutor_system_prompt: str = Field(
        default="",
        description="System prompt for the LLM tutor including rubric, Socratic instructions, pass/fail criteria",
    )


# ── Training element (discriminated union) ───────────────────────────────────
# Each variant is a concrete class with element_type as a Literal default.
# Pydantic v2's Discriminator auto-selects the right class from JSON.


class SlideElement(BaseModel):
    """A slide training element."""

    element_type: Literal["slide"] = "slide"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    slide: Slide


class QuizElement(BaseModel):
    """A quiz training element."""

    element_type: Literal["quiz"] = "quiz"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    quiz: Quiz


class FlashcardElement(BaseModel):
    """A flashcard training element."""

    element_type: Literal["flashcard"] = "flashcard"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    flashcard: Flashcard


class FillInBlankElement(BaseModel):
    """A fill-in-the-blank training element."""

    element_type: Literal["fill_in_the_blank"] = "fill_in_the_blank"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    fill_in_the_blank: FillInTheBlank


class MatchingElement(BaseModel):
    """A matching exercise training element."""

    element_type: Literal["matching"] = "matching"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    matching: MatchingExercise


class MermaidElement(BaseModel):
    """A Mermaid diagram training element."""

    element_type: Literal["mermaid"] = "mermaid"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    mermaid: MermaidDiagram


class ConceptMapElement(BaseModel):
    """A concept map training element."""

    element_type: Literal["concept_map"] = "concept_map"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    concept_map: ConceptMap


class SelfExplainElement(BaseModel):
    """A self-explanation training element."""

    element_type: Literal["self_explain"] = "self_explain"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    self_explain: SelfExplain


class InteractiveEssayElement(BaseModel):
    """An interactive essay training element."""

    element_type: Literal["interactive_essay"] = "interactive_essay"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    interactive_essay: InteractiveEssay


TrainingElement = Annotated[
    Union[
        SlideElement, QuizElement, FlashcardElement, FillInBlankElement,
        MatchingElement, MermaidElement, ConceptMapElement,
        SelfExplainElement, InteractiveEssayElement,
    ],
    Discriminator("element_type"),
]


# ── Curriculum blueprint models ──────────────────────────────────────────────
# Output of the curriculum planner (Stage 1.5). Guides the content designer
# by assigning templates, learning objectives, and Bloom's targets per section.


class SectionBlueprint(BaseModel):
    """Blueprint for a single section within a module."""

    title: str = Field(description="Section heading")
    source_section_title: str = Field(
        default="",
        description="Maps back to extracted Section.title (empty if synthetic)",
    )
    source_book_index: int | None = Field(
        default=None,
        description="Index into the books list (for multi-doc)",
    )
    learning_objectives: list[str] = Field(
        default_factory=list,
        description="Measurable learning objectives for this section",
    )
    template: str = Field(
        default="analogy_first",
        description="Content template to use (e.g., 'analogy_first', 'worked_example')",
    )
    bloom_target: BloomLevel = Field(
        default="understand",
        description="Primary Bloom's level for this section",
    )
    prerequisites: list[str] = Field(
        default_factory=list,
        description="Titles of sections that should be covered first",
    )
    rationale: str = Field(
        default="",
        description="Why this section exists and why this template was chosen",
    )


class ModuleBlueprint(BaseModel):
    """Blueprint for a module (maps to one chapter)."""

    title: str = Field(description="Module title")
    source_chapter_number: int | None = Field(
        default=None,
        description="Maps back to Chapter.chapter_number (None if synthetic)",
    )
    source_book_index: int | None = Field(
        default=None,
        description="Index into the books list (for multi-doc)",
    )
    summary: str = Field(default="", description="2-3 sentence module description")
    sections: list[SectionBlueprint] = Field(
        default_factory=list,
        description="Ordered sections within this module",
    )


class CurriculumBlueprint(BaseModel):
    """Complete curriculum plan output by the planner."""

    course_title: str = Field(description="Course title")
    course_summary: str = Field(default="", description="2-3 sentence overview")
    learner_journey: str = Field(
        default="",
        description="Narrative learning path: 'Module A → Module B → ...'",
    )
    modules: list[ModuleBlueprint] = Field(
        default_factory=list,
        description="Ordered modules making up the curriculum",
    )


# ── Training section and module ──────────────────────────────────────────────


class TrainingSection(BaseModel):
    """A group of training elements corresponding to one book section."""

    title: str
    source_pages: str = Field(default="", description="Page range, e.g. 'pp. 42-48'")
    elements: list[TrainingElement]
    verification_notes: list[str] = Field(
        default_factory=list,
        description="Warnings from post-generation source verification (potential hallucinations)",
    )
    reinforcement_targets: list[dict] = Field(
        default_factory=list,
        description="Phase 1 reinforcement targets used during generation",
    )
    learning_objectives: list[str] = Field(
        default_factory=list,
        description="Section learning objectives from curriculum planner",
    )


class TrainingModule(BaseModel):
    """A complete training module corresponding to one book chapter."""

    chapter_number: int
    title: str
    sections: list[TrainingSection]

    @property
    def all_elements(self) -> list[TrainingElement]:
        """Flat list of all elements across all sections."""
        return [e for s in self.sections for e in s.elements]
