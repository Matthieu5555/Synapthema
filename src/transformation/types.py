"""Data types for LLM-generated interactive training content.

Pydantic models for structured LLM output (used by Instructor for
schema-constrained generation). Includes all interactive element types:
section intros, slides, quizzes, flashcards, fill-in-the-blank, matching
exercises, and interactive essays.

Each element type has a deterministic Bloom's Taxonomy cognitive level
(see ELEMENT_BLOOM_MAP). The LLM does not choose bloom levels — they
are assigned automatically based on element type.

TrainingElement is a Pydantic v2 discriminated union — each variant is
a concrete class, and Pydantic auto-dispatches on the element_type field.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Annotated, ClassVar, Literal, TypedDict, Union

from pydantic import BaseModel, Discriminator, Field, model_validator

logger = logging.getLogger(__name__)


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

# ── Exercise difficulty (LLM-chosen, independent of Bloom level) ─────────────
# Unlike bloom_level (deterministic per element type), difficulty is chosen by
# the LLM to create a progression within each section: easy → medium → hard.
ExerciseDifficulty = Literal["easy", "medium", "hard"]
DIFFICULTY_ORDER: dict[str, int] = {"easy": 1, "medium": 2, "hard": 3}

# ── Exercise element types ───────────────────────────────────────────────────
# Element types that count as practice exercises (used in validators).
EXERCISE_ELEMENT_TYPES: frozenset[str] = frozenset({
    "quiz", "fill_in_the_blank", "matching", "ordering",
    "categorization", "analogy", "worked_example", "far_transfer",
})

# Element types that are passive/reveal-based (click to reveal, not interactive).
# These are placed after exercises alongside flashcards, not counted as exercises.
REVEAL_ELEMENT_TYPES: frozenset[str] = frozenset({"error_detection"})

# ── Canonical element → Bloom mapping (source of truth) ─────────────────────
# Element type determines bloom level deterministically. The LLM does NOT
# choose bloom levels — this mapping overrides whatever it outputs.
ELEMENT_BLOOM_MAP: dict[str, BloomLevel] = {
    "section_intro": "understand",
    "flashcard": "remember",
    "slide": "understand",
    "mermaid": "understand",
    "quiz": "apply",
    "matching": "apply",
    "ordering": "apply",
    "fill_in_the_blank": "analyze",
    "categorization": "analyze",
    "analogy": "analyze",
    "concept_map": "apply",
    "error_detection": "evaluate",
    "interactive_essay": "evaluate",
    "worked_example": "apply",
    "interactive_visualization": "apply",
    "far_transfer": "analyze",
}

# ── Element role classification ──────────────────────────────────────────────
# Used by the interleave-preserving validator to anchor bookend elements
# while preserving the LLM's teach-practice cycle ordering for core elements.
ELEMENT_ROLE: dict[str, str] = {
    "section_intro": "intro",       # Always first
    "slide": "teach",               # Core: LLM order preserved
    "mermaid": "teach",             # Core: LLM order preserved
    "quiz": "practice",             # Core: LLM order preserved
    "ordering": "practice",         # Core: LLM order preserved
    "matching": "practice",         # Core: LLM order preserved
    "fill_in_the_blank": "practice",  # Core: LLM order preserved
    "categorization": "practice",   # Core: LLM order preserved
    "analogy": "practice",          # Core: LLM order preserved
    "error_detection": "reveal",    # Passive reveal — anchored after exercises
    "concept_map": "synthesis",     # Anchored near end
    "worked_example": "teach",      # Core: LLM order preserved
    "interactive_visualization": "teach",  # Core: LLM order preserved
    "far_transfer": "practice",     # Core: LLM order preserved
    "flashcard": "reinforce",       # Anchored after practice
    "interactive_essay": "assess",  # Always last
}

# Angles for reinforcement target selection (Phase 1 of two-phase generation).
ReinforcementAngle = Literal[
    "mechanism",      # How does X work?
    "connection",     # How does X relate to Y?
    "application",    # When would you use X?
    "edge_case",      # When does X fail?
    "contrast",       # How is X different from Y?
    "consequence",    # What follows from X?
    "far_transfer",   # Does the principle hold in a distant domain?
]

# Element types suitable for assessment (used in reinforcement target suggestions).
AssessmentElementType = Literal[
    "quiz", "flashcard", "fill_in_the_blank", "matching", "ordering",
    "categorization", "error_detection", "analogy", "interactive_essay",
    "far_transfer",
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


class SectionIntro(BaseModel):
    """A brief motivational introduction to a section, derived from learning objectives."""

    title: str = Field(description="Section intro heading")
    content: str = Field(
        description="2-3 sentence prose motivating the section and previewing what the learner will gain"
    )
    source_pages: str = Field(default="", description="Source attribution")


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

    @model_validator(mode="after")
    def _validate_indices(self) -> QuizQuestion:
        n = len(self.options)
        if self.correct_index < 0 or self.correct_index >= n:
            raise ValueError(
                f"correct_index {self.correct_index} is out of range for "
                f"{n} options (valid range: 0 to {n - 1}). "
                f"Set correct_index to the 0-based index of the correct answer."
            )
        # hint_eliminate_index is a UI hint — soft fix is acceptable
        if self.hint_eliminate_index >= 0:
            if self.hint_eliminate_index >= n:
                self.hint_eliminate_index = -1
            elif self.hint_eliminate_index == self.correct_index:
                self.hint_eliminate_index = -1
        return self


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

    statement: str = Field(description="The statement with [BLANK] marking each blank")
    answers: list[str] = Field(description="Correct answers for each blank, in order", min_length=1)
    hint: str = Field(default="", description="Optional hint shown to the learner")
    hint_first_letter: bool = Field(
        default=True,
        description="On second failure, reveal first letter of each answer",
    )

    @model_validator(mode="after")
    def _validate_blank_count(self) -> FillInTheBlank:
        blank_count = self.statement.count("[BLANK]")
        if blank_count > 0 and blank_count != len(self.answers):
            raise ValueError(
                f"Number of blanks ({blank_count}) in statement does not match "
                f"number of answers ({len(self.answers)}). "
                f"Provide exactly {blank_count} answers to match the [BLANK] "
                f"markers in the statement."
            )
        return self


class MatchingExercise(BaseModel):
    """A matching exercise where learners pair related items."""

    title: str = Field(description="Exercise heading")
    left_items: list[str] = Field(description="Items on the left (e.g., terms)", min_length=2)
    right_items: list[str] = Field(description="Items on the right in correct order matching left_items", min_length=2)
    pair_explanations: list[str] = Field(
        default_factory=list,
        description="Explanation for each correct pair (same order as left/right items)",
    )

    @model_validator(mode="after")
    def _validate_matching_lengths(self) -> MatchingExercise:
        if len(self.left_items) != len(self.right_items):
            raise ValueError(
                f"left_items has {len(self.left_items)} items but right_items "
                f"has {len(self.right_items)} items. They must have the same "
                f"length so each left item maps to exactly one right item."
            )
        if self.pair_explanations and len(self.pair_explanations) != len(self.left_items):
            raise ValueError(
                f"pair_explanations has {len(self.pair_explanations)} entries "
                f"but there are {len(self.left_items)} pairs. Provide exactly "
                f"one explanation per pair, or omit pair_explanations entirely."
            )
        return self


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

    @model_validator(mode="after")
    def _validate_blank_edge_indices(self) -> ConceptMap:
        n = len(self.edges)
        invalid = [i for i in self.blank_edge_indices if i < 0 or i >= n]
        if invalid:
            raise ValueError(
                f"blank_edge_indices contains invalid indices {invalid}. "
                f"Valid range is 0 to {n - 1} (there are {n} edges). "
                f"Only include indices that reference existing edges."
            )
        return self


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
        ge=0.0,
        le=1.0,
    )
    tutor_system_prompt: str = Field(
        default="",
        description="System prompt for the LLM tutor including rubric, Socratic instructions, pass/fail criteria",
    )


# ── Ordering exercise ─────────────────────────────────────────────────────────


class OrderingExercise(BaseModel):
    """An ordering exercise where learners arrange items in the correct sequence."""

    title: str = Field(description="Exercise heading")
    instruction: str = Field(
        description="What the learner should do (e.g., 'Arrange these steps in the correct order')"
    )
    items: list[str] = Field(
        description="Items in the CORRECT order (shuffled at render time)",
        min_length=3,
    )
    explanation: str = Field(default="", description="Explanation shown after completion")
    hint: str = Field(default="", description="Hint shown after first wrong attempt")


# ── Categorization exercise ──────────────────────────────────────────────────


class CategoryBucket(BaseModel):
    """A named category with its correct items."""

    name: str = Field(description="Category name")
    items: list[str] = Field(description="Items that belong in this category", min_length=1)


class CategorizationExercise(BaseModel):
    """A categorization exercise where learners sort items into named categories."""

    title: str = Field(description="Exercise heading")
    instruction: str = Field(
        description="What the learner should do (e.g., 'Sort these items into the correct categories')"
    )
    categories: list[CategoryBucket] = Field(
        description="2-4 categories with their correct item assignments",
        min_length=2,
        max_length=4,
    )
    explanation: str = Field(default="", description="Explanation shown after completion")
    hint: str = Field(default="", description="Hint shown after first wrong attempt")


# ── Error detection exercise ─────────────────────────────────────────────────


class ErrorItem(BaseModel):
    """A statement containing an error for the learner to identify."""

    statement: str = Field(description="Statement containing an error")
    error_explanation: str = Field(description="Why the statement is wrong")
    corrected_statement: str = Field(description="The corrected version of the statement")


class ErrorDetectionExercise(BaseModel):
    """An error detection exercise where learners identify and correct mistakes."""

    title: str = Field(description="Exercise heading")
    instruction: str = Field(
        description="What the learner should do (e.g., 'Find and explain the error in each statement')"
    )
    items: list[ErrorItem] = Field(
        description="Statements with errors for the learner to find",
        min_length=1,
    )
    context: str = Field(default="", description="Background context for the statements")


# ── Analogy completion exercise ──────────────────────────────────────────────


class AnalogyItem(BaseModel):
    """A single analogy question with multiple-choice options."""

    stem: str = Field(description="The analogy stem (e.g., 'A is to B as C is to ___')")
    answer: str = Field(description="The correct completion")
    distractors: list[str] = Field(
        description="Wrong options (2-3 plausible alternatives)",
        min_length=2,
        max_length=3,
    )
    explanation: str = Field(default="", description="Why this analogy works")


class AnalogyExercise(BaseModel):
    """An analogy completion exercise testing relational reasoning."""

    title: str = Field(description="Exercise heading")
    items: list[AnalogyItem] = Field(
        description="Analogy questions",
        min_length=1,
    )


# ── Far transfer exercise ──────────────────────────────────────────────────


class FarTransferExercise(BaseModel):
    """A far transfer exercise testing cross-domain application of a principle.

    Presents a principle learned in one domain and asks the learner to
    identify or apply the same structural pattern in a maximally distant domain.
    """

    source_principle: str = Field(
        description="The underlying transferable principle (e.g., 'Negative feedback loops maintain equilibrium')"
    )
    source_domain: str = Field(
        description="Domain where the principle was taught (e.g., 'Biology — thermoregulation')"
    )
    transfer_domain: str = Field(
        description="A distant domain for transfer (e.g., 'Economics — monetary policy'). "
        "Must NOT be adjacent to source_domain."
    )
    scenario: str = Field(
        description="A self-contained scenario in the transfer domain. The learner should NOT "
        "need expertise in this domain to understand the scenario."
    )
    question: str = Field(description="What the learner must identify or apply")
    options: list[str] = Field(
        description="Answer choices (3-5)", min_length=3, max_length=5
    )
    correct_index: int = Field(description="0-based index of the correct answer")
    distractors_reasoning: list[str] = Field(
        description="Why each wrong answer is plausible (same length as options minus 1)",
    )
    bridge_insight: str = Field(
        description="The structural mapping connecting both domains — names the shared abstract pattern"
    )
    explanation: str = Field(
        description="Full explanation shown after the learner answers"
    )

    @model_validator(mode="after")
    def _validate_indices(self) -> "FarTransferExercise":
        n = len(self.options)
        if self.correct_index < 0 or self.correct_index >= n:
            raise ValueError(
                f"correct_index {self.correct_index} is out of range for "
                f"{n} options (valid range: 0 to {n - 1}). "
                f"Set correct_index to the 0-based index of the correct answer."
            )
        return self


# ── Worked example (Brilliant-style) ─────────────────────────────────────────


class WorkedExampleChallengeOption(BaseModel):
    """A single option in the try-it-first challenge."""

    text: str = Field(description="Option text")


class WorkedExampleStep(BaseModel):
    """A single step in the worked solution."""

    title: str = Field(description="Short step label (e.g., 'Identify the variables')")
    content: str = Field(
        description="What this step does: the mathematical or logical operation"
    )
    why: str = Field(
        description="WHY this step is necessary: the intuition behind it, "
        "not just a restatement of what was done"
    )


class WorkedExample(BaseModel):
    """A Brilliant-style worked example with try-it-first challenge and step reveal."""

    title: str = Field(description="Worked example heading")
    problem_statement: str = Field(
        description="The problem to be solved, presented clearly with all given information"
    )
    challenge_question: str = Field(
        description="Try-it-first question posed before the solution is revealed"
    )
    challenge_options: list[WorkedExampleChallengeOption] = Field(
        description="Multiple-choice options for the challenge (3-5 options)",
        min_length=3,
        max_length=5,
    )
    challenge_correct_index: int = Field(
        description="0-based index of the correct challenge option"
    )
    challenge_explanation: str = Field(
        default="",
        description="Brief explanation shown after the challenge attempt",
    )
    steps: list[WorkedExampleStep] = Field(
        description="Ordered solution steps (3-7 recommended)",
        min_length=2,
        max_length=8,
    )
    final_answer: str = Field(
        description="The final answer or conclusion, stated clearly"
    )
    source_pages: str = Field(default="", description="Source attribution")

    @model_validator(mode="after")
    def validate_challenge_index(self) -> "WorkedExample":
        if self.challenge_correct_index < 0 or self.challenge_correct_index >= len(self.challenge_options):
            raise ValueError(
                f"challenge_correct_index {self.challenge_correct_index} out of range "
                f"for {len(self.challenge_options)} options"
            )
        return self


# ── Training element (discriminated union) ───────────────────────────────────
# Each variant is a concrete class with element_type as a Literal default.
# Pydantic v2's Discriminator auto-selects the right class from JSON.


class SectionIntroElement(BaseModel):
    """A section introduction training element."""

    element_type: Literal["section_intro"] = "section_intro"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    section_intro: SectionIntro


class SlideElement(BaseModel):
    """A slide training element."""

    element_type: Literal["slide"] = "slide"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    slide: Slide


class QuizElement(BaseModel):
    """A quiz training element."""

    element_type: Literal["quiz"] = "quiz"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    difficulty: ExerciseDifficulty = Field(default="medium", description="Exercise difficulty: easy, medium, or hard")
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
    difficulty: ExerciseDifficulty = Field(default="medium", description="Exercise difficulty: easy, medium, or hard")
    fill_in_the_blank: FillInTheBlank


class MatchingElement(BaseModel):
    """A matching exercise training element."""

    element_type: Literal["matching"] = "matching"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    difficulty: ExerciseDifficulty = Field(default="medium", description="Exercise difficulty: easy, medium, or hard")
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


class OrderingElement(BaseModel):
    """An ordering exercise training element."""

    element_type: Literal["ordering"] = "ordering"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    difficulty: ExerciseDifficulty = Field(default="medium", description="Exercise difficulty: easy, medium, or hard")
    ordering: OrderingExercise


class CategorizationElement(BaseModel):
    """A categorization exercise training element."""

    element_type: Literal["categorization"] = "categorization"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    difficulty: ExerciseDifficulty = Field(default="medium", description="Exercise difficulty: easy, medium, or hard")
    categorization: CategorizationExercise


class ErrorDetectionElement(BaseModel):
    """An error detection exercise training element."""

    element_type: Literal["error_detection"] = "error_detection"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    difficulty: ExerciseDifficulty = Field(default="medium", description="Exercise difficulty: easy, medium, or hard")
    error_detection: ErrorDetectionExercise


class AnalogyElement(BaseModel):
    """An analogy completion exercise training element."""

    element_type: Literal["analogy"] = "analogy"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    difficulty: ExerciseDifficulty = Field(default="medium", description="Exercise difficulty: easy, medium, or hard")
    analogy: AnalogyExercise


class WorkedExampleElement(BaseModel):
    """A Brilliant-style worked example training element."""

    element_type: Literal["worked_example"] = "worked_example"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    difficulty: ExerciseDifficulty = Field(default="medium", description="Exercise difficulty: easy, medium, or hard")
    worked_example: WorkedExample


class FarTransferElement(BaseModel):
    """A far transfer exercise training element."""

    element_type: Literal["far_transfer"] = "far_transfer"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    difficulty: ExerciseDifficulty = Field(default="medium", description="Exercise difficulty: easy, medium, or hard")
    far_transfer: FarTransferExercise


class InteractiveEssayElement(BaseModel):
    """An interactive essay training element."""

    element_type: Literal["interactive_essay"] = "interactive_essay"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    interactive_essay: InteractiveEssay


class InteractiveVisualization(BaseModel):
    """LLM-generated interactive HTML visualization (explorable explanation).

    Contains a complete self-contained HTML document with embedded JS that
    renders an interactive parameter explorer, process stepper, or similar
    visualization. Displayed in a sandboxed iframe.
    """

    title: str = Field(description="Short descriptive title for the visualization")
    description: str = Field(
        description="One-sentence description of what the learner explores",
    )
    html_code: str = Field(
        description="Complete self-contained HTML with embedded JS (rendered in sandboxed iframe)",
    )
    viz_type: str = Field(
        description="Visualization category: parameter_explorer, process_stepper, comparison, system_dynamics, data_explorer",
    )
    fallback_text: str = Field(
        default="",
        description="Plain-text description shown if visualization cannot render",
    )


class InteractiveVisualizationElement(BaseModel):
    """An LLM-generated interactive visualization element."""

    element_type: Literal["interactive_visualization"] = "interactive_visualization"
    bloom_level: BloomLevel = Field(
        default="apply",
        description="Bloom's Taxonomy cognitive level",
    )
    interactive_visualization: InteractiveVisualization


TrainingElement = Annotated[
    Union[
        SectionIntroElement, SlideElement, QuizElement, FlashcardElement,
        FillInBlankElement, MatchingElement, OrderingElement, MermaidElement,
        ConceptMapElement, CategorizationElement, ErrorDetectionElement,
        AnalogyElement, WorkedExampleElement, FarTransferElement,
        InteractiveEssayElement, InteractiveVisualizationElement,
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
    focus_concepts: list[str] = Field(
        default_factory=list,
        description="Concept names this learning unit focuses on. When non-empty, "
        "the content designer generates elements ONLY for these concepts. "
        "When empty, covers all concepts in the source section (backward compatible).",
    )


class ModuleBlueprint(BaseModel):
    """Blueprint for a module (maps to one or more source chapters)."""

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
    additional_source_chapters: list[dict] = Field(
        default_factory=list,
        description="Extra source chapters merged into this module (multi-doc). "
        "Each dict has 'book_index' (int) and 'chapter_number' (int).",
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
    source_section_title: str = Field(
        default="",
        description="Original extracted section title (for concept lookup). "
        "When a source section is split into concept-focused units, the "
        "title changes but source_section_title stays the same.",
    )
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


# ── Course capabilities declaration ──────────────────────────────────────────


class CourseCapabilities(TypedDict):
    """Declares what features are available in a generated course.

    Computed by the pipeline after all stages complete, before rendering.
    The renderer uses this to explicitly enable/disable UI sections instead
    of silently degrading.
    """

    has_concept_graph: bool
    has_mastery_tracking: bool
    has_chapter_review: bool
    has_mixed_review: bool
    has_course_metadata: bool
    has_learning_objectives: bool
    chapter_count: int
    element_types_present: list[str]


# ── LLM response schema ──────────────────────────────────────────────────────


class SectionResponse(BaseModel):
    """Pydantic model for the LLM's structured response.

    Instructor validates the LLM output against this schema automatically.
    If validation fails, Instructor re-prompts with the error for self-correction.

    Validation thresholds can be overridden via class attributes before
    construction (e.g. by a ContentProfile) without subclassing.
    """

    # ── Configurable validation thresholds (class-level) ─────────────────
    # These ClassVars are set by content_designer.py from the active profile
    # before each LLM call.  They are NOT Pydantic fields — ClassVar tells
    # Pydantic to leave them alone.
    _min_exercises: ClassVar[int] = 4
    _min_exercise_types: ClassVar[int] = 3
    _max_quizzes: ClassVar[int] = 1
    _max_interactive_essays: ClassVar[int] = 2

    elements: list[TrainingElement] = Field(
        description="Ordered sequence of interactive training elements"
    )

    @model_validator(mode="after")
    def fix_bloom_levels(self) -> "SectionResponse":
        """Override LLM-chosen bloom levels with the canonical mapping."""
        for element in self.elements:
            correct = ELEMENT_BLOOM_MAP.get(element.element_type)
            if correct and element.bloom_level != correct:
                object.__setattr__(element, "bloom_level", correct)
        return self

    @model_validator(mode="after")
    def enforce_element_distribution(self) -> "SectionResponse":
        """Enforce 1-slide-then-drill section structure.

        Architecture: each section = 1 concept = 1 slide + 4-5 exercises + 2-3 flashcards.
        Hard errors catch truly bad outputs (triggers Instructor retry).
        Warnings flag aspirational targets that are not worth a retry.
        """
        counts = Counter(e.element_type for e in self.elements)

        teach_types = {"slide", "mermaid", "worked_example"}
        # Exercise types (excludes flashcard and error_detection — both are passive)
        exercise_types = {
            "quiz", "fill_in_the_blank", "matching", "ordering",
            "categorization", "analogy", "far_transfer",
        }

        teach_count = sum(counts.get(t, 0) for t in teach_types)
        exercise_count = sum(counts.get(t, 0) for t in exercise_types)
        flashcard_count = counts.get("flashcard", 0)

        # Count distinct exercise types used
        exercise_types_used = {
            t for t in exercise_types if counts.get(t, 0) > 0
        }

        # ── Hard constraints (trigger Instructor retry) ──────────────────

        if teach_count < 1:
            raise ValueError("Section must contain at least 1 teaching element (slide/mermaid)")

        # Max 1 slide per section. Auto-fix by merging extra slides into the
        # first one (content the LLM split across slides is still valuable).
        # worked_examples are structurally different, so extras are dropped.
        slide_count = counts.get("slide", 0) + counts.get("worked_example", 0)
        if slide_count > 1:
            first_slide: SlideElement | None = None
            kept: list[TrainingElement] = []
            for el in self.elements:
                if el.element_type == "slide":
                    if first_slide is None:
                        first_slide = el  # pyright: ignore[reportAssignmentType]
                        kept.append(el)
                    else:
                        # Merge content into the first slide
                        merged = first_slide.slide.content + "\n\n" + el.slide.content  # pyright: ignore[reportAttributeAccessIssue]
                        object.__setattr__(first_slide.slide, "content", merged)  # pyright: ignore[reportAttributeAccessIssue]
                        logger.info("Merged extra slide into first slide")
                elif el.element_type == "worked_example":
                    if first_slide is None:
                        first_slide = el  # pyright: ignore[reportAssignmentType]
                        kept.append(el)
                    else:
                        logger.warning("Dropped extra worked_example element")
                else:
                    kept.append(el)
            self.elements = kept

        # Minimum exercises — accept what the LLM produced rather than
        # throwing away all content.  The LLM already had Instructor retries
        # to self-correct; if it still can't meet the target, the exercises
        # it did generate are still valuable.
        min_ex = self._min_exercises
        if exercise_count < min_ex:
            logger.warning(
                "Section has %d exercise(s) (target: %d) — accepting as-is",
                exercise_count, min_ex,
            )

        # Trim excess interactive essays — keep first N, drop the rest
        max_essays = self._max_interactive_essays
        essay_count = counts.get("interactive_essay", 0)
        if essay_count > max_essays:
            logger.warning(
                "Trimming interactive essays: %d → %d", essay_count, max_essays,
            )
            seen_essays = 0
            kept: list[TrainingElement] = []
            for el in self.elements:
                if el.element_type == "interactive_essay":
                    seen_essays += 1
                    if seen_essays <= max_essays:
                        kept.append(el)
                else:
                    kept.append(el)
            self.elements = kept

        # Low exercise variety — accept, just warn
        min_types = self._min_exercise_types
        if exercise_count >= min_ex and len(exercise_types_used) < min_types:
            logger.warning(
                "Low exercise variety: %d type(s) with %d exercises (target: %d types)",
                len(exercise_types_used), exercise_count, min_types,
            )

        # Trim excess quizzes — keep first N, drop the rest
        quiz_count = counts.get("quiz", 0)
        if quiz_count > self._max_quizzes:
            logger.warning(
                "Trimming quizzes: %d → %d", quiz_count, self._max_quizzes,
            )
            seen_quizzes = 0
            kept2: list[TrainingElement] = []
            for el in self.elements:
                if el.element_type == "quiz":
                    seen_quizzes += 1
                    if seen_quizzes <= self._max_quizzes:
                        kept2.append(el)
                else:
                    kept2.append(el)
            self.elements = kept2

        # ── Warnings (aspirational targets, not worth a retry) ───────────

        # Target: 5 exercises for hard concepts
        if exercise_count < 5:
            logger.warning(
                "Section has %d exercises (target: 5 for difficult concepts)",
                exercise_count,
            )

        # Target: 4+ different exercise types when 4+ exercises
        if exercise_count >= 4 and len(exercise_types_used) < 4:
            logger.warning(
                "Low exercise variety: %d types used with %d exercises (target: all different types)",
                len(exercise_types_used), exercise_count,
            )

        # Warn on flashcard count (target: 2-3 per section)
        if flashcard_count < 2:
            logger.warning(
                "Low flashcard count: %d (target: 2-3 per section)",
                flashcard_count,
            )
        elif flashcard_count > 4:
            logger.warning(
                "High flashcard count: %d (target: 2-3 per section)",
                flashcard_count,
            )

        return self

    @model_validator(mode="after")
    def enforce_interleaved_order(self) -> "SectionResponse":
        """Preserve LLM's teach-practice interleaving while anchoring bookend elements.

        Core elements (slides + practice) keep their LLM-generated order intact
        so teach-practice cycles are not destroyed. Only bookend elements are
        moved to fixed positions: section_intro first, concept_map/flashcard/
        interactive_essay at the end.
        """
        if not self.elements:
            return self

        anchor_end_roles = {"synthesis", "reveal", "reinforce", "assess"}
        intro = [e for e in self.elements if e.element_type == "section_intro"]
        core = [
            e for e in self.elements
            if ELEMENT_ROLE.get(e.element_type) not in ({"intro"} | anchor_end_roles)
        ]
        synthesis = [e for e in self.elements if e.element_type == "concept_map"]
        reveals = [e for e in self.elements if e.element_type == "error_detection"]
        flashcards = [e for e in self.elements if e.element_type == "flashcard"]
        essays = [e for e in self.elements if e.element_type == "interactive_essay"]

        self.elements = intro + core + synthesis + reveals + flashcards + essays
        return self

    @model_validator(mode="after")
    def enforce_difficulty_progression(self) -> "SectionResponse":
        """Sort exercises by difficulty (easy → hard), keeping teach elements in place.

        Only reorders practice elements relative to each other.
        Teaching elements and bookend elements keep their positions.
        """
        if not self.elements:
            return self

        # Identify indices of exercise elements within the element list
        practice_indices = [
            i for i, e in enumerate(self.elements)
            if e.element_type in EXERCISE_ELEMENT_TYPES
        ]

        if len(practice_indices) <= 1:
            return self

        # Extract practice elements, sort by difficulty (stable sort preserves
        # relative order when difficulties are equal)
        practice_elements = [self.elements[i] for i in practice_indices]
        practice_elements.sort(
            key=lambda e: DIFFICULTY_ORDER.get(
                getattr(e, "difficulty", "medium"), 2
            )
        )

        # Put sorted practice elements back into their original positions
        new_elements = list(self.elements)
        for slot_idx, practice_el in zip(practice_indices, practice_elements):
            new_elements[slot_idx] = practice_el
        self.elements = new_elements

        return self
