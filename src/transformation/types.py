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

from typing import Annotated, Literal, TypedDict, Union

from pydantic import BaseModel, Discriminator, Field, model_validator


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
    "error_detection": "practice",  # Core: LLM order preserved
    "concept_map": "synthesis",     # Anchored near end
    "worked_example": "teach",      # Core: LLM order preserved
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
]

# Element types suitable for assessment (used in reinforcement target suggestions).
AssessmentElementType = Literal[
    "quiz", "flashcard", "fill_in_the_blank", "matching", "ordering",
    "categorization", "error_detection", "analogy", "interactive_essay",
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
            self.correct_index = 0
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

    statement: str = Field(description="The statement with _____ marking each blank")
    answers: list[str] = Field(description="Correct answers for each blank, in order", min_length=1)
    hint: str = Field(default="", description="Optional hint shown to the learner")
    hint_first_letter: bool = Field(
        default=True,
        description="On second failure, reveal first letter of each answer",
    )

    @model_validator(mode="after")
    def _validate_blank_count(self) -> FillInTheBlank:
        blank_count = self.statement.count("_____")
        if blank_count > 0 and blank_count != len(self.answers):
            # Pad or truncate answers to match blank count
            if len(self.answers) < blank_count:
                self.answers = self.answers + [""] * (blank_count - len(self.answers))
            else:
                self.answers = self.answers[:blank_count]
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
        # Truncate to the shorter list so pairs always align
        min_len = min(len(self.left_items), len(self.right_items))
        if len(self.left_items) != len(self.right_items):
            self.left_items = self.left_items[:min_len]
            self.right_items = self.right_items[:min_len]
        if self.pair_explanations and len(self.pair_explanations) != min_len:
            # Pad with empty strings or truncate
            self.pair_explanations = (self.pair_explanations + [""] * min_len)[:min_len]
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
        self.blank_edge_indices = [i for i in self.blank_edge_indices if 0 <= i < n]
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


class OrderingElement(BaseModel):
    """An ordering exercise training element."""

    element_type: Literal["ordering"] = "ordering"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    ordering: OrderingExercise


class CategorizationElement(BaseModel):
    """A categorization exercise training element."""

    element_type: Literal["categorization"] = "categorization"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    categorization: CategorizationExercise


class ErrorDetectionElement(BaseModel):
    """An error detection exercise training element."""

    element_type: Literal["error_detection"] = "error_detection"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    error_detection: ErrorDetectionExercise


class AnalogyElement(BaseModel):
    """An analogy completion exercise training element."""

    element_type: Literal["analogy"] = "analogy"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    analogy: AnalogyExercise


class WorkedExampleElement(BaseModel):
    """A Brilliant-style worked example training element."""

    element_type: Literal["worked_example"] = "worked_example"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    worked_example: WorkedExample


class InteractiveEssayElement(BaseModel):
    """An interactive essay training element."""

    element_type: Literal["interactive_essay"] = "interactive_essay"
    bloom_level: BloomLevel = Field(description="Bloom's Taxonomy cognitive level")
    interactive_essay: InteractiveEssay


TrainingElement = Annotated[
    Union[
        SectionIntroElement, SlideElement, QuizElement, FlashcardElement,
        FillInBlankElement, MatchingElement, OrderingElement, MermaidElement,
        ConceptMapElement, CategorizationElement, ErrorDetectionElement,
        AnalogyElement, WorkedExampleElement, InteractiveEssayElement,
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
