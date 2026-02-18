"""Tests for Pydantic model validation — TrainingElement discriminated union."""

import pytest
from pydantic import TypeAdapter, ValidationError

from src.transformation.types import (
    CurriculumBlueprint,
    ELEMENT_BLOOM_MAP,
    ELEMENT_ROLE,
    ModuleBlueprint,
    SectionBlueprint,
    AnalogyElement,
    AnalogyExercise,
    AnalogyItem,
    CategorizationElement,
    CategorizationExercise,
    CategoryBucket,
    ErrorDetectionElement,
    ErrorDetectionExercise,
    ErrorItem,
    Flashcard,
    FlashcardElement,
    FillInBlankElement,
    InteractiveEssay,
    InteractiveEssayElement,
    MatchingElement,
    MatchingExercise,
    OrderingElement,
    OrderingExercise,
    QuizElement,
    QuizQuestion,
    ReinforcementTarget,
    ReinforcementTargetSet,
    SectionIntro,
    SectionIntroElement,
    SelfExplain,
    Slide,
    SlideElement,
    TrainingElement,
    FillInTheBlank,
)

# TypeAdapter for the discriminated union — needed for dict-based construction
_ElementAdapter = TypeAdapter(TrainingElement)


class TestTrainingElementValidation:
    """TrainingElement must dispatch to the correct variant based on element_type."""

    def test_valid_slide(self) -> None:
        elem = _ElementAdapter.validate_python({
            "element_type": "slide",
            "bloom_level": "understand",
            "slide": {"title": "T", "content": "C"},
        })
        assert isinstance(elem, SlideElement)
        assert elem.slide.title == "T"

    def test_valid_quiz(self) -> None:
        elem = _ElementAdapter.validate_python({
            "element_type": "quiz",
            "bloom_level": "analyze",
            "quiz": {
                "title": "Q",
                "questions": [
                    {"question": "?", "options": ["A", "B"], "correct_index": 0}
                ],
            },
        })
        assert isinstance(elem, QuizElement)

    def test_valid_flashcard(self) -> None:
        elem = _ElementAdapter.validate_python({
            "element_type": "flashcard",
            "bloom_level": "remember",
            "flashcard": {"front": "F", "back": "B"},
        })
        assert isinstance(elem, FlashcardElement)

    def test_valid_fill_in_the_blank(self) -> None:
        elem = _ElementAdapter.validate_python({
            "element_type": "fill_in_the_blank",
            "bloom_level": "apply",
            "fill_in_the_blank": {
                "statement": "The _____ is blue",
                "answers": ["sky"],
            },
        })
        assert isinstance(elem, FillInBlankElement)

    def test_valid_matching(self) -> None:
        elem = _ElementAdapter.validate_python({
            "element_type": "matching",
            "bloom_level": "analyze",
            "matching": {
                "title": "Match",
                "left_items": ["a", "b"],
                "right_items": ["1", "2"],
            },
        })
        assert isinstance(elem, MatchingElement)

    def test_valid_ordering(self) -> None:
        elem = _ElementAdapter.validate_python({
            "element_type": "ordering",
            "bloom_level": "apply",
            "ordering": {
                "title": "Order the Steps",
                "instruction": "Arrange in correct order",
                "items": ["Step 1", "Step 2", "Step 3"],
            },
        })
        assert isinstance(elem, OrderingElement)
        assert len(elem.ordering.items) == 3

    def test_valid_categorization(self) -> None:
        elem = _ElementAdapter.validate_python({
            "element_type": "categorization",
            "bloom_level": "analyze",
            "categorization": {
                "title": "Classify",
                "instruction": "Sort these items",
                "categories": [
                    {"name": "Type A", "items": ["a1", "a2"]},
                    {"name": "Type B", "items": ["b1"]},
                ],
            },
        })
        assert isinstance(elem, CategorizationElement)
        assert len(elem.categorization.categories) == 2

    def test_valid_error_detection(self) -> None:
        elem = _ElementAdapter.validate_python({
            "element_type": "error_detection",
            "bloom_level": "evaluate",
            "error_detection": {
                "title": "Spot the Error",
                "instruction": "Find the mistake",
                "items": [
                    {
                        "statement": "Water boils at 50C",
                        "error_explanation": "Water boils at 100C at sea level",
                        "corrected_statement": "Water boils at 100C",
                    },
                ],
            },
        })
        assert isinstance(elem, ErrorDetectionElement)
        assert len(elem.error_detection.items) == 1

    def test_valid_analogy(self) -> None:
        elem = _ElementAdapter.validate_python({
            "element_type": "analogy",
            "bloom_level": "analyze",
            "analogy": {
                "title": "Analogy Challenge",
                "items": [
                    {
                        "stem": "A is to B as C is to ___",
                        "answer": "D",
                        "distractors": ["E", "F"],
                    },
                ],
            },
        })
        assert isinstance(elem, AnalogyElement)
        assert elem.analogy.items[0].answer == "D"

    def test_rejects_unknown_element_type(self) -> None:
        """An unrecognized element_type should fail."""
        with pytest.raises(ValidationError):
            _ElementAdapter.validate_python({
                "element_type": "unknown",
                "bloom_level": "understand",
            })

    def test_rejects_missing_content_field(self) -> None:
        """element_type='slide' but no slide field should fail."""
        with pytest.raises(ValidationError):
            SlideElement(
                bloom_level="understand",
                slide=None,  # type: ignore[arg-type]
            )

    def test_valid_section_intro(self) -> None:
        elem = _ElementAdapter.validate_python({
            "element_type": "section_intro",
            "bloom_level": "understand",
            "section_intro": {
                "title": "Welcome",
                "content": "In this section you will learn about diversification.",
            },
        })
        assert isinstance(elem, SectionIntroElement)
        assert elem.section_intro.title == "Welcome"

    def test_valid_interactive_essay(self) -> None:
        elem = InteractiveEssayElement(
            bloom_level="evaluate",
            interactive_essay=InteractiveEssay(
                title="Chapter 1 Checkpoint",
                concepts_tested=["bonds", "yields"],
                prompts=[
                    SelfExplain(
                        prompt="Explain bond pricing.",
                        key_points=["present value", "coupon rate"],
                        example_response="Bond price is ...",
                    ),
                ],
            ),
        )
        assert elem.interactive_essay.passing_threshold == 0.7

    def test_rejects_interactive_essay_empty_prompts(self) -> None:
        """InteractiveEssay requires at least 1 prompt."""
        with pytest.raises(ValidationError):
            InteractiveEssay(
                title="Checkpoint",
                concepts_tested=["concept"],
                prompts=[],
            )

    def test_direct_construction(self) -> None:
        """Concrete element classes can be constructed directly."""
        elem = SlideElement(
            bloom_level="understand",
            slide=Slide(title="T", content="C"),
        )
        assert elem.element_type == "slide"
        assert elem.slide.title == "T"


class TestQuizQuestionValidation:
    """QuizQuestion must have at least 2 options."""

    def test_rejects_single_option(self) -> None:
        with pytest.raises(ValidationError):
            QuizQuestion(question="?", options=["A"], correct_index=0)

    def test_accepts_two_options(self) -> None:
        q = QuizQuestion(question="?", options=["A", "B"], correct_index=1)
        assert q.correct_index == 1


class TestReinforcementTarget:
    """Tests for ReinforcementTarget and ReinforcementTargetSet models."""

    def test_valid_target(self) -> None:
        target = ReinforcementTarget(
            concept_name="Diversification",
            target_insight="Correlations spike in crises, reducing diversification benefit",
            angle="edge_case",
            bloom_level="analyze",
            suggested_element_type="quiz",
        )
        assert target.angle == "edge_case"
        assert target.bloom_level == "analyze"

    def test_valid_target_set(self) -> None:
        targets = [
            ReinforcementTarget(
                concept_name=f"Concept {i}",
                target_insight=f"Insight {i}",
                angle="mechanism",
                bloom_level="understand",
                suggested_element_type="flashcard",
            )
            for i in range(3)
        ]
        target_set = ReinforcementTargetSet(targets=targets)
        assert len(target_set.targets) == 3

    def test_rejects_too_few_targets(self) -> None:
        with pytest.raises(ValidationError):
            ReinforcementTargetSet(targets=[
                ReinforcementTarget(
                    concept_name="X",
                    target_insight="Y",
                    angle="mechanism",
                    bloom_level="remember",
                    suggested_element_type="quiz",
                ),
            ])

    def test_all_angles_accepted(self) -> None:
        for angle in ("mechanism", "connection", "application", "edge_case", "contrast", "consequence"):
            target = ReinforcementTarget(
                concept_name="C",
                target_insight="I",
                angle=angle,
                bloom_level="understand",
                suggested_element_type="quiz",
            )
            assert target.angle == angle

    def test_all_assessment_types_accepted(self) -> None:
        for elem_type in (
            "quiz", "flashcard", "fill_in_the_blank", "matching", "ordering",
            "categorization", "error_detection", "analogy", "interactive_essay",
        ):
            target = ReinforcementTarget(
                concept_name="C",
                target_insight="I",
                angle="mechanism",
                bloom_level="understand",
                suggested_element_type=elem_type,
            )
            assert target.suggested_element_type == elem_type


class TestElementBloomMap:
    """Tests for the canonical ELEMENT_BLOOM_MAP."""

    def test_all_element_types_mapped(self) -> None:
        expected = {
            "section_intro", "flashcard", "slide", "mermaid", "quiz", "matching",
            "ordering", "fill_in_the_blank", "categorization", "analogy",
            "concept_map", "error_detection", "interactive_essay",
        }
        assert set(ELEMENT_BLOOM_MAP.keys()) == expected

    def test_bloom_values_are_valid(self) -> None:
        valid_levels = {"remember", "understand", "apply", "analyze", "evaluate", "create"}
        for elem_type, level in ELEMENT_BLOOM_MAP.items():
            assert level in valid_levels, f"{elem_type} has invalid bloom level {level}"

    def test_specific_mappings(self) -> None:
        assert ELEMENT_BLOOM_MAP["section_intro"] == "understand"
        assert ELEMENT_BLOOM_MAP["flashcard"] == "remember"
        assert ELEMENT_BLOOM_MAP["slide"] == "understand"
        assert ELEMENT_BLOOM_MAP["mermaid"] == "understand"
        assert ELEMENT_BLOOM_MAP["quiz"] == "apply"
        assert ELEMENT_BLOOM_MAP["matching"] == "apply"
        assert ELEMENT_BLOOM_MAP["ordering"] == "apply"
        assert ELEMENT_BLOOM_MAP["fill_in_the_blank"] == "analyze"
        assert ELEMENT_BLOOM_MAP["categorization"] == "analyze"
        assert ELEMENT_BLOOM_MAP["analogy"] == "analyze"
        assert ELEMENT_BLOOM_MAP["concept_map"] == "apply"
        assert ELEMENT_BLOOM_MAP["error_detection"] == "evaluate"
        assert ELEMENT_BLOOM_MAP["interactive_essay"] == "evaluate"


class TestElementRole:
    """Tests for the ELEMENT_ROLE classification dict."""

    def test_section_intro_is_intro(self) -> None:
        assert ELEMENT_ROLE["section_intro"] == "intro"

    def test_slides_are_teach(self) -> None:
        assert ELEMENT_ROLE["slide"] == "teach"
        assert ELEMENT_ROLE["mermaid"] == "teach"

    def test_exercises_are_practice(self) -> None:
        assert ELEMENT_ROLE["quiz"] == "practice"
        assert ELEMENT_ROLE["matching"] == "practice"
        assert ELEMENT_ROLE["fill_in_the_blank"] == "practice"

    def test_flashcard_is_reinforce(self) -> None:
        assert ELEMENT_ROLE["flashcard"] == "reinforce"

    def test_interactive_essay_is_assess(self) -> None:
        assert ELEMENT_ROLE["interactive_essay"] == "assess"

    def test_all_bloom_map_keys_have_role(self) -> None:
        for key in ELEMENT_BLOOM_MAP:
            assert key in ELEMENT_ROLE, f"{key} missing from ELEMENT_ROLE"


class TestSectionBlueprintFocusConcepts:
    """Tests for the focus_concepts field on SectionBlueprint."""

    def test_default_empty_list(self) -> None:
        bp = SectionBlueprint(title="Test", source_section_title="Test")
        assert bp.focus_concepts == []

    def test_accepts_concept_names(self) -> None:
        bp = SectionBlueprint(
            title="Test",
            source_section_title="Test",
            focus_concepts=["basis", "span"],
        )
        assert bp.focus_concepts == ["basis", "span"]

    def test_backward_compat_from_dict_without_field(self) -> None:
        bp = SectionBlueprint.model_validate({
            "title": "Test",
            "source_section_title": "Test",
        })
        assert bp.focus_concepts == []

    def test_roundtrip_serialization(self) -> None:
        bp = SectionBlueprint(
            title="Test",
            source_section_title="Src",
            focus_concepts=["concept_a", "concept_b"],
        )
        data = bp.model_dump()
        restored = SectionBlueprint.model_validate(data)
        assert restored.focus_concepts == ["concept_a", "concept_b"]

    def test_blueprint_with_focus_concepts_in_module(self) -> None:
        blueprint = CurriculumBlueprint(
            course_title="Test Course",
            modules=[
                ModuleBlueprint(
                    title="M1",
                    source_chapter_number=1,
                    sections=[
                        SectionBlueprint(
                            title="Unit A",
                            source_section_title="Section 1",
                            focus_concepts=["concept_x"],
                        ),
                        SectionBlueprint(
                            title="Unit B",
                            source_section_title="Section 1",
                            focus_concepts=["concept_y"],
                        ),
                    ],
                ),
            ],
        )
        # Both sections share source_section_title but have different focus
        assert blueprint.modules[0].sections[0].focus_concepts == ["concept_x"]
        assert blueprint.modules[0].sections[1].focus_concepts == ["concept_y"]
