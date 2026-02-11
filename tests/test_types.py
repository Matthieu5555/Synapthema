"""Tests for Pydantic model validation — TrainingElement discriminated union."""

import pytest
from pydantic import TypeAdapter, ValidationError

from src.transformation.types import (
    ELEMENT_BLOOM_MAP,
    Flashcard,
    FlashcardElement,
    FillInBlankElement,
    InteractiveEssay,
    InteractiveEssayElement,
    MatchingElement,
    MatchingExercise,
    QuizElement,
    QuizQuestion,
    ReinforcementTarget,
    ReinforcementTargetSet,
    SelfExplain,
    SelfExplainElement,
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

    def test_valid_self_explain(self) -> None:
        elem = SelfExplainElement(
            bloom_level="evaluate",
            self_explain=SelfExplain(
                prompt="Explain diversification.",
                key_points=["reduces risk", "uncorrelated assets"],
                example_response="Diversification works by ...",
            ),
        )
        assert elem.self_explain.minimum_words == 50

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

    def test_rejects_self_explain_single_key_point(self) -> None:
        """SelfExplain requires at least 2 key_points."""
        with pytest.raises(ValidationError):
            SelfExplain(
                prompt="Explain.",
                key_points=["only one"],
                example_response="Example.",
            )

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
        for elem_type in ("quiz", "flashcard", "fill_in_the_blank", "matching", "self_explain"):
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
            "flashcard", "slide", "mermaid", "quiz", "matching",
            "fill_in_the_blank", "concept_map", "self_explain", "interactive_essay",
        }
        assert set(ELEMENT_BLOOM_MAP.keys()) == expected

    def test_bloom_values_are_valid(self) -> None:
        valid_levels = {"remember", "understand", "apply", "analyze", "evaluate", "create"}
        for elem_type, level in ELEMENT_BLOOM_MAP.items():
            assert level in valid_levels, f"{elem_type} has invalid bloom level {level}"

    def test_specific_mappings(self) -> None:
        assert ELEMENT_BLOOM_MAP["flashcard"] == "remember"
        assert ELEMENT_BLOOM_MAP["slide"] == "understand"
        assert ELEMENT_BLOOM_MAP["mermaid"] == "understand"
        assert ELEMENT_BLOOM_MAP["quiz"] == "apply"
        assert ELEMENT_BLOOM_MAP["matching"] == "apply"
        assert ELEMENT_BLOOM_MAP["fill_in_the_blank"] == "apply"
        assert ELEMENT_BLOOM_MAP["concept_map"] == "analyze"
        assert ELEMENT_BLOOM_MAP["self_explain"] == "evaluate"
        assert ELEMENT_BLOOM_MAP["interactive_essay"] == "evaluate"
