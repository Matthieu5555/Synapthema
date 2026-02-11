"""Tests for the content designer — Stage 2 deep module interface.

Uses a mock LLMClient to test transform_chapter() without API calls.
"""

from __future__ import annotations

from typing import TypeVar

import pytest

from src.extraction.types import Chapter, Section
from src.transformation.content_designer import transform_chapter, MIN_SECTION_TEXT_LENGTH
from src.transformation.types import (
    Flashcard,
    FlashcardElement,
    Slide,
    SlideElement,
    TrainingElement,
)

T = TypeVar("T")


class MockLLMClient:
    """Test double returning a fixed structured response.

    Handles both ReinforcementTargetSet (Phase 1) and SectionResponse (Phase 2).
    """

    def __init__(self, elements: list | None = None) -> None:
        self._elements = elements or [
            SlideElement(
                bloom_level="understand",
                slide=Slide(title="Mock Slide", content="Mock content."),
            ),
            FlashcardElement(
                bloom_level="remember",
                flashcard=Flashcard(front="Q", back="A"),
            ),
        ]
        self.call_count = 0
        self.last_system_prompt: str = ""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.last_system_prompt = system_prompt
        return "mock response"

    def complete_light(self, system_prompt: str, user_prompt: str) -> str:
        return self.complete(system_prompt, user_prompt)

    def complete_structured(
        self, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> T:
        self.call_count += 1
        self.last_system_prompt = system_prompt

        from src.transformation.types import ReinforcementTargetSet as RTS
        if response_model is RTS:
            from src.transformation.types import ReinforcementTarget
            return RTS(targets=[
                ReinforcementTarget(
                    concept_name="MockConcept",
                    target_insight="Mock insight about mechanisms",
                    angle="mechanism",
                    bloom_level="understand",
                    suggested_element_type="quiz",
                ),
                ReinforcementTarget(
                    concept_name="MockConcept2",
                    target_insight="Mock insight about connections",
                    angle="connection",
                    bloom_level="apply",
                    suggested_element_type="flashcard",
                ),
                ReinforcementTarget(
                    concept_name="MockConcept3",
                    target_insight="Mock insight about edge cases",
                    angle="edge_case",
                    bloom_level="analyze",
                    suggested_element_type="self_explain",
                ),
            ])  # type: ignore[return-value]

        from src.transformation.content_designer import SectionResponse
        return SectionResponse(elements=self._elements)  # type: ignore[return-value]

    def complete_structured_light(
        self, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> T:
        return self.complete_structured(system_prompt, user_prompt, response_model)


class FailingLLMClient:
    """Test double that always raises LLMError."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("fail")

    def complete_light(self, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("fail")

    def complete_structured(
        self, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> T:
        from src.transformation.llm_client import LLMError
        raise LLMError("API failed")

    def complete_structured_light(
        self, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> T:
        from src.transformation.llm_client import LLMError
        raise LLMError("API failed")


def _make_chapter(
    sections: tuple[Section, ...] | None = None,
) -> Chapter:
    """Build a test Chapter with configurable sections."""
    if sections is None:
        sections = (
            Section(
                title="Section One",
                level=3,
                start_page=1,
                end_page=5,
                text="A" * 200,
            ),
        )
    return Chapter(
        chapter_number=1,
        title="Test Chapter",
        start_page=1,
        end_page=10,
        sections=sections,
    )


class TestTransformChapter:
    """Tests for transform_chapter — the Stage 2 entry point."""

    def test_transforms_sections_into_elements(self) -> None:
        client = MockLLMClient()
        chapter = _make_chapter()

        module = transform_chapter(chapter, client)

        assert module.chapter_number == 1
        assert module.title == "Test Chapter"
        assert len(module.all_elements) == 2  # slide + flashcard from mock
        # 2 calls per section: target selection + element generation
        assert client.call_count == 2

    def test_skips_short_sections(self) -> None:
        short = Section(
            title="Short",
            level=3,
            start_page=1,
            end_page=1,
            text="x" * (MIN_SECTION_TEXT_LENGTH - 1),
        )
        long = Section(
            title="Long",
            level=3,
            start_page=2,
            end_page=5,
            text="y" * 500,
        )
        client = MockLLMClient()
        chapter = _make_chapter(sections=(short, long))

        module = transform_chapter(chapter, client)

        # Only the long section should be processed (2 calls: target + generation)
        assert client.call_count == 2
        assert len(module.all_elements) == 2

    def test_handles_all_short_sections(self) -> None:
        short = Section(
            title="Too Short", level=3, start_page=1, end_page=1, text="tiny"
        )
        client = MockLLMClient()
        chapter = _make_chapter(sections=(short,))

        module = transform_chapter(chapter, client)

        assert len(module.all_elements) == 0
        assert client.call_count == 0

    def test_fallback_on_llm_failure(self) -> None:
        """Failed LLM calls should produce fallback sections, not crash."""
        client = FailingLLMClient()
        chapter = _make_chapter()

        module = transform_chapter(chapter, client)

        # Should produce a fallback section with one slide element
        assert len(module.sections) == 1
        assert len(module.all_elements) == 1
        assert module.all_elements[0].element_type == "slide"
        # Fallback section should have an error verification note
        assert any("[error]" in n for n in module.sections[0].verification_notes)

    def test_multiple_sections_combined(self) -> None:
        s1 = Section(title="S1", level=3, start_page=1, end_page=3, text="a" * 200)
        s2 = Section(title="S2", level=3, start_page=4, end_page=6, text="b" * 200)
        client = MockLLMClient()
        chapter = _make_chapter(sections=(s1, s2))

        module = transform_chapter(chapter, client)

        # 2 sections × 2 calls each (target selection + generation)
        assert client.call_count == 4
        assert len(module.all_elements) == 4  # 2 elements per section


class TestTransformChapterWithAnalysis:
    """Tests for transform_chapter with deep reading analysis context."""

    def test_accepts_chapter_analysis(self) -> None:
        """Backward compat: chapter_analysis is optional."""
        from src.transformation.analysis_types import (
            ChapterAnalysis,
            ConceptEntry,
            SectionCharacterization,
        )

        analysis = ChapterAnalysis(
            chapter_number=1,
            chapter_title="Test Chapter",
            concepts=[
                ConceptEntry(
                    name="TestConcept",
                    definition="A test concept.",
                    concept_type="definition",
                    section_title="Section One",
                    importance="core",
                ),
            ],
            section_characterizations=[
                SectionCharacterization(
                    section_title="Section One",
                    dominant_content_type="conceptual",
                    has_definitions=True,
                ),
            ],
        )

        client = MockLLMClient()
        chapter = _make_chapter()

        module = transform_chapter(
            chapter, client,
            chapter_analysis=analysis,
            prior_concepts=["PriorConcept"],
        )

        assert module.chapter_number == 1
        # 2 calls: target selection + element generation
        assert client.call_count == 2

class TestBloomSupplementIntegration:
    """Tests that Bloom's supplements are applied to the system prompt."""

    def test_bloom_supplement_appended_to_system_prompt(self) -> None:
        """When a blueprint specifies bloom_target, the supplement is appended."""
        from src.transformation.types import ModuleBlueprint, SectionBlueprint

        blueprint = ModuleBlueprint(
            title="Test Module",
            source_chapter_number=1,
            sections=[
                SectionBlueprint(
                    title="Section One",
                    source_section_title="Section One",
                    bloom_target="analyze",
                    template="compare_contrast",
                ),
            ],
        )
        client = MockLLMClient()
        chapter = _make_chapter()

        transform_chapter(chapter, client, blueprint=blueprint)

        # 2 calls: target selection + element generation
        assert client.call_count == 2
        # last_system_prompt is from the element generation call (Phase 2)
        assert "Bloom's Focus: Analyze" in client.last_system_prompt
        assert "comparison" in client.last_system_prompt.lower() or "decomposition" in client.last_system_prompt.lower()

    def test_default_bloom_supplement_without_blueprint(self) -> None:
        """Without a blueprint, defaults to 'understand' supplement."""
        client = MockLLMClient()
        chapter = _make_chapter()

        transform_chapter(chapter, client)

        # 2 calls: 1 target selection + 1 element generation
        assert client.call_count == 2
        assert "Bloom's Focus: Understand" in client.last_system_prompt


class TestTwoPhaseGeneration:
    """Tests for Phase 1 target selection + Phase 2 element generation."""

    def test_falls_back_on_target_selection_failure(self) -> None:
        """If Phase 1 fails, Phase 2 still runs (without targets)."""

        class Phase1FailingClient:
            """Fails on ReinforcementTargetSet, succeeds on SectionResponse."""

            def __init__(self) -> None:
                self.call_count = 0

            def complete(self, system_prompt: str, user_prompt: str) -> str:
                return "mock"

            def complete_light(self, system_prompt: str, user_prompt: str) -> str:
                return "mock"

            def complete_structured(
                self, system_prompt: str, user_prompt: str, response_model: type[T]
            ) -> T:
                self.call_count += 1
                from src.transformation.content_designer import SectionResponse
                from src.transformation.types import ReinforcementTargetSet as RTS
                if response_model is RTS:
                    from src.transformation.llm_client import LLMError
                    raise LLMError("Target selection failed")
                return SectionResponse(elements=[
                    SlideElement(
                        bloom_level="understand",
                        slide=Slide(title="Fallback", content="Content."),
                    ),
                    FlashcardElement(
                        bloom_level="remember",
                        flashcard=Flashcard(front="Q", back="A"),
                    ),
                ])  # type: ignore[return-value]

            def complete_structured_light(
                self, system_prompt: str, user_prompt: str, response_model: type[T]
            ) -> T:
                return self.complete_structured(system_prompt, user_prompt, response_model)

        client = Phase1FailingClient()
        chapter = _make_chapter()

        module = transform_chapter(chapter, client)

        # Phase 1 failed (1 call) + Phase 2 succeeded (1 call) = 2 calls
        assert client.call_count == 2
        assert len(module.all_elements) == 2  # slide + flashcard from Phase 2
        assert module.all_elements[0].slide.title == "Fallback"  # type: ignore[union-attr]


class TestSourceVerification:
    """Tests for post-generation source verification."""

    def test_extract_formulas(self) -> None:
        from src.transformation.content_designer import _extract_formulas

        text = "The formula is $E = mc^2$ and also $$\\frac{a}{b}$$."
        formulas = _extract_formulas(text)
        assert "E = mc^2" in formulas
        assert "\\frac{a}{b}" in formulas

    def test_extract_numeric_claims(self) -> None:
        from src.transformation.content_designer import _extract_numeric_claims

        text = "The rate is 15% and the spread is 100 basis points."
        nums = _extract_numeric_claims(text)
        assert any("15%" in n for n in nums)
        assert any("100 basis points" in n for n in nums)

    def test_extract_definitions(self) -> None:
        from src.transformation.content_designer import _extract_definitions

        text = "Duration is defined as the sensitivity of bond price to yield changes."
        defs = _extract_definitions(text)
        assert len(defs) >= 1
        assert "Duration" in defs[0]

    def test_check_claim_found_in_source(self) -> None:
        from src.transformation.content_designer import _check_claim_against_source

        source = "The Sharpe ratio measures risk-adjusted return using standard deviation."
        assert _check_claim_against_source("Sharpe ratio", source) is True

    def test_check_claim_not_found_in_source(self) -> None:
        from src.transformation.content_designer import _check_claim_against_source

        source = "This section covers bond pricing."
        assert _check_claim_against_source("quantum entanglement theory", source) is False

    def test_verify_elements_flags_hallucinated_formula(self) -> None:
        from src.transformation.content_designer import _verify_elements

        elements = [
            SlideElement(
                bloom_level="understand",
                slide=Slide(
                    title="Test",
                    content="The formula is $Z = \\alpha \\beta$.",
                ),
            ),
        ]
        source_text = "This section covers X and Y variables only."
        warnings = _verify_elements(elements, source_text)
        # Should flag the formula with Z/alpha/beta not in source
        assert len(warnings) > 0
        assert any("formula" in w for w in warnings)

    def test_verify_elements_passes_valid_content(self) -> None:
        from src.transformation.content_designer import _verify_elements

        elements = [
            FlashcardElement(
                bloom_level="remember",
                flashcard=Flashcard(front="What is duration?", back="sensitivity"),
            ),
        ]
        source_text = "Duration is the sensitivity of bond price to yield changes."
        warnings = _verify_elements(elements, source_text)
        # "sensitivity" appears in source, no warnings expected
        assert len(warnings) == 0

    def test_verification_notes_attached_to_section(self) -> None:
        """Verification warnings should appear in TrainingSection.verification_notes."""
        elements_with_hallucination = [
            SlideElement(
                bloom_level="understand",
                slide=Slide(
                    title="Hallucinated",
                    content="The formula is $Z = \\omega \\psi$ and rate is 99.9%.",
                ),
            ),
        ]
        client = MockLLMClient(elements=elements_with_hallucination)
        section = Section(
            title="Section One",
            level=3,
            start_page=1,
            end_page=5,
            text="This section discusses X and Y only. " * 20,
        )
        chapter = _make_chapter(sections=(section,))

        module = transform_chapter(chapter, client)

        # The section should have verification notes
        assert len(module.sections) == 1
        notes = module.sections[0].verification_notes
        assert len(notes) > 0

    def test_jaccard_similarity(self) -> None:
        from src.transformation.content_designer import _jaccard_similarity

        # High overlap
        assert _jaccard_similarity("the quick brown fox", "the quick brown dog") > 0.5
        # No overlap
        assert _jaccard_similarity("abc def", "xyz uvw") == 0.0
        # Empty strings
        assert _jaccard_similarity("", "something") == 0.0


class TestSectionResponseValidators:
    """Tests for SectionResponse bloom override and distribution validators."""

    def test_bloom_override_corrects_wrong_level(self) -> None:
        """LLM-chosen bloom_level is overridden by ELEMENT_BLOOM_MAP."""
        from src.transformation.content_designer import SectionResponse

        resp = SectionResponse(elements=[
            SlideElement(
                bloom_level="analyze",  # wrong — should be "understand"
                slide=Slide(title="T", content="C"),
            ),
            FlashcardElement(
                bloom_level="evaluate",  # wrong — should be "remember"
                flashcard=Flashcard(front="Q", back="A"),
            ),
        ])
        assert resp.elements[0].bloom_level == "understand"
        assert resp.elements[1].bloom_level == "remember"

    def test_rejects_no_slides(self) -> None:
        """Section must contain at least 1 slide."""
        from pydantic import ValidationError
        from src.transformation.content_designer import SectionResponse

        with pytest.raises(ValidationError, match="at least 1 slide"):
            SectionResponse(elements=[
                FlashcardElement(
                    bloom_level="remember",
                    flashcard=Flashcard(front="Q", back="A"),
                ),
            ])

    def test_rejects_no_assessments(self) -> None:
        """Section must contain at least 1 assessment element."""
        from pydantic import ValidationError
        from src.transformation.content_designer import SectionResponse

        with pytest.raises(ValidationError, match="at least 1 assessment"):
            SectionResponse(elements=[
                SlideElement(
                    bloom_level="understand",
                    slide=Slide(title="T", content="C"),
                ),
            ])

    def test_accepts_valid_distribution(self) -> None:
        """A section with slides and assessments should pass validation."""
        from src.transformation.content_designer import SectionResponse

        resp = SectionResponse(elements=[
            SlideElement(
                bloom_level="understand",
                slide=Slide(title="T", content="C"),
            ),
            FlashcardElement(
                bloom_level="remember",
                flashcard=Flashcard(front="Q", back="A"),
            ),
        ])
        assert len(resp.elements) == 2
