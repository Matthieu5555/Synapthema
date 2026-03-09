"""Tests for the prompt template module — pure functions, no LLM calls."""

from src.transformation.content_designer_prompts import (
    _smart_truncate,
    build_section_prompt,
    build_target_selection_prompt,
    BLOOM_PROMPT_SUPPLEMENTS,
    MAX_TEXT_LENGTH,
    SYSTEM_PROMPT,
    TARGET_SELECTION_PROMPT,
    TEMPLATE_DESCRIPTIONS,
)


class TestSmartTruncate:
    """Tests for _smart_truncate — the text truncation logic."""

    def test_short_text_unchanged(self) -> None:
        text = "Hello world"
        assert _smart_truncate(text, 100) == text

    def test_exact_limit_unchanged(self) -> None:
        text = "x" * 100
        assert _smart_truncate(text, 100) == text

    def test_long_text_truncated(self) -> None:
        text = "A" * 5000 + "B" * 5000 + "C" * 5000
        result = _smart_truncate(text, 1000)
        assert len(result) <= 15000  # shorter than original
        assert result.startswith("A")  # preserves beginning
        assert result.endswith("C")  # preserves end
        assert "[... content truncated for length ...]" in result

    def test_preserves_head_and_tail(self) -> None:
        head = "HEAD_CONTENT " * 100
        tail = " TAIL_CONTENT" * 100
        text = head + "MIDDLE " * 500 + tail
        result = _smart_truncate(text, 500)
        assert "HEAD_CONTENT" in result
        assert "TAIL_CONTENT" in result


class TestBuildSectionPrompt:
    """Tests for build_section_prompt — the user prompt template."""

    def test_includes_chapter_and_section_titles(self) -> None:
        result = build_section_prompt(
            section_title="Derivatives",
            section_text="Content here.",
            chapter_title="Chapter 5: Calculus",
            image_count=0,
            table_count=0,
        )
        assert "Chapter 5: Calculus" in result
        assert "Derivatives" in result

    def test_includes_media_notes_when_present(self) -> None:
        result = build_section_prompt(
            section_title="Sec",
            section_text="Text",
            chapter_title="Ch",
            image_count=3,
            table_count=2,
        )
        assert "3 image(s)" in result
        assert "2 table(s)" in result

    def test_no_media_note_when_absent(self) -> None:
        result = build_section_prompt(
            section_title="Sec",
            section_text="Text",
            chapter_title="Ch",
            image_count=0,
            table_count=0,
        )
        assert "no images or tables" in result

    def test_long_text_gets_truncated(self) -> None:
        long_text = "word " * 10000
        result = build_section_prompt(
            section_title="S",
            section_text=long_text,
            chapter_title="C",
            image_count=0,
            table_count=0,
        )
        # The prompt should be shorter than the raw text
        assert len(result) < len(long_text)

    def test_includes_template_description(self) -> None:
        result = build_section_prompt(
            section_title="Spot Rates",
            section_text="Content about spot rates.",
            chapter_title="Term Structure",
            image_count=0,
            table_count=0,
            template="worked_example",
        )
        assert "WORKED EXAMPLE" in result

    def test_includes_source_pages(self) -> None:
        result = build_section_prompt(
            section_title="Sec",
            section_text="Text",
            chapter_title="Ch",
            image_count=0,
            table_count=0,
            source_pages=(42, 48),
        )
        assert "pp. 42-48" in result

    def test_includes_prior_sections_for_crossrefs(self) -> None:
        result = build_section_prompt(
            section_title="Duration",
            section_text="Content",
            chapter_title="Fixed Income",
            image_count=0,
            table_count=0,
            prior_sections=["Bond Pricing", "Yield Measures"],
        )
        assert "Bond Pricing" in result
        assert "Yield Measures" in result
        assert "cross-reference" in result.lower()

    def test_uses_module_in_header(self) -> None:
        result = build_section_prompt(
            section_title="Sec",
            section_text="Text",
            chapter_title="Module Title",
            image_count=0,
            table_count=0,
        )
        assert "Module: Module Title" in result


class TestNewElementTypes:
    """Tests for section_intro, interactive_essay, and ordering in prompts."""

    def test_milestone_assessment_template_exists(self) -> None:
        assert "milestone_assessment" in TEMPLATE_DESCRIPTIONS

    def test_section_intro_in_system_prompt(self) -> None:
        assert "section_intro" in SYSTEM_PROMPT

    def test_interactive_essay_in_system_prompt(self) -> None:
        assert "interactive_essay" in SYSTEM_PROMPT

    def test_element_ordering_instruction_in_system_prompt(self) -> None:
        assert "One Slide, Then Drill" in SYSTEM_PROMPT


class TestConceptContext:
    """Tests for deep reading analysis context in section prompts."""

    def test_includes_section_concepts(self) -> None:
        from src.transformation.analysis_types import ConceptEntry

        concepts = [
            ConceptEntry(
                name="Duration",
                definition="Sensitivity of bond price to interest rate changes.",
                concept_type="formula",
                section_title="Bond Math",
                key_terms=["modified duration", "Macaulay duration"],
            ),
            ConceptEntry(
                name="Convexity",
                definition="Second-order sensitivity of bond price.",
                concept_type="formula",
                section_title="Bond Math",
            ),
        ]
        result = build_section_prompt(
            section_title="Bond Math",
            section_text="Content about bonds.",
            chapter_title="Fixed Income",
            image_count=0,
            table_count=0,
            section_concepts=concepts,
        )
        assert "Concepts in This Section" in result
        assert "Duration" in result
        assert "Convexity" in result
        assert "formula" in result
        assert "modified duration" in result
        assert "EVERY concept" in result

    def test_includes_prior_concepts(self) -> None:
        result = build_section_prompt(
            section_title="Advanced",
            section_text="Content.",
            chapter_title="Ch",
            image_count=0,
            table_count=0,
            prior_concepts=["Present value", "Discount rate", "Bond pricing"],
        )
        assert "Prior concepts (already taught)" in result
        assert "Present value" in result
        assert "Bond pricing" in result
        assert "re-explaining" in result

    def test_includes_section_characterization(self) -> None:
        from src.transformation.analysis_types import SectionCharacterization

        sc = SectionCharacterization(
            section_title="Risk Metrics",
            dominant_content_type="procedural",
            has_formulas=True,
            has_procedures=True,
            has_examples=True,
            difficulty_estimate="advanced",
            summary="Covers VaR calculation step by step.",
        )
        result = build_section_prompt(
            section_title="Risk Metrics",
            section_text="Content.",
            chapter_title="Risk",
            image_count=0,
            table_count=0,
            section_characterization=sc,
        )
        assert "Content Analysis" in result
        assert "procedural" in result
        assert "advanced" in result
        assert "formulas" in result
        assert "VaR calculation" in result

    def test_no_concept_blocks_when_not_provided(self) -> None:
        """Backward compat: without concept args, prompt is unchanged."""
        result = build_section_prompt(
            section_title="Sec",
            section_text="Text",
            chapter_title="Ch",
            image_count=0,
            table_count=0,
        )
        assert "Concepts in This Section" not in result
        assert "Prior concepts (already taught)" not in result
        assert "Content Analysis" not in result

    def test_all_context_combined(self) -> None:
        """All concept context blocks can coexist with existing blocks."""
        from src.transformation.analysis_types import ConceptEntry, SectionCharacterization

        result = build_section_prompt(
            section_title="Sec",
            section_text="Content " * 50,
            chapter_title="Ch",
            image_count=1,
            table_count=1,
            template="worked_example",
            source_pages=(10, 15),
            prior_sections=["Intro"],
            learning_objectives=["Understand X"],
            bloom_target="apply",
            section_concepts=[
                ConceptEntry(
                    name="X",
                    definition="Something.",
                    concept_type="definition",
                    section_title="Sec",
                ),
            ],
            prior_concepts=["Y"],
            section_characterization=SectionCharacterization(
                section_title="Sec",
                dominant_content_type="applied",
                has_examples=True,
            ),
        )
        # All blocks present
        assert "WORKED EXAMPLE" in result
        assert "pp. 10-15" in result
        assert "Intro" in result
        assert "Understand X" in result
        assert "apply" in result.lower()
        assert "Concepts in This Section" in result
        assert "Prior concepts (already taught)" in result
        assert "Content Analysis" in result


class TestBloomPromptSupplements:
    """Tests for Bloom's-level-specific prompt supplements."""

    def test_all_bloom_levels_have_supplements(self) -> None:
        expected_levels = {"remember", "understand", "apply", "analyze", "evaluate", "create"}
        assert set(BLOOM_PROMPT_SUPPLEMENTS.keys()) == expected_levels

    def test_remember_contains_recall_keywords(self) -> None:
        supplement = BLOOM_PROMPT_SUPPLEMENTS["remember"]
        assert "terminology" in supplement.lower()
        assert "flashcard" in supplement.lower()

    def test_understand_contains_analogy_keywords(self) -> None:
        supplement = BLOOM_PROMPT_SUPPLEMENTS["understand"]
        assert "analog" in supplement.lower()
        assert "paraphras" in supplement.lower()

    def test_apply_contains_scenario_keywords(self) -> None:
        supplement = BLOOM_PROMPT_SUPPLEMENTS["apply"]
        assert "scenario" in supplement.lower() or "situation" in supplement.lower()
        assert "numerical" in supplement.lower() or "novel" in supplement.lower()

    def test_analyze_contains_comparison_keywords(self) -> None:
        supplement = BLOOM_PROMPT_SUPPLEMENTS["analyze"]
        assert "compare" in supplement.lower() or "decompos" in supplement.lower() or "error" in supplement.lower()
        assert "step" in supplement.lower()

    def test_evaluate_contains_judgment_keywords(self) -> None:
        supplement = BLOOM_PROMPT_SUPPLEMENTS["evaluate"]
        assert "judgment" in supplement.lower() or "justification" in supplement.lower()

    def test_create_contains_synthesis_keywords(self) -> None:
        supplement = BLOOM_PROMPT_SUPPLEMENTS["create"]
        assert "synthesis" in supplement.lower() or "design" in supplement.lower()
        assert "construct" in supplement.lower() or "new" in supplement.lower()

    def test_supplements_are_non_empty_strings(self) -> None:
        for level, supplement in BLOOM_PROMPT_SUPPLEMENTS.items():
            assert isinstance(supplement, str), f"{level} supplement is not a string"
            assert len(supplement.strip()) > 50, f"{level} supplement is too short"


class TestTargetSelectionPrompt:
    """Tests for Phase 1 target selection prompt."""

    def test_target_selection_prompt_contains_angle_types(self) -> None:
        for angle in ("MECHANISMS", "CONNECTIONS", "APPLICATIONS", "EDGE CASES", "CONTRASTS", "CONSEQUENCES"):
            assert angle in TARGET_SELECTION_PROMPT

    def test_target_selection_prompt_rejects_definitions(self) -> None:
        assert "DO NOT identify definitions" in TARGET_SELECTION_PROMPT

    def test_build_target_selection_prompt_includes_section(self) -> None:
        result = build_target_selection_prompt(
            section_title="Bond Pricing",
            section_text="Content about bond pricing mechanisms.",
            chapter_title="Fixed Income",
        )
        assert "Bond Pricing" in result
        assert "Fixed Income" in result
        assert "bond pricing mechanisms" in result

    def test_build_target_selection_prompt_includes_concepts(self) -> None:
        from src.transformation.analysis_types import ConceptEntry

        concepts = [
            ConceptEntry(
                name="Duration",
                definition="Price sensitivity.",
                concept_type="formula",
                section_title="Sec",
            ),
        ]
        result = build_target_selection_prompt(
            section_title="Sec",
            section_text="Content.",
            chapter_title="Ch",
            section_concepts=concepts,
        )
        assert "Duration" in result

    def test_build_target_selection_prompt_works_without_concepts(self) -> None:
        result = build_target_selection_prompt(
            section_title="Sec",
            section_text="Content.",
            chapter_title="Ch",
        )
        assert "Concepts in this section" not in result


class TestReinforcementTargetsInSectionPrompt:
    """Tests for reinforcement targets block in build_section_prompt."""

    def test_includes_targets_when_provided(self) -> None:
        from src.transformation.types import ReinforcementTarget

        targets = [
            ReinforcementTarget(
                concept_name="Diversification",
                target_insight="Correlations spike in crises",
                angle="edge_case",
                bloom_level="analyze",
                suggested_element_type="quiz",
            ),
            ReinforcementTarget(
                concept_name="Sharpe ratio",
                target_insight="Penalizes upside volatility equally",
                angle="mechanism",
                bloom_level="evaluate",
                suggested_element_type="interactive_essay",
            ),
            ReinforcementTarget(
                concept_name="Duration",
                target_insight="Derivative of price w.r.t. yield",
                angle="connection",
                bloom_level="apply",
                suggested_element_type="fill_in_the_blank",
            ),
        ]
        result = build_section_prompt(
            section_title="Sec",
            section_text="Content " * 50,
            chapter_title="Ch",
            image_count=0,
            table_count=0,
            reinforcement_targets=targets,
        )
        assert "Reinforcement Targets" in result
        assert "Correlations spike in crises" in result
        assert "edge_case" in result
        assert "MUST test" in result
        assert "MUST map to one of these targets" in result

    def test_no_targets_block_when_not_provided(self) -> None:
        result = build_section_prompt(
            section_title="Sec",
            section_text="Content.",
            chapter_title="Ch",
            image_count=0,
            table_count=0,
        )
        assert "Reinforcement Targets" not in result


class TestFocusConceptsPromptBlock:
    """Tests for the CONCEPT FOCUS block in build_section_prompt."""

    def test_includes_focus_block_when_provided(self) -> None:
        result = build_section_prompt(
            section_title="Basis and Span",
            section_text="Content about vector spaces...",
            chapter_title="Linear Algebra",
            image_count=0,
            table_count=0,
            focus_concepts=["basis", "span"],
        )
        assert "CONCEPT FOCUS" in result
        assert "**basis**" in result
        assert "**span**" in result
        assert "ONLY" in result
        assert "5-10 minutes" in result
        # Multi-concept case: should still request exactly 1 slide
        assert "SINGLE slide" in result

    def test_focus_block_single_concept(self) -> None:
        result = build_section_prompt(
            section_title="Basis",
            section_text="Content about basis...",
            chapter_title="Linear Algebra",
            image_count=0,
            table_count=0,
            focus_concepts=["basis"],
        )
        assert "CONCEPT FOCUS" in result
        assert "**basis**" in result
        assert "1 slide covering this concept" in result
        # Single concept should NOT mention "tightly coupled"
        assert "tightly coupled" not in result

    def test_no_focus_block_when_none(self) -> None:
        result = build_section_prompt(
            section_title="Sec",
            section_text="Text",
            chapter_title="Ch",
            image_count=0,
            table_count=0,
            focus_concepts=None,
        )
        assert "CONCEPT FOCUS" not in result

    def test_no_focus_block_when_empty_list(self) -> None:
        result = build_section_prompt(
            section_title="Sec",
            section_text="Text",
            chapter_title="Ch",
            image_count=0,
            table_count=0,
            focus_concepts=[],
        )
        assert "CONCEPT FOCUS" not in result

    def test_focus_block_coexists_with_other_blocks(self) -> None:
        result = build_section_prompt(
            section_title="Sec",
            section_text="Content " * 50,
            chapter_title="Ch",
            image_count=0,
            table_count=0,
            template="worked_example",
            learning_objectives=["Understand X"],
            bloom_target="apply",
            focus_concepts=["concept_a", "concept_b"],
        )
        assert "CONCEPT FOCUS" in result
        assert "WORKED EXAMPLE" in result
        assert "Understand X" in result
        assert "**concept_a**" in result


class TestExerciseAssignmentBlock:
    """Tests for _build_exercise_assignment_block()."""

    def test_empty_when_none(self) -> None:
        from src.transformation.content_designer_prompts import _build_exercise_assignment_block
        assert _build_exercise_assignment_block(None) == ""
        assert _build_exercise_assignment_block([]) == ""

    def test_includes_all_assigned_types(self) -> None:
        from src.transformation.content_designer_prompts import _build_exercise_assignment_block
        types = ["matching", "fill_in_the_blank", "ordering", "analogy"]
        result = _build_exercise_assignment_block(types)
        assert "MANDATORY" in result
        for t in types:
            assert t in result

    def test_difficulty_labels_assigned(self) -> None:
        from src.transformation.content_designer_prompts import _build_exercise_assignment_block
        types = ["matching", "quiz", "categorization", "analogy"]
        result = _build_exercise_assignment_block(types)
        assert "easy)" in result
        assert "medium)" in result

    def test_exercise_types_in_build_section_prompt(self) -> None:
        from src.transformation.content_designer_prompts import build_section_prompt
        result = build_section_prompt(
            section_title="Sec",
            section_text="Content " * 50,
            chapter_title="Ch",
            image_count=0,
            table_count=0,
            exercise_types=["matching", "ordering", "quiz", "categorization"],
        )
        assert "MANDATORY" in result
        assert "matching" in result
        assert "ordering" in result
