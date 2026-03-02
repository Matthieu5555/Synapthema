"""Tests for the HTML rendering module — deep interface tests.

Tests _render_course() with fixture data (no LLM, no PDF) and individual
helper functions like _markdown_to_html and _render_fill_blanks.
"""

import json
from pathlib import Path

import pytest

from src.rendering.html_generator import (
    RENDERERS,
    _deduplicate_math,
    _derive_course_title,
    _element_id,
    _fitb_interactive_answer_indices,
    _generate_chapter_colors,
    _load_course_meta,
    _markdown_to_html,
    _markdown_to_html_inline,
    _prepare_graph_data,
    _render_element,
    _render_fill_blanks,
    _render_fitb_statement,
    _sanitize_html,
    _to_katex_delimiters,
    _write_course_meta,
    _render_course,
)
from src.transformation.analysis_types import (
    ChapterAnalysis,
    ConceptEdge,
    ConceptEntry,
    ConceptGraph,
    ResolvedConcept,
)
from src.transformation.types import (
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
    FillInTheBlank,
    FillInBlankElement,
    MatchingExercise,
    MatchingElement,
    InteractiveEssay,
    InteractiveEssayElement,
    OrderingElement,
    OrderingExercise,
    Quiz,
    QuizElement,
    QuizQuestion,
    SectionIntro,
    SectionIntroElement,
    SelfExplain,
    Slide,
    SlideElement,
    TrainingModule,
    TrainingSection,
    WorkedExample,
    WorkedExampleChallengeOption,
    WorkedExampleElement,
    WorkedExampleStep,
)


# ── _markdown_to_html tests ──────────────────────────────────────────────────


class TestMarkdownToHtml:
    """Tests for the markdown-to-HTML converter."""

    def test_bold(self) -> None:
        assert "<strong>bold</strong>" in _markdown_to_html("**bold**")

    def test_italic(self) -> None:
        assert "<em>italic</em>" in _markdown_to_html("*italic*")

    def test_inline_code(self) -> None:
        assert "<code>x = 1</code>" in _markdown_to_html("`x = 1`")

    def test_fenced_code_block(self) -> None:
        text = "```python\nprint('hi')\n```"
        result = _markdown_to_html(text)
        assert "<code" in result
        assert "print" in result

    def test_bullet_list(self) -> None:
        text = "- item one\n- item two"
        result = _markdown_to_html(text)
        assert "<li>" in result
        assert "item one" in result
        assert "item two" in result

    def test_numbered_list(self) -> None:
        text = "1. first\n2. second"
        result = _markdown_to_html(text)
        assert "<li>" in result

    def test_preserves_latex_inline(self) -> None:
        result = _markdown_to_html("The formula $E = mc^2$ is famous.")
        assert r"\(E = mc^2\)" in result

    def test_preserves_latex_display(self) -> None:
        result = _markdown_to_html("$$\\frac{a}{b}$$")
        assert r"\[\frac{a}{b}\]" in result

    def test_latex_with_underscores(self) -> None:
        """LaTeX subscripts like $x_i$ should not be interpreted as emphasis."""
        result = _markdown_to_html("Variable $x_i$ is indexed.")
        assert r"\(x_i\)" in result

    def test_paragraphs(self) -> None:
        result = _markdown_to_html("Paragraph one.\n\nParagraph two.")
        assert result.count("<p>") == 2


# ── _render_fill_blanks tests ────────────────────────────────────────────────


class TestRenderFillBlanks:
    """Tests for fill-in-the-blank placeholder replacement."""

    def test_single_blank(self) -> None:
        result = _render_fill_blanks("The _____ is blue.")
        assert '<input type="text"' in result
        assert 'data-blank-index="0"' in result
        assert "_____" not in result

    def test_multiple_blanks(self) -> None:
        result = _render_fill_blanks("_____ and _____")
        assert 'data-blank-index="0"' in result
        assert 'data-blank-index="1"' in result

    def test_no_blanks(self) -> None:
        text = "No blanks here."
        assert _render_fill_blanks(text) == text

    def test_varying_underscore_lengths(self) -> None:
        result = _render_fill_blanks("___ and ________")
        assert result.count("<input") == 2


# ── _derive_course_title tests ───────────────────────────────────────────────


class TestDeriveCourseTitle:
    """Tests for course title derivation from module titles."""

    def test_empty_modules(self) -> None:
        assert _derive_course_title([]) == "Interactive Training Course"

    def test_single_module(self) -> None:
        m = TrainingModule(chapter_number=1, title="Calculus Basics", sections=[])
        assert _derive_course_title([m]) == "Calculus Basics"

    def test_common_prefix(self) -> None:
        modules = [
            TrainingModule(chapter_number=1, title="Finance: Bonds", sections=[]),
            TrainingModule(chapter_number=2, title="Finance: Equity", sections=[]),
        ]
        result = _derive_course_title(modules)
        assert "Finance" in result

    def test_no_common_prefix(self) -> None:
        modules = [
            TrainingModule(chapter_number=1, title="Algebra", sections=[]),
            TrainingModule(chapter_number=2, title="Biology", sections=[]),
        ]
        assert _derive_course_title(modules) == "Interactive Training Course"


# ── _render_course integration test ───────────────────────────────────────────


class TestRenderCourse:
    """Integration test for the full _render_course entry point."""

    def _make_module(self) -> TrainingModule:
        """Build a TrainingModule with one of each element type."""
        elements = [
            SectionIntroElement(
                bloom_level="understand",
                section_intro=SectionIntro(
                    title="Why This Matters",
                    content="In this section you will learn the fundamentals of arithmetic and see why they matter in everyday life.",
                ),
            ),
            SlideElement(
                bloom_level="understand",
                slide=Slide(
                    title="Intro Slide",
                    content="**Bold** and *italic* text.",
                ),
            ),
            QuizElement(
                bloom_level="apply",
                quiz=Quiz(
                    title="Check",
                    questions=[
                        QuizQuestion(
                            question="What is 2+2?",
                            options=["3", "4", "5"],
                            correct_index=1,
                            explanation="Basic arithmetic.",
                        )
                    ],
                ),
            ),
            FillInBlankElement(
                bloom_level="apply",
                fill_in_the_blank=FillInTheBlank(
                    statement="The _____ theorem.",
                    answers=["fundamental"],
                    hint="Think about calculus.",
                ),
            ),
            MatchingElement(
                bloom_level="apply",
                matching=MatchingExercise(
                    title="Match terms",
                    left_items=["A", "B"],
                    right_items=["1", "2"],
                ),
            ),
            OrderingElement(
                bloom_level="apply",
                ordering=OrderingExercise(
                    title="Order Steps",
                    instruction="Arrange in order",
                    items=["First", "Second", "Third"],
                    explanation="Natural ordering.",
                    hint="Think about what comes first.",
                ),
            ),
            CategorizationElement(
                bloom_level="analyze",
                categorization=CategorizationExercise(
                    title="Classify Items",
                    instruction="Sort into categories",
                    categories=[
                        CategoryBucket(name="Fruits", items=["Apple", "Banana"]),
                        CategoryBucket(name="Vegetables", items=["Carrot"]),
                    ],
                    explanation="Fruits grow on trees, vegetables in soil.",
                    hint="Think about where they grow.",
                ),
            ),
            ErrorDetectionElement(
                bloom_level="evaluate",
                error_detection=ErrorDetectionExercise(
                    title="Spot Errors",
                    instruction="Find the mistake",
                    items=[
                        ErrorItem(
                            statement="The sun revolves around the Earth.",
                            error_explanation="The Earth revolves around the Sun.",
                            corrected_statement="The Earth revolves around the Sun.",
                        ),
                    ],
                    context="Basic astronomy facts.",
                ),
            ),
            AnalogyElement(
                bloom_level="analyze",
                analogy=AnalogyExercise(
                    title="Analogy Challenge",
                    items=[
                        AnalogyItem(
                            stem="Hot is to cold as up is to ___",
                            answer="down",
                            distractors=["left", "right"],
                            explanation="These are opposite pairs.",
                        ),
                    ],
                ),
            ),
            WorkedExampleElement(
                bloom_level="apply",
                worked_example=WorkedExample(
                    title="Solve for X",
                    problem_statement="Given 2x + 4 = 10, find x.",
                    challenge_question="What is x?",
                    challenge_options=[
                        WorkedExampleChallengeOption(text="2"),
                        WorkedExampleChallengeOption(text="3"),
                        WorkedExampleChallengeOption(text="4"),
                    ],
                    challenge_correct_index=1,
                    steps=[
                        WorkedExampleStep(title="Subtract 4", content="2x = 6", why="Isolate the variable term"),
                        WorkedExampleStep(title="Divide by 2", content="x = 3", why="Solve for x"),
                    ],
                    final_answer="x = 3",
                ),
            ),
            FlashcardElement(
                bloom_level="remember",
                flashcard=Flashcard(front="Term", back="Definition"),
            ),
            # Static interactive essay (single prompt, no tutor — replaces old self_explain)
            InteractiveEssayElement(
                bloom_level="evaluate",
                interactive_essay=InteractiveEssay(
                    title="Quick Reflection",
                    concepts_tested=["diversification"],
                    prompts=[
                        SelfExplain(
                            prompt="Explain diversification in your own words.",
                            key_points=["reduces risk", "uncorrelated assets"],
                            example_response="Diversification works by combining assets.",
                            minimum_words=30,
                        ),
                    ],
                    passing_threshold=0.7,
                    tutor_system_prompt="",
                ),
            ),
            # Dynamic interactive essay (multi-prompt + tutor)
            InteractiveEssayElement(
                bloom_level="evaluate",
                interactive_essay=InteractiveEssay(
                    title="Chapter Checkpoint",
                    concepts_tested=["bonds", "yields"],
                    prompts=[
                        SelfExplain(
                            prompt="Explain bond pricing.",
                            key_points=["present value", "coupon rate"],
                            example_response="Bond price equals the present value of cash flows.",
                            minimum_words=20,
                        ),
                    ],
                    passing_threshold=0.7,
                    tutor_system_prompt="You are a Socratic tutor.",
                ),
            ),
        ]
        section = TrainingSection(title="Test Section", elements=elements)
        return TrainingModule(
            chapter_number=1,
            title="Test Chapter",
            sections=[section],
        )

    def test_renders_chapter_and_index(self, tmp_path: Path) -> None:
        module = self._make_module()
        output_dir = tmp_path / "output"

        index_path = _render_course(
            modules=[module],
            output_dir=output_dir,
            embed_images=False,
        )

        assert index_path.exists()
        assert (output_dir / "chapter_01.html").exists()

        # Index should link to the chapter
        index_html = index_path.read_text(encoding="utf-8")
        assert "chapter_01.html" in index_html
        assert "Test Chapter" in index_html

    def test_chapter_contains_all_element_types(self, tmp_path: Path) -> None:
        module = self._make_module()
        output_dir = tmp_path / "output"

        _render_course(modules=[module], output_dir=output_dir, embed_images=False)

        chapter_html = (output_dir / "chapter_01.html").read_text(encoding="utf-8")

        # Slide content
        assert "Intro Slide" in chapter_html
        assert "<strong>" in chapter_html  # markdown was converted

        # Quiz
        assert "What is 2+2?" in chapter_html
        assert "Basic arithmetic." in chapter_html

        # Flashcard
        assert "Term" in chapter_html
        assert "Definition" in chapter_html

        # Fill-in-the-blank
        assert '<input type="text"' in chapter_html
        assert "fundamental" in chapter_html  # in answers JSON

        # Matching
        assert "Match terms" in chapter_html

        # Ordering
        assert "ordering-container" in chapter_html
        assert "Order Steps" in chapter_html

        # Categorization
        assert "categorization-container" in chapter_html
        assert "Classify Items" in chapter_html

        # Error detection
        assert "error-detection-container" in chapter_html
        assert "Spot Errors" in chapter_html

        # Analogy
        assert "analogy-container" in chapter_html
        assert "Analogy Challenge" in chapter_html

        # Worked example
        assert "worked-example-container" in chapter_html
        assert "Solve for X" in chapter_html

    def test_chapter_contains_section_intro(self, tmp_path: Path) -> None:
        module = self._make_module()
        output_dir = tmp_path / "output"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)
        chapter_html = (output_dir / "chapter_01.html").read_text(encoding="utf-8")

        assert "section-intro-container" in chapter_html
        assert "Why This Matters" in chapter_html

    def test_chapter_contains_static_essay(self, tmp_path: Path) -> None:
        module = self._make_module()
        output_dir = tmp_path / "output"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)
        chapter_html = (output_dir / "chapter_01.html").read_text(encoding="utf-8")

        assert "essay-container" in chapter_html
        assert "Explain diversification" in chapter_html

    def test_chapter_contains_interactive_essay(self, tmp_path: Path) -> None:
        module = self._make_module()
        output_dir = tmp_path / "output"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)
        chapter_html = (output_dir / "chapter_01.html").read_text(encoding="utf-8")

        assert "essay-container" in chapter_html
        assert "Chapter Checkpoint" in chapter_html
        assert "concept-tag" in chapter_html

    def test_chapter_contains_settings_modal(self, tmp_path: Path) -> None:
        module = self._make_module()
        output_dir = tmp_path / "output"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)
        chapter_html = (output_dir / "chapter_01.html").read_text(encoding="utf-8")

        assert "settingsModal" in chapter_html
        assert "tutorApiKey" in chapter_html

    def test_help_modal_present_on_all_pages(self, tmp_path: Path) -> None:
        """Help modal should appear on chapter, index, review, and mixed_review."""
        module = self._make_module()
        output_dir = tmp_path / "output"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)

        for page in ("chapter_01.html", "index.html", "review.html", "mixed_review.html"):
            html = (output_dir / page).read_text(encoding="utf-8")
            assert "helpModal" in html, f"helpModal missing from {page}"
            assert "help-btn" in html, f"help-btn missing from {page}"
            assert "How This Course Works" in html, f"Help title missing from {page}"

    def test_multi_chapter_navigation(self, tmp_path: Path) -> None:
        """Chapters should have prev/next links to adjacent chapters."""
        m1 = self._make_module()
        m2 = TrainingModule(
            chapter_number=2,
            title="Chapter Two",
            sections=[
                TrainingSection(
                    title="Section Two",
                    elements=[
                        SlideElement(
                            bloom_level="understand",
                            slide=Slide(title="Slide Two", content="Content two."),
                        ),
                    ],
                ),
            ],
        )
        output_dir = tmp_path / "output"
        _render_course(modules=[m1, m2], output_dir=output_dir, embed_images=False)

        ch1 = (output_dir / "chapter_01.html").read_text(encoding="utf-8")
        ch2 = (output_dir / "chapter_02.html").read_text(encoding="utf-8")

        # Chapter 1 should link forward to chapter 2 in the header
        assert 'href="chapter_02.html" class="chapter-nav-btn"' in ch1
        # Chapter 1 should NOT have a prev link (it's the first)
        assert 'href="chapter_01.html" class="chapter-nav-btn"' not in ch1

        # Chapter 2 should link back to chapter 1 in the header
        assert 'href="chapter_01.html" class="chapter-nav-btn"' in ch2
        # Chapter 2 should NOT have a next link (it's the last)
        assert 'href="chapter_02.html" class="chapter-nav-btn"' not in ch2

    def test_duplicate_chapter_numbers_produce_unique_files(self, tmp_path: Path) -> None:
        """Modules sharing the same source chapter_number must get unique files."""
        modules = [
            TrainingModule(
                chapter_number=2,
                title="Module A from Ch2",
                sections=[
                    TrainingSection(
                        title="Section A",
                        elements=[
                            SlideElement(
                                bloom_level="understand",
                                slide=Slide(title="Slide A", content="Content A."),
                            ),
                        ],
                    ),
                ],
            ),
            TrainingModule(
                chapter_number=2,
                title="Module B from Ch2",
                sections=[
                    TrainingSection(
                        title="Section B",
                        elements=[
                            SlideElement(
                                bloom_level="understand",
                                slide=Slide(title="Slide B", content="Content B."),
                            ),
                        ],
                    ),
                ],
            ),
            TrainingModule(
                chapter_number=4,
                title="Module C from Ch4",
                sections=[
                    TrainingSection(
                        title="Section C",
                        elements=[
                            SlideElement(
                                bloom_level="understand",
                                slide=Slide(title="Slide C", content="Content C."),
                            ),
                        ],
                    ),
                ],
            ),
        ]
        output_dir = tmp_path / "output"
        _render_course(modules=modules, output_dir=output_dir, embed_images=False)

        # Should produce 3 unique files named by sequential index, not source chapter
        assert (output_dir / "chapter_01.html").exists()
        assert (output_dir / "chapter_02.html").exists()
        assert (output_dir / "chapter_03.html").exists()

        ch1 = (output_dir / "chapter_01.html").read_text(encoding="utf-8")
        ch2 = (output_dir / "chapter_02.html").read_text(encoding="utf-8")
        ch3 = (output_dir / "chapter_03.html").read_text(encoding="utf-8")

        # Each file should contain its own module's content, not be overwritten
        assert "Slide A" in ch1
        assert "Slide B" in ch2
        assert "Slide C" in ch3

        # Element IDs should be unique across modules
        import re
        ids1 = set(re.findall(r'data-element-id="([^"]+)"', ch1))
        ids2 = set(re.findall(r'data-element-id="([^"]+)"', ch2))
        ids3 = set(re.findall(r'data-element-id="([^"]+)"', ch3))
        assert ids1.isdisjoint(ids2), "Modules with same chapter_number must have unique element IDs"
        assert ids2.isdisjoint(ids3)

        # Navigation: ch1 → ch2 → ch3
        assert 'href="chapter_02.html" class="chapter-nav-btn"' in ch1
        assert 'href="chapter_01.html" class="chapter-nav-btn"' in ch2
        assert 'href="chapter_03.html" class="chapter-nav-btn"' in ch2
        assert 'href="chapter_02.html" class="chapter-nav-btn"' in ch3

        # Index should list all 3 modules with correct links
        index_html = (output_dir / "index.html").read_text(encoding="utf-8")
        assert "chapter_01.html" in index_html
        assert "chapter_02.html" in index_html
        assert "chapter_03.html" in index_html

    def test_theme_init_script_present(self, tmp_path: Path) -> None:
        """Both chapter and index HTML should contain the theme init script."""
        module = self._make_module()
        output_dir = tmp_path / "output"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)

        chapter_html = (output_dir / "chapter_01.html").read_text(encoding="utf-8")
        index_html = (output_dir / "index.html").read_text(encoding="utf-8")

        # Both files should have the localStorage theme init in <head>
        assert "lxp_output_theme" in chapter_html
        assert "lxp_output_theme" in index_html
        assert "data-theme" in chapter_html
        assert "data-theme" in index_html

    def test_course_slug_in_output(self, tmp_path: Path) -> None:
        """The course slug should be embedded in HTML for localStorage keying."""
        module = self._make_module()
        output_dir = tmp_path / "my-course"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)

        chapter_html = (output_dir / "chapter_01.html").read_text(encoding="utf-8")
        index_html = (output_dir / "index.html").read_text(encoding="utf-8")

        assert "lxp_my-course_" in chapter_html
        assert "lxp_my-course_" in index_html

    def test_course_header_and_breadcrumbs(self, tmp_path: Path) -> None:
        """Chapter pages should have a course header with breadcrumbs."""
        module = self._make_module()
        output_dir = tmp_path / "output"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)

        chapter_html = (output_dir / "chapter_01.html").read_text(encoding="utf-8")

        assert "course-header" in chapter_html
        assert "breadcrumb-home" in chapter_html
        assert 'href="index.html"' in chapter_html
        assert "theme-toggle" in chapter_html

    def test_index_has_theme_toggle(self, tmp_path: Path) -> None:
        """Index page should have a theme toggle button."""
        module = self._make_module()
        output_dir = tmp_path / "output"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)

        index_html = (output_dir / "index.html").read_text(encoding="utf-8")

        assert "theme-toggle" in index_html
        assert "__lxpToggleTheme" in index_html

    def test_progress_tracking_script_present(self, tmp_path: Path) -> None:
        """Chapter pages should contain progress tracking JavaScript."""
        module = self._make_module()
        output_dir = tmp_path / "output"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)

        chapter_html = (output_dir / "chapter_01.html").read_text(encoding="utf-8")

        assert "updateProgress" in chapter_html
        assert "loadProgress" in chapter_html
        assert "PROGRESS_KEY" in chapter_html
        assert "lxp_" in chapter_html

    def test_autoescape_prevents_xss(self, tmp_path: Path) -> None:
        """User-controlled strings with HTML should be escaped in the output."""
        xss_title = '<img src=x onerror="alert(1)">'
        section = TrainingSection(
            title="Safe Section",
            elements=[
                SlideElement(
                    bloom_level="understand",
                    slide=Slide(
                        title=xss_title,
                        content="Safe content.",
                    ),
                ),
            ],
        )
        module = TrainingModule(
            chapter_number=1,
            title="Safe Title",
            sections=[section],
        )
        output_dir = tmp_path / "output"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)

        chapter_html = (output_dir / "chapter_01.html").read_text(encoding="utf-8")

        # The raw <img> tag from the slide title should be escaped
        assert xss_title not in chapter_html
        assert 'onerror="alert' not in chapter_html
        # The escaped version should be present instead
        assert "&lt;img" in chapter_html

    def test_elements_have_deterministic_ids(self, tmp_path: Path) -> None:
        """Each element should have a data-element-id attribute in HTML."""
        module = self._make_module()
        output_dir = tmp_path / "output"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)

        chapter_html = (output_dir / "chapter_01.html").read_text(encoding="utf-8")

        assert 'data-element-id="card_ch01_s00_e' in chapter_html

    def test_element_ids_stable_across_rerenders(self, tmp_path: Path) -> None:
        """Element IDs should be identical between two separate renders."""
        module = self._make_module()

        dir1 = tmp_path / "render1"
        dir2 = tmp_path / "render2"
        _render_course(modules=[module], output_dir=dir1, embed_images=False)
        _render_course(modules=[module], output_dir=dir2, embed_images=False)

        html1 = (dir1 / "chapter_01.html").read_text(encoding="utf-8")
        html2 = (dir2 / "chapter_01.html").read_text(encoding="utf-8")

        # Extract all element IDs from both renders
        import re
        ids1 = re.findall(r'data-element-id="([^"]+)"', html1)
        ids2 = re.findall(r'data-element-id="([^"]+)"', html2)
        assert ids1 == ids2
        assert len(ids1) > 0

    def test_elements_have_concepts_data(self, tmp_path: Path) -> None:
        """Elements should have data-concepts attribute (empty when no analysis)."""
        module = self._make_module()
        output_dir = tmp_path / "output"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)

        chapter_html = (output_dir / "chapter_01.html").read_text(encoding="utf-8")
        assert 'data-concepts=' in chapter_html

    def test_concepts_populated_from_chapter_analysis(self, tmp_path: Path) -> None:
        """When chapter_analyses are passed, concepts_tested should appear in HTML."""
        module = self._make_module()
        analysis = ChapterAnalysis(
            chapter_number=1,
            chapter_title="Test Chapter",
            concepts=[
                ConceptEntry(
                    name="bond",
                    definition="A debt instrument",
                    concept_type="definition",
                    section_title="Test Section",
                ),
                ConceptEntry(
                    name="yield",
                    definition="Return on investment",
                    concept_type="definition",
                    section_title="Test Section",
                ),
            ],
        )
        output_dir = tmp_path / "output"
        _render_course(
            modules=[module], output_dir=output_dir, embed_images=False,
            chapter_analyses=[analysis],
        )

        chapter_html = (output_dir / "chapter_01.html").read_text(encoding="utf-8")
        assert '"bond"' in chapter_html
        assert '"yield"' in chapter_html

    def test_chapter_has_review_button(self, tmp_path: Path) -> None:
        """Chapter pages should contain 'Add to Review' buttons."""
        module = self._make_module()
        output_dir = tmp_path / "output"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)

        chapter_html = (output_dir / "chapter_01.html").read_text(encoding="utf-8")
        assert "add-review-btn" in chapter_html
        assert "addToReview" in chapter_html

    def test_chapter_has_review_nav_badge(self, tmp_path: Path) -> None:
        """Chapter pages should have a review nav badge in the header."""
        module = self._make_module()
        output_dir = tmp_path / "output"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)

        chapter_html = (output_dir / "chapter_01.html").read_text(encoding="utf-8")
        assert "reviewNavBadge" in chapter_html
        assert "review-nav-badge" in chapter_html

    def test_chapter_has_learner_model(self, tmp_path: Path) -> None:
        """Chapter pages should include the learner model JavaScript."""
        module = self._make_module()
        output_dir = tmp_path / "output"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)

        chapter_html = (output_dir / "chapter_01.html").read_text(encoding="utf-8")
        assert "__lxpCreateLearnerModel" in chapter_html
        assert "recordAnswer" in chapter_html
        assert "classifyMastery" in chapter_html

    def test_index_has_learner_model(self, tmp_path: Path) -> None:
        """Index page should include the learner model JavaScript."""
        module = self._make_module()
        output_dir = tmp_path / "output"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)

        index_html = (output_dir / "index.html").read_text(encoding="utf-8")
        assert "__lxpCreateLearnerModel" in index_html
        assert "masteryOverview" in index_html

    def test_index_no_graph_without_concept_graph(self, tmp_path: Path) -> None:
        """Index should not have concept graph section without graph data."""
        module = self._make_module()
        output_dir = tmp_path / "output"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)

        index_html = (output_dir / "index.html").read_text(encoding="utf-8")
        # The graph section and vis-network script should not be rendered
        assert 'id="concept-graph"' not in index_html
        assert "vis-network.min.js" not in index_html

    def test_index_has_graph_with_concept_graph(self, tmp_path: Path) -> None:
        """Index should render concept graph when graph data is provided."""
        module = self._make_module()
        graph = ConceptGraph(
            concepts=[
                ResolvedConcept(
                    canonical_name="bond",
                    definition="A debt instrument",
                    first_introduced_chapter=1,
                    mentioned_in_chapters=[1],
                ),
                ResolvedConcept(
                    canonical_name="yield",
                    definition="Return on investment",
                    first_introduced_chapter=1,
                    mentioned_in_chapters=[1],
                ),
            ],
            edges=[
                ConceptEdge(source="yield", target="bond", relationship="requires"),
            ],
        )
        output_dir = tmp_path / "output"
        _render_course(
            modules=[module], output_dir=output_dir, embed_images=False,
            concept_graph=graph,
        )

        index_html = (output_dir / "index.html").read_text(encoding="utf-8")
        assert "concept-graph" in index_html
        assert "vis-network" in index_html
        assert '"bond"' in index_html
        assert '"yield"' in index_html

    def test_review_pages_have_learner_model(self, tmp_path: Path) -> None:
        """Review and mixed review pages should include learner model."""
        module = self._make_module()
        output_dir = tmp_path / "output"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)

        review_html = (output_dir / "review.html").read_text(encoding="utf-8")
        mixed_html = (output_dir / "mixed_review.html").read_text(encoding="utf-8")

        assert "__lxpCreateLearnerModel" in review_html
        assert "__lxpCreateLearnerModel" in mixed_html
        assert "learnerModel" in review_html
        assert "learnerModel" in mixed_html


# ── _element_id tests ───────────────────────────────────────────────────────


class TestElementId:
    """Tests for deterministic element ID generation."""

    def test_format(self) -> None:
        assert _element_id(1, 0, 0) == "card_ch01_s00_e00"
        assert _element_id(5, 2, 10) == "card_ch05_s02_e10"

    def test_deterministic(self) -> None:
        assert _element_id(3, 1, 7) == _element_id(3, 1, 7)

    def test_unique_across_sections(self) -> None:
        assert _element_id(1, 0, 0) != _element_id(1, 1, 0)

    def test_unique_across_chapters(self) -> None:
        assert _element_id(1, 0, 0) != _element_id(2, 0, 0)


# ── _prepare_graph_data tests ───────────────────────────────────────────────


class TestPrepareGraphData:
    """Tests for concept graph data preparation for vis-network."""

    def test_empty_graph(self) -> None:
        graph = ConceptGraph()
        modules = [TrainingModule(chapter_number=1, title="Ch1", sections=[])]
        result = _prepare_graph_data(graph, modules)
        assert result == {"nodes": [], "edges": []}

    def test_nodes_include_concept_data(self) -> None:
        graph = ConceptGraph(
            concepts=[
                ResolvedConcept(
                    canonical_name="bond",
                    definition="A debt instrument",
                    first_introduced_chapter=1,
                    mentioned_in_chapters=[1, 2],
                ),
            ],
            edges=[],
        )
        modules = [
            TrainingModule(chapter_number=1, title="Ch1", sections=[]),
            TrainingModule(chapter_number=2, title="Ch2", sections=[]),
        ]
        result = _prepare_graph_data(graph, modules)

        assert len(result["nodes"]) == 1
        node = result["nodes"][0]
        assert node["id"] == "bond"
        assert node["label"] == "bond"
        assert node["title"] == "<b>bond</b><br>A debt instrument"
        assert node["group"] == 1
        # Size should scale with mention count
        assert node["size"] == 10 + 3 * 2  # 2 chapters

    def test_edges_include_relationship(self) -> None:
        graph = ConceptGraph(
            concepts=[
                ResolvedConcept(
                    canonical_name="bond", definition="debt",
                    first_introduced_chapter=1, mentioned_in_chapters=[1],
                ),
                ResolvedConcept(
                    canonical_name="yield", definition="return",
                    first_introduced_chapter=1, mentioned_in_chapters=[1],
                ),
            ],
            edges=[
                ConceptEdge(source="yield", target="bond", relationship="requires"),
            ],
        )
        modules = [TrainingModule(chapter_number=1, title="Ch1", sections=[])]
        result = _prepare_graph_data(graph, modules)

        assert len(result["edges"]) == 1
        edge = result["edges"][0]
        assert edge["from"] == "yield"
        assert edge["to"] == "bond"
        assert edge["arrows"] == "to"


# ── _generate_chapter_colors tests ──────────────────────────────────────────


class TestGenerateChapterColors:
    """Tests for chapter color palette generation."""

    def test_single_chapter(self) -> None:
        colors = _generate_chapter_colors(1)
        assert 1 in colors
        assert colors[1].startswith("#")

    def test_multiple_chapters(self) -> None:
        colors = _generate_chapter_colors(3)
        assert len(colors) == 3
        # Colors should be distinct
        assert colors[1] != colors[2]
        assert colors[2] != colors[3]

    def test_zero_chapters(self) -> None:
        colors = _generate_chapter_colors(0)
        assert colors == {}


# ── Course metadata override tests ──────────────────────────────────────────


class TestCourseMetaJson:
    """Tests for course_meta.json read/write/override."""

    def test_write_creates_file(self, tmp_path: Path) -> None:
        _write_course_meta(tmp_path, "Title", "Summary", "Journey", "Subtitle")
        meta_path = tmp_path / "course_meta.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert data["course_title"] == "Title"
        assert data["course_summary"] == "Summary"
        assert data["learner_journey"] == "Journey"
        assert data["subtitle"] == "Subtitle"

    def test_write_does_not_overwrite(self, tmp_path: Path) -> None:
        meta_path = tmp_path / "course_meta.json"
        meta_path.write_text('{"course_title":"Custom"}', encoding="utf-8")
        _write_course_meta(tmp_path, "Original", "", "", "Sub")
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert data["course_title"] == "Custom"

    def test_load_returns_none_when_missing(self, tmp_path: Path) -> None:
        assert _load_course_meta(tmp_path) is None

    def test_load_returns_dict(self, tmp_path: Path) -> None:
        (tmp_path / "course_meta.json").write_text(
            '{"course_title":"My Course"}', encoding="utf-8"
        )
        result = _load_course_meta(tmp_path)
        assert result == {"course_title": "My Course"}

    def test_load_ignores_invalid_json(self, tmp_path: Path) -> None:
        (tmp_path / "course_meta.json").write_text("not json", encoding="utf-8")
        assert _load_course_meta(tmp_path) is None

    def test__render_course_creates_meta_json(self, tmp_path: Path) -> None:
        module = TrainingModule(
            chapter_number=1,
            title="Ch1",
            sections=[
                TrainingSection(
                    title="S1",
                    elements=[
                        SlideElement(
                            bloom_level="understand",
                            slide=Slide(title="Slide", content="Hello"),
                        )
                    ],
                )
            ],
        )
        output_dir = tmp_path / "output"
        _render_course(
            modules=[module],
            output_dir=output_dir,
            embed_images=False,
            course_title="My Title",
            course_summary="My Summary",
        )
        meta_path = output_dir / "course_meta.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert data["course_title"] == "My Title"
        assert data["course_summary"] == "My Summary"

    def test__render_course_applies_meta_override(self, tmp_path: Path) -> None:
        module = TrainingModule(
            chapter_number=1,
            title="Ch1",
            sections=[
                TrainingSection(
                    title="S1",
                    elements=[
                        SlideElement(
                            bloom_level="understand",
                            slide=Slide(title="Slide", content="Hello"),
                        )
                    ],
                )
            ],
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        (output_dir / "course_meta.json").write_text(
            json.dumps({"course_title": "Override Title", "course_summary": "Override Summary"}),
            encoding="utf-8",
        )
        index = _render_course(
            modules=[module],
            output_dir=output_dir,
            embed_images=False,
            course_title="Original Title",
            course_summary="Original Summary",
        )
        html = index.read_text(encoding="utf-8")
        assert "Override Title" in html
        assert "Override Summary" in html


class TestEditableMetadata:
    """Tests for contenteditable metadata in rendered HTML."""

    def test_index_has_contenteditable(self, tmp_path: Path) -> None:
        module = TrainingModule(
            chapter_number=1,
            title="Ch1",
            sections=[
                TrainingSection(
                    title="S1",
                    elements=[
                        SlideElement(
                            bloom_level="understand",
                            slide=Slide(title="Slide", content="Hello"),
                        )
                    ],
                )
            ],
        )
        output_dir = tmp_path / "output"
        index = _render_course(
            modules=[module],
            output_dir=output_dir,
            embed_images=False,
            course_title="Test Course",
        )
        html = index.read_text(encoding="utf-8")
        assert 'contenteditable="true"' in html
        assert 'data-field="course_title"' in html
        assert 'data-original="Test Course"' in html

    def test_index_has_reset_button(self, tmp_path: Path) -> None:
        module = TrainingModule(
            chapter_number=1,
            title="Ch1",
            sections=[
                TrainingSection(
                    title="S1",
                    elements=[
                        SlideElement(
                            bloom_level="understand",
                            slide=Slide(title="Slide", content="Hello"),
                        )
                    ],
                )
            ],
        )
        output_dir = tmp_path / "output"
        index = _render_course(
            modules=[module],
            output_dir=output_dir,
            embed_images=False,
        )
        html = index.read_text(encoding="utf-8")
        assert "resetMetaBtn" in html
        assert "editableControls" in html

    def test_index_has_meta_js(self, tmp_path: Path) -> None:
        module = TrainingModule(
            chapter_number=1,
            title="Ch1",
            sections=[
                TrainingSection(
                    title="S1",
                    elements=[
                        SlideElement(
                            bloom_level="understand",
                            slide=Slide(title="Slide", content="Hello"),
                        )
                    ],
                )
            ],
        )
        output_dir = tmp_path / "output"
        index = _render_course(
            modules=[module],
            output_dir=output_dir,
            embed_images=False,
        )
        html = index.read_text(encoding="utf-8")
        assert "_meta" in html
        assert "saveMeta" in html


# ── Currency protection tests ────────────────────────────────────────────


class TestCurrencyProtection:
    """Tests for currency $ signs not being treated as LaTeX delimiters."""

    def test_currency_not_treated_as_latex(self) -> None:
        result = _markdown_to_html("Buy for **$90** and earn $100")
        assert "<strong>" in result
        # Currency $ is rendered as &#36; HTML entity so KaTeX ignores it
        assert "&#36;90" in result
        assert "&#36;100" in result

    def test_currency_with_decimal(self) -> None:
        result = _markdown_to_html("The price is $3.14 per unit")
        assert "&#36;3.14" in result

    def test_currency_with_comma(self) -> None:
        result = _markdown_to_html("Worth $1,000,000 total")
        assert "&#36;1,000,000" in result

    def test_latex_still_works_with_currency(self) -> None:
        result = _markdown_to_html("Cost is $50 and $x^2$ is math")
        assert "&#36;50" in result
        assert r"\(x^2\)" in result

    def test_only_dollar_sign_is_protected(self) -> None:
        """Digits after the $ should remain visible."""
        result = _markdown_to_html("Earn $200 per day")
        assert "&#36;200" in result
        assert "200" in result


# ── Currency + LaTeX interaction tests ────────────────────────────────────


class TestCurrencyLatexInteraction:
    """Currency $ mixed with LaTeX delimiters — the finance content problem."""

    def test_currency_in_braces_stripped(self) -> None:
        r"""\frac{$60}{1.09} should have inner $ stripped."""
        result = _markdown_to_html(r"Price $= \frac{$60}{1.09}$")
        assert r"\frac{60}{1.09}" in result
        assert "&#36;60" not in result  # NOT treated as currency

    def test_equation_starting_with_digits(self) -> None:
        r"""$931.08 = \frac{60}{X}$ should NOT have opening $ consumed as currency."""
        result = _markdown_to_html(r"$931.08 = \frac{60}{X}$")
        assert r"\frac{60}{X}" in result
        assert "&#36;931" not in result

    def test_latex_heavy_extracted_before_currency(self) -> None:
        r"""$..$ with \commands extracted before currency can interfere."""
        result = _markdown_to_html(r"$z_1 = 9\%$ and $50 profit")
        assert r"z_1 = 9\%" in result      # LaTeX preserved
        assert "&#36;50" in result            # Currency still works

    def test_simple_math_still_works(self) -> None:
        """$x^2$ (no backslash) still works after currency protection."""
        result = _markdown_to_html("Cost $50 and $x^2$ is math")
        assert "&#36;50" in result
        assert r"\(x^2\)" in result

    def test_display_math_unaffected(self) -> None:
        result = _markdown_to_html(r"$$\frac{a}{b}$$")
        assert r"\[\frac{a}{b}\]" in result

    def test_currency_on_non_math_line_unchanged(self) -> None:
        result = _markdown_to_html("Pay $50 and $1,000 total")
        assert "&#36;50" in result
        assert "&#36;1,000" in result

    def test_fix_currency_in_braces_function(self) -> None:
        from src.rendering.html_generator import _fix_currency_in_latex_braces

        assert _fix_currency_in_latex_braces(r"\frac{$60}{$1,000}") == r"\frac{60}{1,000}"
        assert _fix_currency_in_latex_braces(r"\frac{x}{y}") == r"\frac{x}{y}"  # no change
        assert _fix_currency_in_latex_braces("{$3.14}") == "{3.14}"


# ── Inline markdown variant tests ────────────────────────────────────────


class TestMarkdownToHtmlInline:
    """Tests for the inline markdown variant that strips <p> wrappers."""

    def test_strips_single_paragraph_wrapper(self) -> None:
        result = _markdown_to_html_inline("**bold** text")
        assert "<strong>bold</strong> text" == result

    def test_preserves_multi_paragraph(self) -> None:
        result = _markdown_to_html_inline("Para one.\n\nPara two.")
        assert "<p>" in result

    def test_preserves_latex_inline(self) -> None:
        result = _markdown_to_html_inline("The formula $x^2$")
        assert r"\(x^2\)" in result
        assert "<p>" not in result

    def test_preserves_currency(self) -> None:
        result = _markdown_to_html_inline("Pay $50 now")
        assert "&#36;50" in result
        assert "<p>" not in result


# ── HTML sanitization tests ──────────────────────────────────────────────


class TestHtmlSanitization:
    """Tests for HTML sanitization in markdown output."""

    def test_strips_script_tags(self) -> None:
        result = _sanitize_html('<p>Safe</p><script>alert(1)</script>')
        assert "<script>" not in result
        assert "alert" not in result
        assert "<p>Safe</p>" in result

    def test_strips_style_tags(self) -> None:
        result = _sanitize_html('<p>Safe</p><style>body{display:none}</style>')
        assert "<style>" not in result
        assert "<p>Safe</p>" in result

    def test_strips_event_handlers(self) -> None:
        result = _sanitize_html('<img src="x" onerror="alert(1)">')
        assert "onerror" not in result
        assert "alert" not in result

    def test_preserves_normal_html(self) -> None:
        result = _sanitize_html("<p><strong>bold</strong> and <em>italic</em></p>")
        assert "<strong>bold</strong>" in result
        assert "<em>italic</em>" in result

    def test_markdown_to_html_sanitizes(self) -> None:
        """_markdown_to_html should sanitize its output."""
        result = _markdown_to_html('Normal **bold** <script>alert(1)</script>')
        assert "<strong>bold</strong>" in result
        assert "<script>" not in result


# ── FITB statement rendering tests ───────────────────────────────────────


class TestRenderFitbStatement:
    """Tests for LaTeX-aware FITB statement rendering."""

    def test_blanks_outside_latex_become_inputs(self) -> None:
        result = _render_fitb_statement("The _____ theorem.")
        assert '<input type="text"' in result
        assert "_____" not in result

    def test_blanks_inside_latex_become_katex_markers(self) -> None:
        result = _render_fitb_statement(r"$\Delta z \sim N(0, _____) $")
        assert "<input" not in result
        assert r"\underline" in result
        assert r"\hspace{3em}" in result

    def test_mixed_blanks_inside_and_outside_latex(self) -> None:
        result = _render_fitb_statement(
            r"The _____ is $\Delta z \sim N(0, _____)$"
        )
        assert "<input" in result  # outside blank
        assert r"\underline" in result  # inside-latex blank

    def test_markdown_preserved_in_fitb(self) -> None:
        result = _render_fitb_statement("The **important** _____ value.")
        assert "<strong>important</strong>" in result
        assert "<input" in result

    def test_currency_in_fitb(self) -> None:
        result = _render_fitb_statement("The price is $50 and _____ dollars.")
        assert "&#36;50" in result
        assert "<input" in result

    def test_multiple_blanks_indexing(self) -> None:
        result = _render_fitb_statement("_____ and _____")
        assert 'data-blank-index="0"' in result
        assert 'data-blank-index="1"' in result

    def test_hint_button_onclick_preserved(self) -> None:
        """Hint buttons must keep their onclick handler after sanitization."""
        result = _render_fitb_statement("The _____ theorem.", answers=["central"])
        assert 'onclick="revealNextLetters' in result
        assert 'class="hint-letter-btn fitb-hint-letter-btn"' in result


class TestFitbInteractiveAnswerIndices:
    """Tests for identifying which FITB blanks are interactive."""

    def test_all_outside_latex(self) -> None:
        indices = _fitb_interactive_answer_indices("_____ and _____")
        assert indices == [0, 1]

    def test_blank_inside_latex_excluded(self) -> None:
        indices = _fitb_interactive_answer_indices(
            r"The _____ is $N(0, _____)$"
        )
        assert indices == [0]

    def test_no_blanks(self) -> None:
        indices = _fitb_interactive_answer_indices("No blanks here")
        assert indices == []

    def test_all_inside_latex(self) -> None:
        indices = _fitb_interactive_answer_indices(r"$_____ + _____$")
        assert indices == []


# ── Markdown in all element types (integration) ─────────────────────────


class TestMarkdownInAllElements:
    """Integration tests for markdown rendering in non-slide elements."""

    def test_flashcard_renders_markdown(self, tmp_path: Path) -> None:
        module = TrainingModule(
            chapter_number=1,
            title="Ch",
            sections=[
                TrainingSection(
                    title="S",
                    elements=[
                        FlashcardElement(
                            bloom_level="remember",
                            flashcard=Flashcard(
                                front="**Bold** term with $x^2$",
                                back="The definition uses *emphasis*",
                            ),
                        ),
                    ],
                ),
            ],
        )
        output_dir = tmp_path / "output"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)
        html = (output_dir / "chapter_01.html").read_text(encoding="utf-8")
        assert "<strong>Bold</strong>" in html
        assert "<em>emphasis</em>" in html
        assert r"\(x^2\)" in html

    def test_quiz_renders_markdown(self, tmp_path: Path) -> None:
        module = TrainingModule(
            chapter_number=1,
            title="Ch",
            sections=[
                TrainingSection(
                    title="S",
                    elements=[
                        QuizElement(
                            bloom_level="analyze",
                            quiz=Quiz(
                                title="Q",
                                questions=[
                                    QuizQuestion(
                                        question="What is **bold** in $E=mc^2$?",
                                        options=["Option with `code`", "Plain option"],
                                        correct_index=0,
                                        explanation="Because **this** is the reason.",
                                    ),
                                ],
                            ),
                        ),
                    ],
                ),
            ],
        )
        output_dir = tmp_path / "output"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)
        html = (output_dir / "chapter_01.html").read_text(encoding="utf-8")
        assert "<strong>bold</strong>" in html
        assert r"\(E=mc^2\)" in html
        assert "<code>code</code>" in html

    def test_matching_renders_markdown(self, tmp_path: Path) -> None:
        module = TrainingModule(
            chapter_number=1,
            title="Ch",
            sections=[
                TrainingSection(
                    title="S",
                    elements=[
                        MatchingElement(
                            bloom_level="analyze",
                            matching=MatchingExercise(
                                title="Match",
                                left_items=["**Bold** item", "Normal item"],
                                right_items=["Match A", "Match B"],
                            ),
                        ),
                    ],
                ),
            ],
        )
        output_dir = tmp_path / "output"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)
        html = (output_dir / "chapter_01.html").read_text(encoding="utf-8")
        assert "<strong>Bold</strong>" in html

    def test_fitb_currency_not_garbled(self, tmp_path: Path) -> None:
        module = TrainingModule(
            chapter_number=1,
            title="Ch",
            sections=[
                TrainingSection(
                    title="S",
                    elements=[
                        FillInBlankElement(
                            bloom_level="apply",
                            fill_in_the_blank=FillInTheBlank(
                                statement="Buy for $90 and the _____ is $100.",
                                answers=["price"],
                                hint="Think about it.",
                            ),
                        ),
                    ],
                ),
            ],
        )
        output_dir = tmp_path / "output"
        _render_course(modules=[module], output_dir=output_dir, embed_images=False)
        html = (output_dir / "chapter_01.html").read_text(encoding="utf-8")
        assert "&#36;90" in html
        assert "&#36;100" in html
        assert "<input" in html


# ── Math deduplication tests ─────────────────────────────────────────────


class TestDeduplicateMath:
    """Tests for _deduplicate_math which fixes LLM doubled math expressions."""

    def test_no_math_passes_through(self) -> None:
        assert _deduplicate_math("No math here") == "No math here"

    def test_normal_math_unchanged(self) -> None:
        text = "The value is $p = 0.12$ for this case."
        assert _deduplicate_math(text) == text

    def test_removes_plain_echo_after_inline_math(self) -> None:
        # LLM writes "$p=0.12$p=0.12"
        text = "probability $p=0.12$p=0.12 for the device"
        result = _deduplicate_math(text)
        assert result == "probability $p=0.12$ for the device"

    def test_removes_plain_echo_after_with_spaces(self) -> None:
        text = "$X=1$X=1 if it fails"
        result = _deduplicate_math(text)
        assert result == "$X=1$ if it fails"

    def test_removes_plain_echo_before_inline_math(self) -> None:
        # LLM writes "p=0.12$p=0.12$"
        text = "probability p=0.12$p=0.12$ for the device"
        result = _deduplicate_math(text)
        assert result == "probability $p=0.12$ for the device"

    def test_removes_echo_with_latex_commands(self) -> None:
        # LLM writes "$f_X(x)$fX(x)" — LaTeX stripped version matches
        text = "The PMF $f_X(x)$fX(x) gives"
        result = _deduplicate_math(text)
        assert result == "The PMF $f_X(x)$ gives"

    def test_display_math_unchanged(self) -> None:
        text = "The equation is $$E = mc^2$$ which shows"
        assert _deduplicate_math(text) == text

    def test_multiple_expressions_each_deduped(self) -> None:
        text = "We have $p=0.12$p=0.12 and $X=1$X=1"
        result = _deduplicate_math(text)
        assert result == "We have $p=0.12$ and $X=1$"

    def test_end_to_end_through_markdown(self) -> None:
        """Verify dedup runs inside _markdown_to_html."""
        text = "The probability is $p=0.12$p=0.12 for this."
        html = _markdown_to_html(text)
        # Should contain the math delimiters once, not doubled
        assert r"\(p=0.12\)" in html
        # The plain text echo should be gone
        assert "p=0.12p=0.12" not in html
        assert r"\(p=0.12\)p=0.12" not in html


class TestToKatexDelimiters:
    """Tests for the _to_katex_delimiters converter."""

    def test_inline_dollar(self) -> None:
        assert _to_katex_delimiters("$E = mc^2$") == r"\(E = mc^2\)"

    def test_display_dollar(self) -> None:
        assert _to_katex_delimiters(r"$$\frac{a}{b}$$") == r"\[\frac{a}{b}\]"

    def test_paren_passthrough(self) -> None:
        assert _to_katex_delimiters(r"\(x^2\)") == r"\(x^2\)"

    def test_bracket_passthrough(self) -> None:
        assert _to_katex_delimiters(r"\[x^2\]") == r"\[x^2\]"

    def test_plain_text_passthrough(self) -> None:
        assert _to_katex_delimiters("no delimiters") == "no delimiters"


class TestMathOutputNosDollarDelimiters:
    """Integration: math output uses \\(...\\) / \\[...\\], never $...$."""

    def test_inline_math_uses_paren_delimiters(self) -> None:
        result = _markdown_to_html("Compute $x^2 + y^2$ now.")
        assert r"\(x^2 + y^2\)" in result
        assert "$x^2" not in result

    def test_display_math_uses_bracket_delimiters(self) -> None:
        result = _markdown_to_html(r"$$\sum_{i=1}^{n} x_i$$")
        assert r"\[\sum_{i=1}^{n} x_i\]" in result
        assert "$$" not in result

    def test_currency_dollar_survives(self) -> None:
        result = _markdown_to_html("Costs $100 and $x^2$ is math")
        assert "&#36;100" in result
        assert r"\(x^2\)" in result

    def test_orphaned_dollar_escaped(self) -> None:
        """A lone $ that isn't currency or math gets HTML-escaped."""
        result = _markdown_to_html("Pay $ or not")
        assert "&#36;" in result
        assert "$" not in result.replace("&#36;", "")


class TestElementRendererDispatch:
    """Tests for the RENDERERS registry and _render_element dispatch."""

    def test_renderers_covers_all_element_types(self) -> None:
        expected = {
            "section_intro", "slide", "quiz", "flashcard", "fill_in_the_blank",
            "matching", "ordering", "categorization", "error_detection",
            "analogy", "mermaid", "concept_map", "worked_example", "interactive_essay",
        }
        assert set(RENDERERS.keys()) == expected

    def test_renderers_values_are_callable(self) -> None:
        for etype, renderer in RENDERERS.items():
            assert callable(renderer), f"RENDERERS['{etype}'] is not callable"

    def test_unknown_element_type_returns_empty(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader(
            str(Path(__file__).parent.parent / "src" / "rendering" / "templates")
        ))
        result = _render_element({"element_type": "nonexistent"}, env)
        assert result == ""

    def test_render_slide_returns_html(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader(
            str(Path(__file__).parent.parent / "src" / "rendering" / "templates")
        ))
        data = {
            "element_type": "slide",
            "bloom_level": "understand",
            "slide": {
                "title": "Test Slide",
                "content_html": "<p>Hello</p>",
                "image_data": None,
                "source_pages": "pp. 1-2",
            },
        }
        html = _render_element(data, env)
        assert "Test Slide" in html
        assert "<p>Hello</p>" in html
        assert "slide-badge" in html
        assert "bloom-understand" in html
