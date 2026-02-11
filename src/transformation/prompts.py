"""LLM prompt templates for instructional content generation.

Contains the system prompt (implementing evidence-based learning science
principles and anti-LLM guardrails) and the user prompt template.

Key references:
- docs/learning_science.md — retrieval practice, spaced repetition, etc.
- docs/llm_failure_modes.md — the 10 failure modes and anti-LLM checklist
- docs/architecture_next.md — content templates and course structure
"""

# ── Content template descriptions ─────────────────────────────────────────────
# Each template defines a pedagogical approach the LLM must follow.
# The system rotates through these to prevent structural monotony.

TEMPLATE_DESCRIPTIONS = {
    "analogy_first": (
        "ANALOGY-FIRST: Start with a vivid, concrete, real-world analogy that "
        "the learner already understands. Build the analogy fully before "
        "introducing the formal concept. Then bridge from the analogy to the "
        "formal definition. Finally, show a worked numerical example."
    ),
    "narrative": (
        "NARRATIVE: Tell the story of this concept. Who created it and why? "
        "What problem were they trying to solve? What was the intellectual "
        "journey? Use this history to explain why the concept exists in its "
        "current form. Embed the formal content within the story."
    ),
    "worked_example": (
        "WORKED EXAMPLE WITH FADING: Present a concrete problem with a COMPLETE "
        "worked solution showing every step with reasoning. Then present the SAME "
        "problem type with the LAST step replaced by a fill-in-the-blank. Then "
        "present a THIRD variant with the last TWO steps as blanks. Finally, present "
        "a full practice problem as a quiz with NO steps shown. This 4-stage backward "
        "fading sequence (based on Cognitive Load Theory) builds independence gradually."
    ),
    "compare_contrast": (
        "COMPARE & CONTRAST: Take two related or confusable concepts and "
        "explain them side by side. What do they share? Where do they diverge? "
        "When would you use one vs. the other? Use a concrete scenario where "
        "choosing the wrong one leads to a specific, identifiable error."
    ),
    "problem_first": (
        "PROBLEM-FIRST: Start by posing a challenging question or paradox that "
        "the learner cannot yet answer. Let them sit with the difficulty. Then "
        "unfold the explanation that resolves the question. This leverages the "
        "generation effect — struggling first makes the answer stick."
    ),
    "socratic": (
        "SOCRATIC: Guide the learner through a series of questions, each "
        "building on the last. Do not give the answer directly — instead, ask "
        "questions that lead the learner to discover it. The slides should pose "
        "questions, and the quizzes should test whether the discovery landed."
    ),
    "visual_walkthrough": (
        "VISUAL WALKTHROUGH: Describe a diagram, chart, or visual model in "
        "words (since we generate text, not images). Walk through each part of "
        "the visual: 'Imagine a graph where the x-axis is... the y-axis is... "
        "the curve shows...' Make the learner see it mentally."
    ),
    "error_identification": (
        "ERROR IDENTIFICATION: Present a plausible but WRONG solution or "
        "explanation. Ask the learner to find the mistake. Then reveal the "
        "error and explain the correct approach. This builds critical thinking "
        "and guards against common misconceptions."
    ),
    "vignette": (
        "VIGNETTE: Create a realistic scenario (a financial analyst, a "
        "portfolio manager, etc.) facing a decision. Embed the technical "
        "concepts in the scenario. Then ask questions that require applying "
        "the concepts to resolve the scenario."
    ),
    "visual_summary": (
        "VISUAL SUMMARY: After explaining the concepts in this section, create "
        "a concept map showing how the 5-10 key concepts relate to each other. "
        "Use specific relationship labels (not just 'relates to'). Then create "
        "an interactive version where 2-3 edge labels are blanked out for the "
        "learner to fill in."
    ),
    "milestone_assessment": (
        "INTERACTIVE ESSAY ASSESSMENT: This is a chapter-end checkpoint. Generate all the "
        "teaching content for the section (slides, quizzes, etc.) and THEN add an "
        "interactive_essay element as the FINAL element. The essay should test 2-4 core "
        "concepts from the entire chapter. Each explanation prompt should require the "
        "learner to demonstrate UNDERSTANDING, not just recall. Ask them to explain "
        "mechanisms, make connections, or apply concepts to novel situations. Also "
        "generate the tutor_system_prompt for the LLM evaluator."
    ),
}


# System prompt establishing the LLM's role with evidence-based pedagogical
# guardrails. Implements the anti-LLM checklist from docs/llm_failure_modes.md.
SYSTEM_PROMPT = """You are a world-class instructional designer who transforms technical book material into deeply engaging, intuition-building interactive training content.

## Your Pedagogical Philosophy

You believe that **understanding comes before memorization**. Every concept has an intuitive core that can be explained through analogy, story, or visual metaphor. Your job is to find that core and make the learner FEEL it before they formalize it.

You follow these evidence-based principles:
- **Analogy before definition**: For EVERY abstract concept, provide a real-world analogy the learner already understands BEFORE the formal definition. Never start with "X is defined as..."
- **Narrative prose, not bullet points**: Write in flowing, connected paragraphs. Bullet points are ONLY for genuine enumerations (a list of 4+ items). A slide that is entirely bullet points is a FAILURE.
- **Worked examples with reasoning**: For any formula or calculation, show a COMPLETE worked numerical example. Walk through each step explaining WHY, not just HOW.
- **Source attribution**: Every slide must include a source_pages field with the page range from the original text (you'll be told the pages in the prompt).
- **Cross-references**: When a concept builds on something from a previous section, SAY SO explicitly: "Recall from [section name] that..." This builds the interconnected knowledge structure that distinguishes experts from novices.

## Anti-Pattern Guardrails

You MUST AVOID these common LLM failure modes:
1. **Dictionary definitions** — Never produce "X is a Y that does Z" as the primary explanation. Lead with intuition, analogy, or story. The formal definition comes AFTER understanding.
2. **Bullet point soup** — Break this habit. Use paragraphs. Use narrative. Tell stories. Explain the WHY. Bullets are for lists, not for explanations.
3. **Trivial recall questions** — No more than 20% of quiz questions should be "What is the definition of X?" type. Questions should test understanding, application, and analysis.
4. **Obvious wrong answers** — Every quiz distractor must be PLAUSIBLE. A wrong answer should represent a REAL misconception, not a joke. Explain why each wrong answer is wrong.
5. **Missing intuition** — If you explain what something IS without explaining WHY it matters, HOW it works intuitively, and WHAT would happen without it, you have failed.
6. **Repetitive structure** — Vary your approach. Not every section should be slide-quiz-flashcard. Follow the content template specified in the prompt.

## Available Element Types (with fixed Bloom levels)

1. **slide** [understand] — Explanatory content. This is where the teaching happens. Use narrative prose, analogies, worked examples, stories. Each slide should be substantial (150-400 words of actual content), not a few bullet points.
2. **quiz** [apply] — Multiple-choice questions that require APPLYING knowledge, not just recalling definitions. Must have plausible distractors with explanations for EVERY option (correct and incorrect).
3. **flashcard** [remember] — Key term/concept pairs for spaced repetition review. Front should be a question or prompt, back should be a concise answer.
4. **fill_in_the_blank** [apply] — Contextual recall with scaffolding. Test whether the learner can produce key terms in context, not in isolation.
5. **matching** [apply] — Pair related items. Use for concept↔example pairs, not just term↔definition.
6. **mermaid** [understand] — A diagram (flowchart, sequence, state, mindmap, etc.) rendered via Mermaid.js. Use for: processes/workflows (flowchart), time-ordered interactions (sequence), state transitions (state), topic hierarchies (mindmap). The diagram_code field must contain valid Mermaid syntax. Keep diagrams focused: 5-12 nodes maximum.
7. **concept_map** [analyze] — An interactive concept map with labeled edges. Use at the END of a section to summarize relationships between key concepts. Include 3-12 nodes with specific relationship labels. Set blank_edge_indices to hide 2-3 labels for the learner to fill in (interactive retrieval practice).
8. **self_explain** [evaluate] — A free-text exercise asking the learner to explain a concept in their own words. Include 3-5 key_points that a good explanation should cover (used as a self-assessment checklist). Include an example_response (150-250 words) that models a strong explanation. The prompt should be specific enough to test real understanding, not just "explain X" — ask about the WHY, the mechanism, or a specific application. Use sparingly (max 1-2 per section, at the end).
9. **interactive_essay** [evaluate] — A checkpoint exam placed at the END of a chapter/module. Contains 2-4 self_explain prompts testing the chapter's core concepts. Each prompt has its own rubric (key_points). Also generates a tutor_system_prompt: a complete system prompt for an LLM acting as a Socratic evaluator of the learner's responses. The tutor prompt should include: the concepts being tested; a rubric with key points and common misconceptions; instructions to ask follow-up questions when the learner's explanation is vague; instructions to gently correct misconceptions rather than just saying "wrong"; pass/fail criteria (the learner must demonstrate understanding of at least 70% of key points across all prompts). Use interactive_essay elements ONLY at the end of chapters, after all teaching content.

When generating the tutor_system_prompt for an interactive_essay element, write it as a complete system prompt for a different LLM that will evaluate the learner's free-text responses. The tutor system prompt should include:
1. Context: "You are evaluating a learner who just studied [chapter topic]."
2. Rubric: For each concept, list the key points and common misconceptions.
3. Socratic instructions: If the learner's explanation is vague, ask "Can you be more specific about [aspect]?" If the learner has a misconception, say "Interesting — but consider what happens when [scenario]. Does your explanation still hold?" If the learner covers some points but not all, say "Good — you've explained [covered points] well. But what about [missing point]?"
4. Pass/fail: "The learner passes if they demonstrate understanding of at least [threshold]% of key points. After 3 rounds of dialog, make your final assessment. Include [PASS] or [FAIL] in your message."
5. Tone: "Be warm and encouraging. Frame corrections as collaborative discovery, not judgment."

## Mermaid Diagram Syntax

When generating mermaid elements, use valid Mermaid syntax. Examples:

Flowchart:
graph TD
    A[Start] --> B{Decision}
    B -->|Yes| C[Action 1]
    B -->|No| D[Action 2]

Mindmap:
mindmap
  root((Central Topic))
    Subtopic A
      Detail 1
      Detail 2
    Subtopic B
      Detail 3

Sequence:
sequenceDiagram
    Actor A->>Actor B: Message
    Actor B-->>Actor A: Response

IMPORTANT: In JSON strings, use \\n for newlines in Mermaid code.
Keep diagrams simple (5-12 nodes). Complex diagrams confuse rather than clarify.

## When to Use Visual Elements

Generate a mermaid diagram when:
- The content describes a PROCESS or WORKFLOW → use flowchart
- The content describes INTERACTIONS between entities → use sequence diagram
- The content describes STATES and TRANSITIONS → use state diagram
- You want to summarize a TOPIC HIERARCHY → use mindmap

Generate a concept_map when:
- A section covers 5+ interrelated concepts that benefit from showing connections
- At the END of a section as a visual summary
- NOT for simple lists or hierarchies (use mindmap instead)

Do NOT generate visuals when:
- The content is purely narrative or historical
- There are fewer than 3 concepts to relate
- A text explanation is clearer than a diagram

## Graduated Hint Requirements

For EVERY quiz question, you MUST generate graduated hints:
- hint_metacognitive: A Socratic prompt like "Think about what happens when [concept]. Which principle governs this?" Do NOT give away the answer. Redirect the learner's thinking toward the relevant concept.
- hint_strategic: A specific clue like "Consider that option C assumes [X], but the passage states [Y]." This should narrow the field without revealing the answer.
- hint_eliminate_index: The 0-based index of the MOST obviously wrong distractor to grey out. Choose the option that represents the weakest misconception.

For fill-in-the-blank exercises, the hint field should provide a conceptual clue (not just "starts with the letter X"). First-letter reveal happens automatically on the second failed attempt.

For matching exercises, provide pair_explanations: one explanation per pair (in the same order as left_items/right_items) explaining WHY these items belong together. These are shown after the learner completes the exercise.

## Element Difficulty Progression

Each element type has a fixed cognitive level. Do NOT set bloom_level yourself — it is assigned automatically. Instead, focus on generating the right MIX of element types:

- **slide** (understand) — Teaching content. Narrative prose, analogies, worked examples.
- **flashcard** (remember) — Key term/definition pairs for spaced repetition.
- **quiz** (apply) — MCQs that require applying knowledge, not just recalling definitions.
- **fill_in_the_blank** (apply) — Contextual recall with scaffolding.
- **matching** (apply) — Pair related items (concept↔example, not just term↔definition).
- **concept_map** (analyze) — Relationships between 5+ concepts with labeled edges.
- **self_explain** (evaluate) — Learner generates original explanation. Use sparingly (max 1-2 per section, at the end).
- **interactive_essay** (evaluate) — Chapter-end checkpoint with AI tutor. ONLY at end of chapters.
- **mermaid** (understand) — Diagrams for processes, workflows, hierarchies.

Every section MUST contain:
- At least 1 slide (the teaching)
- At least 1 assessment (quiz, flashcard, fill_in_the_blank, or matching)
- At most 2 self_explain elements

## Math Formatting

For mathematical expressions, use LaTeX delimiters:
- Inline math: $E = mc^2$
- Block math: $$\\frac{\\partial f}{\\partial x} = 0$$

IMPORTANT: In JSON strings, backslashes must be double-escaped: use \\\\frac not \\frac.

## Output Format

Return a JSON object with this exact structure:

```json
{
  "elements": [
    {
      "element_type": "slide",
      "bloom_level": "understand",
      "slide": {
        "title": "Concise slide title",
        "content": "Substantial narrative prose (150-400 words) with **markdown** and $math$. Tell stories. Use analogies. Explain the WHY.",
        "speaker_notes": "Additional context or teaching tips.",
        "image_path": null,
        "source_pages": "pp. 42-43"
      }
    },
    {
      "element_type": "quiz",
      "bloom_level": "analyze",
      "quiz": {
        "title": "Quiz title",
        "questions": [
          {
            "question": "A scenario-based question testing deep understanding, not mere recall.",
            "options": ["Plausible option A", "Plausible option B", "Plausible option C", "Plausible option D"],
            "correct_index": 0,
            "explanation": "A is correct because [reasoning]. B is wrong because [specific misconception it represents]. C is wrong because [different misconception]. D is wrong because [yet another].",
            "hint_metacognitive": "Think about which principle governs the relationship between X and Y.",
            "hint_strategic": "Option C assumes a linear relationship, but the passage describes a non-linear one.",
            "hint_eliminate_index": 2
          }
        ]
      }
    },
    {
      "element_type": "flashcard",
      "bloom_level": "remember",
      "flashcard": {
        "front": "Term or question",
        "back": "Definition or answer"
      }
    },
    {
      "element_type": "fill_in_the_blank",
      "bloom_level": "apply",
      "fill_in_the_blank": {
        "statement": "The _____ theorem states that the integral of f over [a,b] equals _____.",
        "answers": ["fundamental", "F(b) - F(a)"],
        "hint": "Think about the relationship between derivatives and integrals."
      }
    },
    {
      "element_type": "matching",
      "bloom_level": "analyze",
      "matching": {
        "title": "Match the Concept to its Real-World Example",
        "left_items": ["Concept A", "Concept B", "Concept C"],
        "right_items": ["Example of A", "Example of B", "Example of C"],
        "pair_explanations": ["A connects to its example because...", "B connects because...", "C connects because..."]
      }
    },
    {
      "element_type": "mermaid",
      "bloom_level": "understand",
      "mermaid": {
        "title": "Process Overview",
        "diagram_code": "graph TD\\n    A[Step 1] --> B[Step 2]\\n    B --> C[Step 3]",
        "caption": "High-level overview of the process.",
        "diagram_type": "flowchart"
      }
    },
    {
      "element_type": "concept_map",
      "bloom_level": "analyze",
      "concept_map": {
        "title": "How Key Concepts Relate",
        "nodes": [
          {"id": "a", "label": "Concept A"},
          {"id": "b", "label": "Concept B"},
          {"id": "c", "label": "Concept C"}
        ],
        "edges": [
          {"source": "a", "target": "b", "label": "depends on"},
          {"source": "b", "target": "c", "label": "is a type of"}
        ],
        "blank_edge_indices": [1]
      }
    },
    {
      "element_type": "self_explain",
      "bloom_level": "evaluate",
      "self_explain": {
        "prompt": "Explain in your own words why diversification reduces portfolio risk. What is the mechanism, and under what conditions does it fail?",
        "key_points": ["Reduces unsystematic/idiosyncratic risk", "Works because asset returns are not perfectly correlated", "Does not eliminate systematic/market risk", "Fails when correlations spike (e.g., financial crises)"],
        "example_response": "Diversification reduces portfolio risk by combining assets whose returns do not move in perfect lockstep...",
        "minimum_words": 50,
        "source_pages": "pp. 42-43"
      }
    },
    {
      "element_type": "interactive_essay",
      "bloom_level": "evaluate",
      "interactive_essay": {
        "title": "Chapter 3 Checkpoint",
        "concepts_tested": ["diversification", "correlation", "portfolio risk"],
        "prompts": [
          {
            "prompt": "Explain the relationship between correlation and diversification benefit.",
            "key_points": ["Lower correlation = greater diversification benefit", "Perfect positive correlation = no benefit"],
            "example_response": "The diversification benefit depends critically on the correlation between assets...",
            "minimum_words": 50,
            "source_pages": ""
          }
        ],
        "passing_threshold": 0.7,
        "tutor_system_prompt": "You are evaluating a learner who just studied portfolio theory..."
      }
    }
  ]
}
```

Return ONLY the JSON object. No markdown fences, no explanation."""


# ── Bloom's-level-specific prompt supplements ────────────────────────────────
# Appended to SYSTEM_PROMPT when a bloom_target is set. Progressively more
# complex prompting for higher cognitive levels (based on AEQG study,
# arXiv:2408.04394: PS1 for Remember/Understand, PS3 for Apply/Analyze,
# PS5 for Evaluate/Create).

BLOOM_PROMPT_SUPPLEMENTS: dict[str, str] = {
    "remember": """

## Bloom's Focus: Remember
Focus on precise terminology and definitions. Keep answers unambiguous.
Generate plenty of flashcards for key terms and formulas.
""",
    "understand": """

## Bloom's Focus: Understand
Emphasize analogies, paraphrasing, and "explain why" narratives. Build intuition before
formalism. Slides should tell stories and connect ideas, not list definitions.
""",
    "apply": """

## Bloom's Focus: Apply
Use novel scenarios and worked examples. Every exercise should require USING knowledge,
not just recognizing it. For quantitative sections, include numerical problems with
step-by-step solutions.
""",
    "analyze": """

## Bloom's Focus: Analyze
Emphasize comparison, decomposition, and error identification. Ask what's different,
what would change, what's the key variable. Use chain-of-thought reasoning: think
step-by-step about what analytical skill is being tested.
""",
    "evaluate": """

## Bloom's Focus: Evaluate
Require judgment and justification. Present debatable scenarios. Self-explain prompts
should ask "which approach is better and why?" or "under what conditions would you
change your recommendation?"
""",
    "create": """

## Bloom's Focus: Create
Generate synthesis and design tasks. The learner should construct something new — a strategy,
a model, an argument — rather than evaluating someone else's work.
""",
}


# ── Target selection prompt (Phase 1 of two-phase generation) ────────────────
# Used to identify reinforcement-worthy insights before generating elements.
# Deliberately minimal — no template instructions, no Bloom's distribution.

TARGET_SELECTION_PROMPT = """You are an expert at identifying what's worth testing in educational material.

Given a section of text, identify 5-10 specific insights that are worth reinforcing through practice. DO NOT identify definitions or surface facts. Instead, identify:

- MECHANISMS: How does X work? Why does it behave this way?
- CONNECTIONS: How does X relate to Y? Why does understanding A help with B?
- APPLICATIONS: When would you use X? What goes wrong if you use it incorrectly?
- EDGE CASES: When does X fail? What assumptions does it depend on?
- CONTRASTS: How is X different from Y? Why is this distinction important?
- CONSEQUENCES: What follows from X? If X is true, what else must be true?

For each target, specify:
1. The concept it relates to
2. The specific insight (NOT a definition — a mechanism, connection, or application)
3. The angle (mechanism, connection, application, edge_case, contrast, consequence)
4. The Bloom's level it naturally maps to
5. Which element type best tests it (quiz, flashcard, fill_in_the_blank, matching, self_explain)

Order targets from foundational to advanced. Return ONLY a JSON object."""


def build_target_selection_prompt(
    section_title: str,
    section_text: str,
    chapter_title: str,
    section_concepts: list[object] | None = None,
    bloom_target: str | None = None,
) -> str:
    """Build the user prompt for Phase 1: reinforcement target selection.

    Deliberately minimal — only the section text and concept list.
    No template instructions, no Bloom's distribution, no cross-references.
    """
    truncated_text = _smart_truncate(section_text, MAX_TEXT_LENGTH)

    concepts_block = ""
    if section_concepts:
        concept_names = [getattr(c, "name", str(c)) for c in section_concepts]
        concepts_block = (
            f"\n\n### Concepts in this section:\n"
            + ", ".join(concept_names)
        )

    bloom_hint = ""
    if bloom_target:
        bloom_hint = f"\n\nFocus on testing at the '{bloom_target}' Bloom's level."

    return f"""## Module: {chapter_title}
## Section: {section_title}
{concepts_block}

### Section Content:

{truncated_text}

---

Identify 5-10 specific insights worth reinforcing through practice. Return ONLY a JSON object.{bloom_hint}"""


# Maximum characters of section text to include in the LLM prompt.
# Longer sections are truncated to stay within token limits while preserving
# the beginning (context) and end (conclusions) of the text.
# Increasing this sends more content to the LLM but risks exceeding token limits.
# Used by: build_section_prompt()
MAX_TEXT_LENGTH = 12_000

# ── Document type generation hints ────────────────────────────────────────────
# Short tips appended to the section prompt when a document type is known.

DOC_TYPE_HINTS: dict[str, str] = {
    "quantitative": "Heavy on formulas and worked examples. Use step-by-step calculations.",
    "narrative": "Emphasize storytelling, timelines, and character-driven explanations.",
    "procedural": "Focus on step-by-step processes, checklists, and decision trees.",
    "analytical": "Emphasize frameworks, comparisons, and critical analysis.",
    "regulatory": "Highlight rules, exceptions, and compliance implications.",
}


def build_section_prompt(
    section_title: str,
    section_text: str,
    chapter_title: str,
    image_count: int,
    table_count: int,
    template: str = "analogy_first",
    source_pages: tuple[int, int] = (0, 0),
    prior_sections: list[str] | None = None,
    learning_objectives: list[str] | None = None,
    bloom_target: str | None = None,
    section_concepts: list[object] | None = None,
    prior_concepts: list[str | dict] | None = None,
    section_characterization: object | None = None,
    reinforcement_targets: list[object] | None = None,
    module_summary: str | None = None,
    section_rationale: str | None = None,
    document_type: str | None = None,
    tables: list | None = None,
    images: list | None = None,
) -> str:
    """Build the user prompt for transforming a single section.

    Includes the section's text, metadata, content template instruction,
    cross-reference context from prior sections, and optional curriculum
    planner guidance (learning objectives and Bloom's target).

    When deep reading analysis is available, also includes concept context:
    which concepts are being taught, what the learner already knows, and
    content characterization signals.

    When reinforcement targets are provided (from Phase 1), includes them
    as explicit instructions for what the exercises MUST test.

    Args:
        section_title: Title of the section being transformed.
        section_text: Full extracted text of the section.
        chapter_title: Parent chapter title for context.
        image_count: Number of images in this section.
        table_count: Number of tables in this section.
        template: Content template to use (from TEMPLATE_DESCRIPTIONS).
        source_pages: (start_page, end_page) for source attribution.
        prior_sections: Titles of previously covered sections for cross-refs.
        learning_objectives: From curriculum planner — what the learner should
            be able to do after this section.
        bloom_target: From curriculum planner — primary Bloom's taxonomy level.
        section_concepts: ConceptEntry objects for concepts taught in this section.
        prior_concepts: Concept names the learner already knows from prior sections.
        section_characterization: SectionCharacterization from deep reading analysis.
        reinforcement_targets: ReinforcementTarget objects from Phase 1 target
            selection. When provided, exercises MUST test these specific insights.

    Returns:
        Formatted user prompt string.
    """
    truncated_text = _smart_truncate(section_text, MAX_TEXT_LENGTH)

    media_notes = []
    if image_count > 0:
        media_notes.append(f"This section contains {image_count} image(s).")
    if table_count > 0:
        media_notes.append(f"This section contains {table_count} table(s).")
    media_line = " ".join(media_notes) if media_notes else "This section has no images or tables."

    template_desc = TEMPLATE_DESCRIPTIONS.get(template, TEMPLATE_DESCRIPTIONS["analogy_first"])

    prior_context = ""
    if prior_sections:
        titles_str = ", ".join(f'"{t}"' for t in prior_sections[-5:])  # Last 5 sections
        prior_context = f"\n\n### Previously Covered Sections (for cross-references):\n{titles_str}\nWhen relevant, reference these by name: \"Recall from [section] that...\""

    source_line = ""
    if source_pages[0] > 0:
        source_line = f"\n\nSource pages: pp. {source_pages[0]}-{source_pages[1]}. Include this in the source_pages field of every slide."

    objectives_block = ""
    if learning_objectives:
        obj_list = "\n".join(f"- {obj}" for obj in learning_objectives)
        objectives_block = f"\n\n### Learning Objectives for This Section:\n{obj_list}\nEnsure every objective is addressed by at least one element."

    bloom_block = ""
    if bloom_target:
        bloom_descriptions = {
            "remember": "emphasize recall and recognition activities (flashcards, fill-in-the-blank)",
            "understand": "emphasize explanation, paraphrasing, analogy-based activities",
            "apply": "emphasize worked examples, calculations, scenario-based problems",
            "analyze": "emphasize comparison, differentiation, error identification",
            "evaluate": "emphasize judgment, justification, critique activities",
            "create": "emphasize synthesis and design activities",
        }
        bloom_desc = bloom_descriptions.get(bloom_target, "")
        bloom_block = f"\n\n### Target Bloom's Level: {bloom_target}\n{bloom_desc}"

    concepts_block = ""
    if section_concepts:
        concept_lines = []
        for concept in section_concepts:
            line = f"- **{concept.name}** ({concept.concept_type}): {concept.definition}"  # type: ignore[union-attr]
            if hasattr(concept, "key_terms") and concept.key_terms:  # type: ignore[union-attr]
                line += f" [terms: {', '.join(concept.key_terms[:5])}]"  # type: ignore[union-attr]
            concept_lines.append(line)
        concepts_block = (
            "\n\n### Concepts in This Section:\n"
            + "\n".join(concept_lines)
            + "\nEnsure EVERY concept listed above is addressed in your output. "
            "Each concept should be explained with intuition, not just defined."
        )

    prior_concepts_block = ""
    if prior_concepts:
        items = []
        for pc in prior_concepts[-20:]:  # Last 20 to avoid bloat
            if isinstance(pc, dict):
                name = pc.get("name", "")
                ctype = pc.get("type", "")
                importance = pc.get("importance", "")
                items.append(f"- **{name}** ({ctype}, {importance})")
            else:
                items.append(f"- {pc}")
        prior_concepts_block = (
            "\n\n### Prior concepts (already taught):\n"
            + "\n".join(items)
            + "\nReference these concepts when relevant without re-explaining them. "
            "Use phrases like \"As we saw with [concept]...\" to build connections."
        )

    characterization_block = ""
    if section_characterization:
        sc = section_characterization
        flags = []
        if getattr(sc, "has_formulas", False):
            flags.append("formulas")
        if getattr(sc, "has_procedures", False):
            flags.append("procedures/steps")
        if getattr(sc, "has_comparisons", False):
            flags.append("comparisons")
        if getattr(sc, "has_definitions", False):
            flags.append("definitions")
        if getattr(sc, "has_examples", False):
            flags.append("examples")
        content_type = getattr(sc, "dominant_content_type", "mixed")
        difficulty = getattr(sc, "difficulty_estimate", "intermediate")
        summary = getattr(sc, "summary", "")
        char_lines = [f"Content type: {content_type} | Difficulty: {difficulty}"]
        if flags:
            char_lines.append(f"Contains: {', '.join(flags)}")
        if summary:
            char_lines.append(f"Summary: {summary}")
        characterization_block = (
            "\n\n### Content Analysis:\n" + "\n".join(char_lines)
        )

    targets_block = ""
    if reinforcement_targets:
        target_lines = []
        for i, target in enumerate(reinforcement_targets, 1):
            angle = getattr(target, "angle", "mechanism")
            insight = getattr(target, "target_insight", str(target))
            bloom = getattr(target, "bloom_level", "understand")
            elem_type = getattr(target, "suggested_element_type", "quiz")
            target_lines.append(
                f"{i}. [{angle}] \"{insight}\" → {elem_type} at {bloom} level"
            )
        targets_block = (
            "\n\n### Reinforcement Targets (from analysis):\n"
            "These are the specific insights your exercises MUST test. "
            "Do not substitute definitions or surface-level recall.\n\n"
            + "\n".join(target_lines)
            + "\n\nEvery quiz, flashcard, fill-in-the-blank, and self-explain element "
            "MUST map to one of these targets."
        )

    # Gap 6: Document type hint
    doc_type_block = ""
    if document_type and document_type not in ("auto", "mixed"):
        hint = DOC_TYPE_HINTS.get(document_type, "")
        if hint:
            doc_type_block = f"\n\n### Document Type: {document_type}\n{hint}"

    # Gap 13: Module context and section rationale
    module_summary_block = ""
    if module_summary:
        module_summary_block = f"\n\n### Module Context:\n{module_summary}"

    rationale_block = ""
    if section_rationale:
        rationale_block = f"\n\n### Why This Template:\n{section_rationale}"

    # Gap 14: Table content
    tables_block = ""
    if tables:
        table_parts: list[str] = []
        for table in tables[:3]:  # Max 3 tables
            headers = getattr(table, "headers", ())
            rows = getattr(table, "rows", ())
            if headers:
                header_line = "| " + " | ".join(str(h) for h in headers) + " |"
                sep_line = "| " + " | ".join("---" for _ in headers) + " |"
                table_parts.append(header_line)
                table_parts.append(sep_line)
            for row in rows[:10]:  # Max 10 rows per table
                row_line = "| " + " | ".join(str(c) for c in row) + " |"
                table_parts.append(row_line)
            table_parts.append("")  # blank line between tables
        if table_parts:
            table_text = "\n".join(table_parts)
            tables_block = f"\n\n### Key Tables:\n{table_text}"

    # Gap 15: Image metadata
    images_block = ""
    if images:
        image_lines: list[str] = []
        for img in images:
            caption = getattr(img, "caption", "")
            path = getattr(img, "path", "")
            page = getattr(img, "page", 0)
            parts = []
            if caption:
                parts.append(caption)
            if path:
                parts.append(f"(file: {path})")
            if page:
                parts.append(f"[page {page}]")
            if parts:
                image_lines.append("- " + " ".join(parts))
        if image_lines:
            image_text = "\n".join(image_lines)
            images_block = f"\n\n### Image References:\n{image_text}"

    return f"""## Module: {chapter_title}
## Section: {section_title}

{media_line}{source_line}

### Content Template for This Section:
{template_desc}
{prior_context}{objectives_block}{bloom_block}{concepts_block}{prior_concepts_block}{characterization_block}{targets_block}{doc_type_block}{module_summary_block}{rationale_block}{tables_block}{images_block}

### Section Content:

{truncated_text}

---

Transform the above section into interactive training elements following the specified content template. Use narrative prose, real-world analogies, and worked examples. Return ONLY a JSON object with an "elements" array."""


def _smart_truncate(text: str, max_length: int) -> str:
    """Truncate text preserving the beginning and end.

    For texts exceeding max_length, keeps the first 80% and last 20%
    of the allowed length, inserting a truncation marker in between.
    This preserves both the introduction (context-setting) and conclusion
    (key takeaways) of the section.
    """
    if len(text) <= max_length:
        return text

    # Ratio of truncated text allocated to the beginning vs. end.
    # 80/20 split keeps most of the introductory context while preserving conclusions.
    head_size = int(max_length * 0.8)
    tail_size = max_length - head_size - 50  # 50 chars for the marker

    return (
        text[:head_size]
        + "\n\n[... content truncated for length ...]\n\n"
        + text[-tail_size:]
    )
