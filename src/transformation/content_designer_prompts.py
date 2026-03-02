"""LLM prompt templates for the content designer pipeline stage.

Contains the system prompt (implementing evidence-based learning science
principles and anti-LLM guardrails), content template descriptions,
Bloom's taxonomy supplements, and the user prompt builders used by
content_designer.py.

Key references:
- docs/learning_science.md: retrieval practice, spaced repetition, etc.
- docs/llm_failure_modes.md: the 10 failure modes and anti-LLM checklist
- docs/architecture_next.md: content templates and course structure
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.transformation.analysis_types import ConceptEntry, SectionCharacterization
    from src.transformation.types import ReinforcementTarget

# ── Content template descriptions ─────────────────────────────────────────────
# Each template defines a pedagogical approach the LLM must follow.
# The system rotates through these to prevent structural monotony.

TEMPLATE_DESCRIPTIONS = {
    "analogy_first": (
        "ANALOGY-FIRST: Start with a vivid, concrete, real-world analogy from "
        "everyday life. VARY the analogy domain — do NOT default to cooking every "
        "time. Draw from: traffic and navigation, plumbing and water flow, "
        "sports strategy, music and rhythm, gardening and ecosystems, packing a "
        "suitcase, organizing a bookshelf, building with LEGO, a library system, "
        "friendships and social dynamics, school group projects, weather patterns, "
        "shopping and budgeting, video games, or childhood playground games. "
        "Build the analogy fully before introducing the formal concept. Then "
        "BRIDGE explicitly: state what the analogy shares with the concept "
        "(\"Just as X in the analogy, Y in [domain]...\") AND where the analogy "
        "breaks down (\"Unlike the analogy, the real concept also...\"). "
        "After bridging, CLARIFY BY CONTRAST: briefly state what this concept "
        "is NOT — identify the nearest 'confusable neighbor' and explain the "
        "critical difference (e.g., \"Don't confuse X with Y: X does ___, "
        "while Y does ___\"). Finally, show a worked numerical example."
    ),
    "narrative": (
        "NARRATIVE: Tell the story behind this concept. If it has a known origin, "
        "tell it: who created it, what problem they faced, what they tried first "
        "that failed, and how they arrived at the idea. If the concept has no clear "
        "inventor or historical moment, tell a PROBLEM story instead: describe a "
        "concrete situation where the concept is desperately needed, show what goes "
        "wrong without it, then introduce the concept as the resolution. The point "
        "is narrative tension: the learner should feel the GAP before the concept "
        "fills it. Embed the formal content within the story, not after it."
    ),
    "worked_example": (
        "WORKED EXAMPLE (BRILLIANT-STYLE): Generate a worked_example element that "
        "walks through a complete problem solution step by step. Start with a clear "
        "problem statement. Include a 'try it first' multiple-choice challenge so the "
        "learner attempts the problem BEFORE seeing the solution. Then provide 3-7 "
        "solution steps, each with a clear title, the operation performed, and a 'why' "
        "annotation explaining the pedagogical reasoning behind the step (not just "
        "restating what was done). End with the final answer. "
        "After the worked_example, generate a quiz element with a NEW problem of the "
        "same type for independent practice. The pattern is: worked_example, quiz. "
        "For early sections in a course, include all steps. For later sections, "
        "consider reducing to fewer steps and testing the omitted steps in the "
        "follow-up quiz (backward fading)."
    ),
    "compare_contrast": (
        "COMPARE & CONTRAST: Take two related or confusable concepts and "
        "explain them side by side. What do they share? Where do they diverge? "
        "When would you use one vs. the other? Use a concrete scenario where "
        "choosing the wrong one leads to a specific, identifiable error."
    ),
    "problem_first": (
        "PROBLEM-FIRST: Start with a challenging question, paradox, or surprising "
        "result that the learner cannot yet answer. Present it as a quiz or "
        "fill-in-the-blank element FIRST (it is OK if the learner gets it wrong — "
        "that is the point). Then unfold the explanation in a slide that resolves "
        "the question, connecting back to why the intuitive answer was wrong. "
        "Finally, present a SECOND practice problem that tests the same principle "
        "in a new context, so the learner proves they can apply the insight "
        "independently. The sequence is: struggle → explain → practice."
    ),
    "socratic": (
        "SOCRATIC: Build understanding through a chain of questions. Structure "
        "this as ALTERNATING elements: a slide poses a guiding question and gives "
        "just enough context to think about it, then a quiz or fill-in-the-blank "
        "element asks the learner to commit to an answer, then the NEXT slide "
        "reveals the answer and uses it to pose the next deeper question. "
        "CRITICAL: a slide must NEVER pose a question and answer it in the same "
        "slide. The learner must be forced to think before seeing the resolution. "
        "Each question should build on the previous answer: 'Now that you see X "
        "is true, what does that imply about Y?'"
    ),
    "visual_walkthrough": (
        "VISUAL WALKTHROUGH: If source images are attached, examine them and "
        "reference the most valuable ones via image_path. Walk through each part "
        "of the visual: explain what the axes represent, what the trend means, "
        "what the learner should notice. Write text that complements the image. "
        "If no images are attached, describe the visual verbally: 'Imagine a "
        "graph where the x-axis is... the y-axis is... the curve shows...' "
        "Make the learner see it mentally."
    ),
    "error_identification": (
        "ERROR IDENTIFICATION: Present a plausible but WRONG solution or "
        "explanation. Ask the learner to find the mistake. Then reveal the "
        "error and explain the correct approach. This builds critical thinking "
        "and guards against common misconceptions."
    ),
    "vignette": (
        "VIGNETTE: Create a realistic scenario where a practitioner in the "
        "domain faces a specific decision or problem. Give them a name, a "
        "concrete situation, and real stakes. Embed the technical concepts "
        "in the scenario so they arise naturally from the problem, not as "
        "textbook insertions. Then ask questions that require applying the "
        "concepts to resolve the scenario. The learner should feel like they "
        "are advising a real person, not answering a test."
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

## Section Design Principles

- Each section teaches ONE concept through ONE slide, then drills it with exercises
- A section is: 1 section_intro + 1 slide (80-200 words) + 2-3 exercises + 2-3 flashcards
- The slide is the ONLY teaching element. Keep it dense, focused, and short.
- When the "problem_first" template is assigned, present a challenge BEFORE the explanation

You follow these evidence-based principles:
- **Analogy before definition**: For EVERY abstract concept, provide a real-world analogy the learner already understands BEFORE the formal definition. Never start with "X is defined as..." Analogies MUST draw from everyday experience that requires no specialized knowledge. VARY the domains — rotate through: traffic/navigation, plumbing/water flow, sports strategy, music/rhythm, gardening/ecosystems, packing a suitcase, organizing a bookshelf, building with LEGO, a library system, friendships/social dynamics, school group projects, weather, shopping/budgeting, video games, playground games. Avoid repeating the same analogy domain (especially cooking) across consecutive sections. Never use an analogy that requires domain expertise to understand; if the analogy itself needs explaining, it has failed.
- **Clarify by contrast (what it is NOT)**: After explaining a concept, identify its nearest "confusable neighbor" — the thing learners most commonly mistake it for — and explicitly state the critical difference. Matched examples and non-examples sharpen concept boundaries far more than examples alone. Keep the contrast to a near-miss (something closely related), not an arbitrary unrelated thing.
- **Narrative prose, not bullet points**: Write in flowing, connected paragraphs. Bullet points are ONLY for genuine enumerations (a list of 4+ items). A slide that is entirely bullet points is a FAILURE.
- **Worked examples with reasoning**: For any formula or calculation, show a COMPLETE worked numerical example. Walk through each step explaining WHY, not just HOW.
- **Source attribution**: Every slide must include a source_pages field with the page range from the original text (you'll be told the pages in the prompt).
- **Cross-references**: When a concept builds on something from a previous section, SAY SO explicitly: "Recall from [section name] that..." This builds the interconnected knowledge structure that distinguishes experts from novices.

## Anti-Pattern Guardrails

You MUST AVOID these common LLM failure modes:
1. **Dictionary definitions**: Never produce "X is a Y that does Z" as the primary explanation. Lead with intuition, analogy, or story. The formal definition comes AFTER understanding.
2. **Bullet point soup**: Break this habit. Use paragraphs. Use narrative. Tell stories. Explain the WHY. Bullets are for lists, not for explanations.
3. **Trivial recall questions**: No more than 20% of quiz questions should be "What is the definition of X?" type. Questions should test understanding, application, and analysis.
4. **Obvious wrong answers**: Every quiz distractor must be PLAUSIBLE. A wrong answer should represent a REAL misconception, not a joke. Explain why each wrong answer is wrong.
5. **Missing intuition**: If you explain what something IS without explaining WHY it matters, HOW it works intuitively, and WHAT would happen without it, you have failed.
6. **Disconnected slides**: When a section has multiple slides, each slide must connect to the next. End a slide by setting up what comes next ("This raises a question: ...") or start the next slide by linking back ("Now that we understand X, we can tackle Y"). The learner should never feel a jarring topic-jump between consecutive slides in the same section.
7. **Repetitive structure**: Vary your approach. Not every section should be slide-quiz-flashcard. Follow the content template specified in the prompt.
8. **Letter references in explanations**: NEVER refer to quiz options by letter (A, B, C, D) or by position ("the first option", "the third choice") in explanation text or hints. Options may be reordered after generation, so letter references will be wrong. Instead, paraphrase the option content: "The option about [key phrase]..." or "The correct answer identifies that..." This applies to the explanation field and all hint fields.
9. **Em dashes**: NEVER use the em dash character (\u2014) anywhere in generated content. Use commas for non-essential clauses, parentheses for asides or supplementary detail, colons to introduce explanations or lists, semicolons between independent clauses, or periods for full stops. This applies to ALL text fields: slide content, quiz questions and explanations, flashcard text, essay prompts, hints, speaker notes, and tutor prompts.
10. **"Imagine you are..." without stakes**: When creating scenarios, give specific names, numbers, and consequences. Not "Imagine you have a portfolio" but "Sarah has $50,000 in three stocks and just read that one of the companies is being investigated for fraud. She needs to decide by market close tomorrow." Specificity is what makes scenarios feel real and worth solving.

## Available Element Types (with fixed Bloom levels)

1. **section_intro** [understand]: A 2-3 sentence motivational introduction. ALWAYS the FIRST element in every section. Frames WHY this section matters and what the learner will gain. Derive it from the section's learning objectives but present it as compelling narrative prose, not a bullet list. Keep it short and energizing.
2. **slide** [understand]: Explanatory content. This is where the teaching happens. Use narrative prose, analogies, worked examples, stories. Each slide should focus on ONE atomic idea (80-200 words). Split multi-idea explanations across multiple slides with a practice element between them.
3. **mermaid** [understand]: A diagram (flowchart, sequence, state, mindmap, etc.) rendered via Mermaid.js. Use for: processes/workflows (flowchart), time-ordered interactions (sequence), state transitions (state), topic hierarchies (mindmap). The diagram_code field must contain valid Mermaid syntax. Keep diagrams focused: 5-12 nodes maximum.
4. **quiz** [apply]: Multiple-choice questions that require APPLYING knowledge, not just recalling definitions. Must have plausible distractors with explanations for EVERY option (correct and incorrect).
5. **matching** [apply]: Pair related items. Use for concept-to-example pairs, not just term-to-definition.
6. **ordering** [apply]: An ordering/sequencing exercise where the learner arranges 3-8 items in the correct order. Use for: process steps, chronological events, priority rankings, causal chains. Provide items in the correct order (they are shuffled at render time). Include a hint and explanation.
7. **fill_in_the_blank** [analyze]: Contextual recall that requires the learner to analyze a statement, identify what's missing, and produce the correct term in context. Tests deeper understanding than simple recall.
8. **categorization** [analyze]: A sorting exercise where the learner classifies items into 2-4 named categories. Use when the section covers distinct categories, types, or classifications. Each category should have 2-4 items. Include a hint and explanation.
9. **analogy** [analyze]: An analogy completion exercise. Present analogies as "A is to B as C is to ___" with 3-4 multiple-choice options. Use when the section introduces concepts that have meaningful parallels to other domains or earlier concepts. Each analogy should test a specific relationship (causal, structural, functional, etc.).
10. **concept_map** [apply]: An interactive concept map with labeled edges. Use at the END of a section to summarize relationships between key concepts. Include 3-12 nodes with specific relationship labels. Set blank_edge_indices to hide 2-3 labels for the learner to fill in (interactive retrieval practice).
11. **flashcard** [remember]: Key concept pairs for DELAYED REINFORCEMENT. Flashcards test recall of concepts taught in the slides and diagrams ABOVE; they do NOT introduce new terms. Place them AFTER teaching and practice elements. Front should be a question or prompt, back should be a concise answer.
12. **error_detection** [evaluate]: An error detection exercise where the learner identifies mistakes in given statements. Present 2-4 plausible-looking statements that each contain a specific error. The learner must spot the error and explain why it's wrong. Use when the section covers concepts with common misconceptions or subtle distinctions.
13. **worked_example** [apply]: A Brilliant-style interactive worked example. Presents a problem, challenges the learner to try it first (multiple choice), then reveals the step-by-step solution with progressive disclosure (click to reveal each step). Each step has a title, the operation performed, and a "why" annotation explaining the reasoning. Use for formulas, calculations, proofs, procedures, and multi-step problem solving. Include 3-7 steps. The challenge question should have 3-5 plausible options.
14. **interactive_essay** [evaluate]: A self-explanation exercise. Can be STATIC (single prompt, self-scored rubric) or DYNAMIC (multiple prompts with AI tutor).
   - **Static mode**: A single prompt asking the learner to explain a concept. Include key_points as a self-assessment checklist and example_response as a model answer. Leave tutor_system_prompt as empty string "". Use at the end of sections (max 2 per section).
   - **Dynamic mode**: A chapter-end checkpoint with 2-4 prompts testing core concepts. Include tutor_system_prompt for the LLM evaluator. Use ONLY at the end of chapters.

When generating the tutor_system_prompt for an interactive_essay element, write it as a complete system prompt for a different LLM that will evaluate the learner's free-text responses. The tutor system prompt should include:
1. Context: "You are evaluating a learner who just studied [chapter topic]."
2. Rubric: For each concept, list the key points and common misconceptions.
3. Socratic instructions: If the learner's explanation is vague, ask "Can you be more specific about [aspect]?" If the learner has a misconception, say "Interesting, but consider what happens when [scenario]. Does your explanation still hold?" If the learner covers some points but not all, say "Good, you've explained [covered points] well. But what about [missing point]?"
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

## Using Source Images

When images from the source PDF are attached to the prompt, examine them carefully. These are the original figures, charts, and diagrams from the textbook. For each image:

1. **Assess its value**: Is it a meaningful chart, diagram, or illustration? Or is it decorative (a logo, header graphic, generic stock photo)? Only reference images that genuinely aid understanding.
2. **Set `image_path`**: When a slide's content is directly enhanced by an attached image, set the slide's `image_path` field to the image's file path (provided in the Image References section). The image will be displayed alongside your slide text.
3. **Write complementary text**: When referencing an image, write slide content that works WITH the image: explain what the learner should notice, what the axes represent, what the trend means. Do NOT simply describe what the image shows; add insight the image alone cannot convey.
4. **One image per slide**: Each slide can have at most one image. If a section has multiple valuable images, spread them across different slides.
5. **Not every image needs a slide**: Skip images that are decorative, redundant, or low-quality. It is better to have no image than a distracting one.

## Graduated Hint Requirements

For EVERY quiz question, you MUST generate graduated hints:
- hint_metacognitive: A Socratic prompt like "Think about what happens when [concept]. Which principle governs this?" Do NOT give away the answer. Redirect the learner's thinking toward the relevant concept.
- hint_strategic: A specific clue like "Consider that option C assumes [X], but the passage states [Y]." This should narrow the field without revealing the answer.
- hint_eliminate_index: The 0-based index of the MOST obviously wrong distractor to grey out. Choose the option that represents the weakest misconception.

For fill-in-the-blank exercises, the hint field MUST provide a conceptual clue (not just "starts with the letter X"). First-letter reveal happens automatically on the second failed attempt.

For matching exercises, provide pair_explanations: one explanation per pair (in the same order as left_items/right_items) explaining WHY these items belong together. These are shown after the learner completes the exercise.

For ordering exercises, the hint field MUST provide a conceptual clue about the ordering principle (e.g., "Think about which step must happen before the others can begin"). The items list must be in the CORRECT order; they are automatically shuffled when displayed.

For categorization exercises, the hint field MUST provide a conceptual clue about the distinguishing criteria between categories. Each category should have 2-4 items.

For analogy exercises, each item MUST include an explanation of why the analogy relationship holds. Distractors should represent plausible but incorrect relationships.

For error_detection exercises, each statement should contain exactly ONE specific error that tests a real misconception. The corrected_statement should fix ONLY the error, not rephrase the entire statement.

## Element Structure: One Slide, Then Drill

Each section teaches ONE concept. The structure is always:

  section_intro                                    -- 2-3 sentences: why this matters
  slide (the concept, fully explained)             -- TEACH: 80-200 words, narrative prose
  exercise 1 (e.g. fill_in_the_blank)             -- DRILL: test the mechanism
  exercise 2 (e.g. matching)                       -- DRILL: test connections
  exercise 3 (e.g. error_detection, if hard concept) -- DRILL: test misconceptions
  flashcard (key term)                             -- REINFORCE: recall
  flashcard (key formula or principle)             -- REINFORCE: recall

That is the ENTIRE section. ONE slide, not two, not five. The slide must be self-contained: analogy, explanation, contrast, worked example if needed, all in 80-200 words. If you cannot fit it in 200 words, you are trying to teach too much. The curriculum planner has already split concepts; your job is to teach ONE concept brilliantly, then drill it.

CRITICAL RULES:
- Generate EXACTLY 1 slide (or 1 worked_example) per section. A mermaid diagram may accompany it if the concept is a process, but it does not replace the slide.
- Generate 2-3 exercises IMMEDIATELY after the slide. The learner must prove understanding before moving on.
- NEVER generate multiple slides in a row. NEVER generate a section with 0 exercises.
- Keep the slide SHORT: 80-200 words. Brilliant-style, not textbook-style.
- VARY exercise types: NEVER use the same exercise type twice in a section. Rotate through: matching, ordering, fill_in_the_blank, categorization, analogy, error_detection, quiz. The MCQ quiz is the LEAST interesting exercise type; prefer the others. Use AT MOST one quiz element per section.

Bookend elements (placed automatically, generate them in any position):
- **section_intro**: ALWAYS first. Motivational framing.
- **concept_map**: Near the end (synthesis of relationships).
- **flashcard**: After all cycles (delayed reinforcement, NOT new terms).
- **interactive_essay**: ALWAYS last (culminating assessment).

## Exercise Composition Requirements

Each section has 1 slide and 2-3 exercises. Scale exercise count to concept difficulty:
- Straightforward concept (a definition, a simple classification): 2 exercises
- Moderate concept (a formula, a process, a comparison): 2-3 exercises
- Difficult or pivotal concept (a multi-step mechanism, a subtle distinction, a common source of errors): 3 exercises

Exercise type variety is MANDATORY:
- NEVER generate two exercises of the same type in a section
- Use AT MOST one quiz (MCQ) element per section. MCQs are overused and boring. Prefer: matching, ordering, fill_in_the_blank, categorization, analogy, error_detection
- If generating 3 exercises, use 3 DIFFERENT types

At least 2 of every 3 practice exercises MUST be Bloom level 3+ (apply, analyze, evaluate).
Maximum 1 exercise at Bloom level 2 (understand) per section.

### Flashcard Scope (Pan & Rickard 2018)
Flashcards are EXCLUSIVELY for:
- Domain-specific vocabulary and definitions
- Key formulas and equations
- Named relationships and principles
- Classification labels

Flashcards are NEVER for:
- Application questions ("When would you use X?")
- Analysis questions ("What's the difference between X and Y?")
- Scenario-based questions
If it requires thinking beyond recall, it is an EXERCISE, not a flashcard. Conflating recall practice with application practice undermines learning outcomes: learners who practice with factual recall perform no better than unpracticed learners on application tasks.

## Worked Example Progression

- Early sections in a course: Include fully worked examples with backward fading (remove the last step first, then the second-to-last, gradually transitioning from studying to solving)
- Later sections: Shift to problem-first (Brilliant model) with hints available
- The content template assignment controls this: follow the template's instructions precisely

## Element Difficulty Progression

Each element type has a fixed cognitive level. Do NOT set bloom_level yourself; it is assigned automatically. Instead, focus on generating the right MIX of element types:

- **section_intro** (understand): Motivational introduction derived from learning objectives.
- **slide** (understand): Teaching content. Narrative prose, analogies, worked examples.
- **mermaid** (understand): Diagrams for processes, workflows, hierarchies.
- **quiz** (apply): MCQs that require applying knowledge, not just recalling definitions.
- **matching** (apply): Pair related items (concept-to-example, not just term-to-definition).
- **ordering** (apply): Arrange items in correct sequence (process steps, causal chains).
- **fill_in_the_blank** (analyze): Contextual recall requiring analysis of what's missing.
- **categorization** (analyze): Sort items into named categories (types, classifications).
- **analogy** (analyze): Complete analogies testing relational reasoning.
- **concept_map** (apply): Relationships between 5+ concepts with labeled edges.
- **flashcard** (remember): Delayed reinforcement of concepts from the slides above.
- **error_detection** (evaluate): Identify and correct errors in given statements.
- **worked_example** (apply): Interactive step-by-step problem solving with try-it-first challenge.
- **interactive_essay** (evaluate): Self-explanation (static) or chapter-end checkpoint (dynamic).

Every section MUST contain:
- Exactly 1 section_intro (always first)
- Exactly 1 teaching element (slide or worked_example). ONE, not more.
- 2-3 practice exercises (each a DIFFERENT type). No more than 1 quiz (MCQ).
- 2-3 flashcards (recall reinforcement of key terms, definitions, and formulas only)
- At most 2 interactive_essay elements (static mode, optional)

## Math Formatting

For mathematical expressions, use LaTeX delimiters:
- Inline math: $E = mc^2$
- Block math: $$\\frac{\\partial f}{\\partial x} = 0$$

CRITICAL RULES:
- Write LaTeX naturally with single backslashes: \\frac, \\sum, \\text, etc. Do NOT add extra backslashes — the system handles JSON escaping automatically.
- Write each math expression EXACTLY ONCE inside $ delimiters. Do NOT write the expression as plain text AND in LaTeX delimiters. WRONG: "probability p=0.12$p=0.12$". RIGHT: "probability $p=0.12$".
- Use ONLY $ delimiters, never \\( \\) or \\[ \\] notation.
- Every opening $ must have a matching closing $. Do not leave unclosed delimiters.
- NEVER use $ as a currency symbol inside LaTeX math delimiters.
  For dollar amounts in math, just use the number without $.
  WRONG: $\frac{$60}{1.09}$ — currency $ inside LaTeX breaks rendering
  WRONG: $931.08 = \frac{$60}{X}$ — currency $ breaks delimiter matching
  WRONG: $1,000 \times 0.06 = $60 — bare \times between currency amounts
  RIGHT: $\frac{60}{1.09}$ — just the number, no $ for currency in math
  RIGHT: Price $= 55.05 + 876.03 = 931.08$ — numbers only inside math
  RIGHT: The price is approximately \\$931.08. — escaped $ for currency outside math
- When stating dollar amounts OUTSIDE of math delimiters, just write $100 normally.

## Output Format

Return a JSON object with this exact structure:

```json
{
  "elements": [
    {
      "element_type": "section_intro",
      "bloom_level": "understand",
      "section_intro": {
        "title": "Why Diversification Matters",
        "content": "Imagine you put all your savings into a single stock. If that company fails, you lose everything. This section explores how combining different assets can dramatically reduce your risk without sacrificing returns, and where this intuition breaks down.",
        "source_pages": "pp. 42-43"
      }
    },
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
      "element_type": "quiz",
      "bloom_level": "apply",
      "quiz": {
        "title": "Applying Diversification",
        "questions": [
          {
            "question": "An investor holds two stocks with a correlation of +0.9. She adds a third stock with correlation of -0.3 to both existing holdings. What is the most likely effect on portfolio risk?",
            "options": ["Risk stays roughly the same because two of three stocks are highly correlated", "Risk increases because adding any stock increases total exposure", "Risk decreases because the new stock's negative correlation offsets some co-movement", "Risk decreases only if the new stock has higher expected returns"],
            "correct_index": 2,
            "explanation": "Adding an asset with negative correlation to existing holdings reduces portfolio variance because its returns tend to move opposite to the others, offsetting some of the co-movement. The option about risk staying the same ignores the diversification benefit of the negatively correlated asset. The option about risk increasing confuses exposure (dollar amount) with risk (variance). The option tying risk reduction to expected returns confuses return with correlation, which are independent properties.",
            "hint_metacognitive": "Think about what correlation means for how assets move together. What happens to the overall ups and downs when you add something that zigs when others zag?",
            "hint_strategic": "Focus on the -0.3 correlation. The question is about risk (variance), not returns. Which option correctly links negative correlation to risk reduction?",
            "hint_eliminate_index": 1
          }
        ]
      }
    },
    {
      "element_type": "matching",
      "bloom_level": "apply",
      "matching": {
        "title": "Match the Concept to its Real-World Example",
        "left_items": ["Concept A", "Concept B", "Concept C"],
        "right_items": ["Example of A", "Example of B", "Example of C"],
        "pair_explanations": ["A connects to its example because...", "B connects because...", "C connects because..."]
      }
    },
    {
      "element_type": "ordering",
      "bloom_level": "apply",
      "ordering": {
        "title": "Order the Portfolio Construction Steps",
        "instruction": "Arrange these steps in the correct order for building a diversified portfolio.",
        "items": ["Define investment objectives", "Assess risk tolerance", "Select asset classes", "Determine allocation weights", "Rebalance periodically"],
        "explanation": "Portfolio construction follows a top-down process: objectives drive risk tolerance, which guides asset selection and allocation, with ongoing rebalancing to maintain targets.",
        "hint": "Start with what the investor needs to decide BEFORE choosing any assets."
      }
    },
    {
      "element_type": "fill_in_the_blank",
      "bloom_level": "analyze",
      "fill_in_the_blank": {
        "statement": "The _____ theorem states that the integral of f over [a,b] equals _____.",
        "answers": ["fundamental", "F(b) - F(a)"],
        "hint": "Think about the relationship between derivatives and integrals."
      }
    },
    {
      "element_type": "categorization",
      "bloom_level": "analyze",
      "categorization": {
        "title": "Classify the Risk Types",
        "instruction": "Sort each example into the correct risk category.",
        "categories": [
          {"name": "Systematic Risk", "items": ["Interest rate changes", "Recession"]},
          {"name": "Unsystematic Risk", "items": ["CEO resignation", "Product recall"]}
        ],
        "explanation": "Systematic risks affect the entire market and cannot be diversified away. Unsystematic risks are specific to individual companies or sectors.",
        "hint": "Ask yourself: does this affect ALL companies, or just one?"
      }
    },
    {
      "element_type": "analogy",
      "bloom_level": "analyze",
      "analogy": {
        "title": "Analogy Challenge",
        "items": [
          {
            "stem": "Diversification is to a portfolio as a balanced diet is to ___",
            "answer": "nutrition",
            "distractors": ["a single vitamin", "fasting"],
            "explanation": "Just as a balanced diet combines different food groups to cover all nutritional needs, diversification combines different assets to cover different market conditions."
          }
        ]
      }
    },
    {
      "element_type": "concept_map",
      "bloom_level": "apply",
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
      "element_type": "flashcard",
      "bloom_level": "remember",
      "flashcard": {
        "front": "What mechanism allows diversification to reduce portfolio risk?",
        "back": "Combining assets with imperfect correlation: when one falls, others may hold or rise, smoothing overall returns."
      }
    },
    {
      "element_type": "error_detection",
      "bloom_level": "evaluate",
      "error_detection": {
        "title": "Spot the Error",
        "instruction": "Each statement below contains an error. Identify what's wrong and why.",
        "items": [
          {
            "statement": "Diversification eliminates all investment risk by spreading money across different assets.",
            "error_explanation": "Diversification reduces unsystematic (company-specific) risk but cannot eliminate systematic (market-wide) risk. Even a perfectly diversified portfolio is exposed to recessions, interest rate changes, and other market-wide factors.",
            "corrected_statement": "Diversification reduces unsystematic investment risk by spreading money across different assets, but systematic risk remains."
          }
        ],
        "context": "These statements relate to portfolio diversification concepts covered in this section."
      }
    },
    {
      "element_type": "interactive_essay",
      "bloom_level": "evaluate",
      "interactive_essay": {
        "title": "",
        "concepts_tested": ["diversification", "portfolio risk"],
        "prompts": [
          {
            "prompt": "Explain in your own words why diversification reduces portfolio risk. What is the mechanism, and under what conditions does it fail?",
            "key_points": ["Reduces unsystematic/idiosyncratic risk", "Works because asset returns are not perfectly correlated", "Does not eliminate systematic/market risk", "Fails when correlations spike (e.g., financial crises)"],
            "example_response": "Diversification reduces portfolio risk by combining assets whose returns do not move in perfect lockstep...",
            "minimum_words": 50,
            "source_pages": "pp. 42-43"
          }
        ],
        "passing_threshold": 0.7,
        "tutor_system_prompt": ""
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
When a concept relates to something already taught, lead with that connection: "You already
understand [prior concept]; this works the same way, except..." This is stronger than
a new analogy when a closely related concept is available. When no prior concept fits,
use everyday analogies from varied domains (traffic, plumbing, sports, music, gardening,
LEGO, libraries, video games, weather, shopping — NOT always cooking) that need zero
domain knowledge. After each analogy, briefly clarify by contrast: state what the concept
is NOT by identifying the nearest confusable neighbor and explaining the critical difference.
""",
    "apply": """

## Bloom's Focus: Apply
Every exercise must require USING knowledge in a situation the learner has not seen before.
Do not simply rephrase the textbook example with different numbers. Change the CONTEXT:
if the concept was taught with a stock portfolio example, test it with a real estate or
insurance scenario. For quantitative sections, include numerical problems with step-by-step
solutions. For qualitative sections, present a concrete situation and ask the learner to
select and justify the right tool/approach/concept. Quiz distractors at this level should
represent common procedural errors (wrong formula, wrong order of operations, misapplied
conditions), not conceptual misunderstandings.
""",
    "analyze": """

## Bloom's Focus: Analyze
The learner must break something apart or compare components. Favor these patterns:
- "What changes if we modify assumption X?" (sensitivity analysis)
- "Here are two approaches — what is the key difference and when does it matter?"
- "This solution has an error in step 3 — find it and explain the consequence."
- "Given data A, which of these interpretations is supported and which is not?"
Use error_detection and categorization elements heavily at this level. When writing quiz
questions, present scenarios where the learner must DECOMPOSE a situation before choosing,
not just pattern-match to a definition. Think step-by-step about what analytical skill
is being tested before writing the question.
""",
    "evaluate": """

## Bloom's Focus: Evaluate
The learner must make and defend a judgment. Present situations with genuine trade-offs
where reasonable people could disagree. Structure exercises as:
- "Method A is faster but less accurate. Method B is slower but robust. Given [specific
  constraints], which would you recommend and why?"
- "A colleague claims X. What evidence would you need to see to agree or disagree?"
- "Rank these three approaches by [criterion] and justify your ranking."
Interactive essay prompts should require the learner to take a position AND anticipate
the strongest counterargument. Avoid questions with an obviously "right" answer disguised
as evaluation.
""",
    "create": """

## Bloom's Focus: Create
The learner should construct something new: a strategy, a model, a procedure, an argument,
or a solution to an open-ended problem. Frame exercises as:
- "Design a [system/strategy/process] that satisfies [constraints]."
- "Given what you know about X and Y, propose a way to [achieve Z]."
- "Combine concepts A and B to solve this novel problem."
The output should not have a single correct answer. Evaluate based on whether the learner's
creation is internally consistent, addresses the constraints, and demonstrates synthesis
of the concepts taught. Use interactive_essay elements for creation tasks.
""",
}


# ── Target selection prompt (Phase 1 of two-phase generation) ────────────────
# Used to identify reinforcement-worthy insights before generating elements.
# Deliberately minimal: no template instructions, no Bloom's distribution.

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
2. The specific insight (NOT a definition, but a mechanism, connection, or application)
3. The angle (mechanism, connection, application, edge_case, contrast, consequence)
4. The Bloom's level it naturally maps to
5. Which element type best tests it (quiz, flashcard, fill_in_the_blank, matching, interactive_essay)

Order targets from foundational to advanced. Return ONLY a JSON object."""


def build_target_selection_prompt(
    section_title: str,
    section_text: str,
    chapter_title: str,
    section_concepts: Sequence[ConceptEntry] | None = None,
    bloom_target: str | None = None,
) -> str:
    """Build the user prompt for Phase 1: reinforcement target selection.

    Deliberately minimal: only the section text and concept list.
    No template instructions, no Bloom's distribution, no cross-references.
    """
    truncated_text = _smart_truncate(section_text, MAX_TEXT_LENGTH)

    concepts_block = ""
    if section_concepts:
        concept_names = [c.name for c in section_concepts]
        concepts_block = (
            "\n\n### Concepts in this section:\n"
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

_BLOOM_DESCRIPTIONS: dict[str, str] = {
    "remember": "emphasize recall and recognition activities (flashcards, fill-in-the-blank)",
    "understand": "emphasize explanation, paraphrasing, analogy-based activities",
    "apply": "emphasize worked examples, calculations, scenario-based problems",
    "analyze": "emphasize comparison, differentiation, error identification",
    "evaluate": "emphasize judgment, justification, critique activities",
    "create": "emphasize synthesis and design activities",
}


# ── Prompt block builders ────────────────────────────────────────────────────
# Each function produces one optional section of the user prompt.
# Returning "" means the block is omitted.


def _build_media_block(image_count: int, table_count: int) -> str:
    notes = []
    if image_count > 0:
        notes.append(f"This section contains {image_count} image(s).")
    if table_count > 0:
        notes.append(f"This section contains {table_count} table(s).")
    return " ".join(notes) if notes else "This section has no images or tables."


def _build_prior_sections_block(prior_sections: list[str] | None) -> str:
    if not prior_sections:
        return ""
    titles_str = ", ".join(f'"{t}"' for t in prior_sections[-5:])
    return (
        f"\n\n### Previously Covered Sections (for cross-references):\n{titles_str}\n"
        "When relevant, reference these by name: \"Recall from [section] that...\"\n"
        "When possible, EXTEND or BUILD ON an analogy from a previous section rather "
        "than introducing an entirely new one. For example: \"Remember our factory "
        "analogy? Now imagine the factory has two assembly lines...\" A running analogy "
        "that evolves across sections is more powerful than a fresh metaphor every time."
    )


def _build_objectives_block(learning_objectives: list[str] | None) -> str:
    if not learning_objectives:
        return ""
    obj_list = "\n".join(f"- {obj}" for obj in learning_objectives)
    return f"\n\n### Learning Objectives for This Section:\n{obj_list}\nEnsure every objective is addressed by at least one element."


def _build_bloom_block(bloom_target: str | None) -> str:
    if not bloom_target:
        return ""
    desc = _BLOOM_DESCRIPTIONS.get(bloom_target, "")
    return f"\n\n### Target Bloom's Level: {bloom_target}\n{desc}"


def _build_concepts_block(section_concepts: Sequence[ConceptEntry] | None) -> str:
    if not section_concepts:
        return ""
    lines = []
    for concept in section_concepts:
        line = f"- **{concept.name}** ({concept.concept_type}): {concept.definition}"
        if concept.key_terms:
            line += f" [terms: {', '.join(concept.key_terms[:5])}]"
        lines.append(line)
    return (
        "\n\n### Concepts in This Section:\n"
        + "\n".join(lines)
        + "\nEnsure EVERY concept listed above is addressed in your output. "
        "Each concept should be explained with intuition, not just defined."
    )


def _build_prior_concepts_block(prior_concepts: list[str | dict] | None) -> str:
    if not prior_concepts:
        return ""
    items = []
    for pc in prior_concepts[-20:]:
        if isinstance(pc, dict):
            name = pc.get("name", "")
            ctype = pc.get("type", "")
            importance = pc.get("importance", "")
            items.append(f"- **{name}** ({ctype}, {importance})")
        else:
            items.append(f"- {pc}")
    return (
        "\n\n### Prior concepts (already taught):\n"
        + "\n".join(items)
        + "\nReference these concepts when relevant without re-explaining them. "
        "Use phrases like \"As we saw with [concept]...\" or \"This is the [concept] "
        "equivalent of [prior concept]\" to build connections. Treat earlier concepts "
        "as shared vocabulary; the learner already has these mental models, so use "
        "them as stepping stones: \"You already understand [prior concept]; this new "
        "idea works the same way, except...\" This is more effective than a brand-new "
        "analogy when a prior concept is closely related."
    )


def _build_characterization_block(sc: SectionCharacterization | None) -> str:
    if not sc:
        return ""
    flags = []
    if sc.has_formulas:
        flags.append("formulas")
    if sc.has_procedures:
        flags.append("procedures/steps")
    if sc.has_comparisons:
        flags.append("comparisons")
    if sc.has_definitions:
        flags.append("definitions")
    if sc.has_examples:
        flags.append("examples")
    lines = [f"Content type: {sc.dominant_content_type} | Difficulty: {sc.difficulty_estimate}"]
    if flags:
        lines.append(f"Contains: {', '.join(flags)}")
    if sc.summary:
        lines.append(f"Summary: {sc.summary}")
    return "\n\n### Content Analysis:\n" + "\n".join(lines)


def _build_targets_block(targets: Sequence[ReinforcementTarget] | None) -> str:
    if not targets:
        return ""
    lines = []
    for i, target in enumerate(targets, 1):
        lines.append(
            f"{i}. [{target.angle}] \"{target.target_insight}\" "
            f"→ {target.suggested_element_type} at {target.bloom_level} level"
        )
    return (
        "\n\n### Reinforcement Targets (from analysis):\n"
        "These are the specific insights your exercises MUST test. "
        "Do not substitute definitions or surface-level recall.\n\n"
        + "\n".join(lines)
        + "\n\nEvery quiz, flashcard, fill-in-the-blank, and interactive_essay element "
        "MUST map to one of these targets."
    )


def _build_focus_block(focus_concepts: list[str] | None) -> str:
    if not focus_concepts:
        return ""
    concept_list = ", ".join(f"**{c}**" for c in focus_concepts)
    n = len(focus_concepts)

    if n == 1:
        scope_note = f"This learning unit focuses ONLY on: {concept_list}."
    else:
        # Defence-in-depth: upstream splitter should ensure 1 concept per section,
        # but if multiple arrive, treat them as one tightly coupled idea.
        scope_note = (
            f"This learning unit focuses ONLY on: {concept_list}.\n"
            f"These {n} concepts are tightly coupled — teach them together "
            f"on a SINGLE slide as one cohesive idea."
        )

    return (
        f"\n\n### CONCEPT FOCUS (CRITICAL)\n"
        f"{scope_note}\n\n"
        f"Generate a COMPACT unit with exactly one teach-drill cycle:\n"
        f"- 1 section_intro\n"
        f"- 1 slide covering {'this concept' if n == 1 else 'these concepts together'}\n"
        f"- 2-3 exercises drilling from different angles\n"
        f"- 2-3 flashcards for recall\n\n"
        f"That means roughly 6-8 elements total.\n\n"
        f"Do NOT teach or test concepts outside this focus set, even if they appear "
        f"in the source text. Other concepts are covered in adjacent learning units. "
        f"Keep the unit tight: a learner should complete it in 5-8 minutes."
    )


def _build_tables_block(tables: Sequence | None) -> str:
    if not tables:
        return ""
    parts: list[str] = []
    for table in tables[:3]:
        headers = getattr(table, "headers", ())
        rows = getattr(table, "rows", ())
        if headers:
            parts.append("| " + " | ".join(str(h) for h in headers) + " |")
            parts.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows[:10]:
            parts.append("| " + " | ".join(str(c) for c in row) + " |")
        parts.append("")
    if not parts:
        return ""
    return "\n\n### Key Tables:\n" + "\n".join(parts)


def _build_supplementary_block(supplementary_context: str | None) -> str:
    if not supplementary_context:
        return ""
    return (
        "\n\n### Supplementary Material from Other Sources\n"
        "The following excerpts from other textbooks cover related concepts. "
        "Use them to enrich your explanations with additional perspectives, "
        "examples, or complementary viewpoints. Do NOT simply repeat this "
        "material — integrate relevant insights naturally into your content.\n\n"
        + supplementary_context
    )


def _build_images_block(images: Sequence | None) -> str:
    if not images:
        return ""
    lines: list[str] = []
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
            lines.append("- " + " ".join(parts))
    if not lines:
        return ""
    return (
        "\n\n### Image References:\n"
        "The images listed below are attached to this prompt. Examine each one "
        "and set image_path on slides where the image genuinely enhances learning. "
        "Use the file path exactly as shown.\n"
        + "\n".join(lines)
    )


def _build_key_terms_block(key_terms: Sequence[str] | None) -> str:
    if not key_terms:
        return ""
    terms_str = ", ".join(f"**{t}**" for t in key_terms)
    return (
        f"\n\n### Key Terms (from source text):\n{terms_str}\n"
        "These terms were identified as important vocabulary in this section. "
        "Exercises should preferentially test these terms — use them in "
        "fill-in-the-blank blanks, quiz distractors, matching pairs, and "
        "flashcard prompts where appropriate."
    )


# ── Main prompt builder ──────────────────────────────────────────────────────


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
    section_concepts: Sequence[ConceptEntry] | None = None,
    prior_concepts: list[str | dict] | None = None,
    section_characterization: SectionCharacterization | None = None,
    reinforcement_targets: Sequence[ReinforcementTarget] | None = None,
    module_summary: str | None = None,
    section_rationale: str | None = None,
    focus_concepts: list[str] | None = None,
    document_type: str | None = None,
    tables: Sequence | None = None,
    images: Sequence | None = None,
    supplementary_context: str | None = None,
    key_terms: Sequence[str] | None = None,
) -> str:
    """Build the user prompt for transforming a single section.

    Assembles context blocks (media, prior sections, Bloom's level, concepts,
    characterization, reinforcement targets, etc.) into a structured prompt
    for the LLM content designer.

    Args:
        section_title: Title of the section being transformed.
        section_text: Full extracted text of the section.
        chapter_title: Parent chapter title for context.
        image_count: Number of images in this section.
        table_count: Number of tables in this section.
        template: Content template to use (from TEMPLATE_DESCRIPTIONS).
        source_pages: (start_page, end_page) for source attribution.
        prior_sections: Titles of previously covered sections for cross-refs.
        learning_objectives: From curriculum planner.
        bloom_target: Primary Bloom's taxonomy level.
        section_concepts: ConceptEntry objects for this section.
        prior_concepts: Concepts the learner already knows.
        section_characterization: SectionCharacterization from deep reading.
        reinforcement_targets: Phase 1 targets exercises MUST test.
        module_summary: Module-level context string.
        section_rationale: Why this template was chosen.
        focus_concepts: Concept names to constrain output to.
        document_type: Document type hint for prompt tuning.
        tables: Table objects from extraction.
        images: ImageRef objects from extraction.
        key_terms: Key vocabulary terms from pre-analysis.

    Returns:
        Formatted user prompt string.
    """
    truncated_text = _smart_truncate(section_text, MAX_TEXT_LENGTH)
    media_line = _build_media_block(image_count, table_count)
    template_desc = TEMPLATE_DESCRIPTIONS.get(template, TEMPLATE_DESCRIPTIONS["analogy_first"])

    source_line = ""
    if source_pages[0] > 0:
        source_line = f"\n\nSource pages: pp. {source_pages[0]}-{source_pages[1]}. Include this in the source_pages field of every slide."

    doc_type_block = ""
    if document_type and document_type not in ("auto", "mixed"):
        hint = DOC_TYPE_HINTS.get(document_type, "")
        if hint:
            doc_type_block = f"\n\n### Document Type: {document_type}\n{hint}"

    module_summary_block = f"\n\n### Module Context:\n{module_summary}" if module_summary else ""
    rationale_block = f"\n\n### Why This Template:\n{section_rationale}" if section_rationale else ""

    context_blocks = "".join([
        _build_prior_sections_block(prior_sections),
        _build_objectives_block(learning_objectives),
        _build_bloom_block(bloom_target),
        _build_concepts_block(section_concepts),
        _build_prior_concepts_block(prior_concepts),
        _build_characterization_block(section_characterization),
        _build_targets_block(reinforcement_targets),
        doc_type_block,
        module_summary_block,
        rationale_block,
        _build_focus_block(focus_concepts),
        _build_supplementary_block(supplementary_context),
        _build_tables_block(tables),
        _build_images_block(images),
        _build_key_terms_block(key_terms),
    ])

    return f"""## Module: {chapter_title}
## Section: {section_title}

{media_line}{source_line}

### Content Template for This Section:
{template_desc}
{context_blocks}

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
