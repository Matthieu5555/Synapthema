# Where LLMs Fail at Teaching (And How to Fix It)

LLMs are powerful content generators but terrible instructional designers by default. Every failure mode listed here has been observed in practice and must be actively counteracted through prompt engineering, system architecture, or post-processing.

> **Implementation status key:** Each failure mode is annotated with its current mitigation status in the codebase.

## 1. No Logical Ordering — MITIGATED

> **Status:** Deep reader analyzes each chapter for concepts and prerequisites. Concept consolidator builds a dependency graph with topological sort. Curriculum planner uses the graph to order modules. `prior_concepts` context passed to every section prompt. See: `deep_reader.py`, `concept_consolidator.py`, `curriculum_planner.py`.

**The problem:** LLMs generate text token-by-token, optimizing for local coherence rather than global pedagogical structure. They don't naturally build concepts bottom-up. The result: jumping between topics, mentioning advanced concepts before foundations, and a general lack of narrative arc.

**Why it matters:** Cognitive load theory says intrinsic load is driven by element interactivity — how many new things must be held in mind simultaneously. Poor ordering maximizes this unnecessarily.

**Fix:**
- Use an explicit knowledge graph with prerequisite ordering that the LLM must follow
- Generate content within a pre-defined curriculum structure, not open-ended
- The prompt should specify: "This section assumes the learner has already mastered X and Y. Do not re-explain those. Build on them."
- Post-processing: validate that generated content doesn't reference undefined concepts

## 2. Missing Prerequisites — MITIGATED

> **Status:** Concept graph tracks prerequisites across chapters. `prior_concepts` list passed to every section prompt ("The learner already knows: X, Y, Z"). Deep reader detects `external_prerequisites` per chapter. See: `prompts.py` ("What the Learner Already Knows" block), `content_designer.py` (cumulative_concepts tracking).

**The problem:** LLMs explain derivatives without checking if the user understands limits. They discuss portfolio optimization without verifying knowledge of expected returns. There is no inherent awareness of what the learner knows.

**Why it matters:** A Stanford study showed a 22 percentage point improvement in test scores when a learner model was added to ChatGPT-based tutoring vs. raw ChatGPT alone.

**Fix:**
- Maintain a learner model separate from the LLM that tracks mastered concepts
- Before generating content on topic X, check that all prerequisites of X in the knowledge graph are marked as mastered
- Include prerequisite context in every prompt: "The learner has already mastered: [list]. They have NOT yet learned: [list]."

## 3. Over-Formatting (Bullet Point Soup) — MITIGATED

> **Status:** System prompt in `prompts.py` explicitly instructs: narrative prose, not bullet point soup. Template-specific instructions enforce varied formatting per content type.

**The problem:** LLMs default to excessive bullet points, headers, bold text, and structured formatting regardless of whether the content warrants it. This is because over-formatted text is overrepresented in training data (blog posts, documentation, tutorials). The result: fragmented information that should be connected prose.

**Why it matters:** Fragmenting explanations into bullet points breaks the narrative thread. Learners get a list of facts instead of understanding how those facts connect. This directly undermines the storytelling principle.

**Fix:**
- Explicit system prompt: "Use narrative prose for explanations. Use bullet points ONLY for genuine lists of items. Vary formatting across content. Do NOT default to headers and bullet points for everything."
- Post-process output to detect excessive formatting (e.g., >60% of content in bullet points) and flag for regeneration
- Include examples of good narrative explanations in the system prompt

## 4. Lack of Analogies — MITIGATED

> **Status:** `analogy_first` template mandates: real-world analogy → formal definition → worked example. Template rotation ensures analogies appear regularly. System prompt requires concrete examples before abstractions. See: `prompts.py` (TEMPLATE_DESCRIPTIONS).

**The problem:** LLMs default to abstract, definitional explanations rather than concrete analogies. They will define "diversification" technically instead of saying "Don't put all your eggs in one basket." They explain recursion with a formal definition instead of "Russian nesting dolls."

**Why it matters:** Concrete examples and analogies are one of the strongest effects in instructional design research (Sweller's worked example effect). Analogies bridge the gap between existing knowledge and new concepts.

**Fix:**
- Explicit prompt: "For EVERY abstract concept, provide a real-world analogy or concrete example before the formal definition. The analogy should be something a non-expert would immediately understand."
- Include analogy generation as a mandatory step: concept → analogy → formal definition → worked example
- Post-process: flag sections that introduce abstract concepts without an accompanying analogy

## 5. No Difficulty Calibration — MITIGATED

> **Status:** Bloom's targets assigned per section by curriculum planner based on concept position in dependency graph (foundation → remember, advanced → analyze/evaluate). Section characterizations include `difficulty_estimate`. **Bloom's-level-specific prompt supplements** now tailor the LLM's instructional strategy per level (simple recall prompting for Remember, chain-of-thought for Analyze, step-by-step judgment scaffolding for Evaluate). No runtime adaptive difficulty yet (needs learner model). See: `prompts.py` (BLOOM_PROMPT_SUPPLEMENTS), `content_designer.py`, `curriculum_planner.py`, `analysis_types.py`.

**The problem:** Without a learner model, LLMs produce everything at roughly the same level of complexity. Quiz questions are either all easy recall or all hard analysis, with no progression.

**Why it matters:** The zone of proximal development research shows learners need challenges just beyond their current ability. Too easy = boredom, too hard = frustration, both = no learning.

**Fix:**
- Pass Bloom's level and learner proficiency as explicit parameters in every prompt
- "Generate a Bloom's Level 3 (Apply) question for a learner who has mastered X and Y but not Z"
- Enforce a difficulty distribution: roughly 20% Remember, 30% Understand, 25% Apply, 15% Analyze, 10% Evaluate/Create
- Track accuracy per topic and adjust the target Bloom's distribution based on performance

## 6. Hallucination in Educational Context — MITIGATED

> **Status:** Source page citations (`source_pages`) on all elements. Content grounded in extracted text (full section text passed to LLM, not summaries). **Post-generation source verification** now extracts claims (formulas, numeric assertions, definitions) from generated elements and checks them against the source text via substring matching and Jaccard similarity. Unverifiable claims are logged as warnings and attached to `TrainingSection.verification_notes` in the intermediate JSON for human review. Zero LLM cost — all verification is rule-based. See: `content_designer.py` (_verify_elements, _extract_formulas, _extract_numeric_claims, _extract_definitions), `types.py` (TrainingSection.verification_notes).

**The problem:** LLMs confidently present plausible but incorrect information as fact. The authoritative tone makes errors especially dangerous in education. Research on math tutoring found that hallucinated feedback negatively affects learning gain and creates lasting misconceptions.

**Why it matters:** One wrong explanation can be worse than no explanation, because the learner builds incorrect schema that must later be unlearned.

**Fix:**
- Factual validation pipeline: LLM generates → second pass validates claims → flag uncertain content
- For quantitative domains: validate calculations programmatically
- Ground all generated content in source material — include citations back to specific pages/sections
- Add a visible "Source: p.42" attribution to every generated element so the learner (or course creator) can verify

## 7. Repetitive Structure — MITIGATED

> **Status:** 11 content templates with mandatory rotation. Curriculum planner assigns templates based on section content type (conceptual → analogy_first, procedural → worked_example, comparative → compare_contrast). Fallback rotation prevents repeats. See: `content_designer.py` (TEMPLATE_ROTATION), `curriculum_planner.py`.

**The problem:** LLMs fall into slide-quiz-slide-quiz-flashcard monotony. Same structure every section, every chapter.

**Why it matters:** Monotony reduces engagement and violates the interleaving principle. Varied practice formats improve transfer.

**Fix:**
- Define a content template library: narrative explanation, worked example with fading, analogy-first introduction, case study/vignette, Socratic dialogue, compare/contrast, error identification, timeline/history, visual walkthrough
- Rotate templates on a defined schedule — track which templates were used recently and avoid repetition
- The prompt should specify which template to use for each section, not let the LLM choose the same one every time

## 8. No Learner Model — PARTIALLY MITIGATED

> **Status:** Client-side learner model implemented (`_learner_model.js`). Tracks per-concept accuracy, Bloom's-level breakdown, and mastery classification (new/progressing/mastered/struggling). Mastery dashboard on index page. Concept-based review prioritization in review.html. Integrated into quiz, flashcard, fill-in-blank, matching, self-explain, and milestone handlers. Still missing: server-side persistence, adaptive difficulty (adjusting Bloom's distribution based on accuracy), and multi-device sync.

**The problem:** The LLM has no memory of what the learner knows, what they've gotten wrong, or how they're progressing. Every interaction starts from zero context.

**Why it matters:** The single most impactful architectural decision. 22 percentage point improvement in the Stanford study.

**Fix:**
- Build a separate learner model outside the LLM:
  - Mastered concepts (boolean per concept node in the knowledge graph)
  - Accuracy history (per topic, per Bloom's level)
  - Common misconceptions (what they get wrong repeatedly)
  - Learning pace (time per element, sessions per week)
  - Review schedule (FSRS state per flashcard/question)
- Feed relevant learner state into every LLM prompt as context
- Use the learner model to drive adaptive difficulty, prerequisite gating, and spaced repetition

## 9. Poor Question Quality — MITIGATED

> **Status:** Bloom's level is a mandatory parameter on every element. Curriculum planner assigns bloom_target per section. System prompt enforces plausible distractors with explanations. Graduated hints (metacognitive → strategic → eliminator) on quiz questions. Multiple question types: quiz, fill-in-the-blank, matching, self-explain, milestone exams. **Two-phase assessment generation**: Phase 1 identifies reinforcement-worthy insights (mechanisms, connections, edge cases — not definitions) before Phase 2 generates elements. Phase 2 receives explicit targets that exercises MUST test. See: `types.py` (ReinforcementTarget, ReinforcementTargetSet, QuizQuestion hints), `prompts.py` (TARGET_SELECTION_PROMPT, BLOOM_PROMPT_SUPPLEMENTS), `content_designer.py` (_select_reinforcement_targets).

**The problem:** LLMs default to trivial recall questions ("What is the definition of...?") rather than questions testing deep understanding. They also generate questions with obvious wrong answers, or questions where the answer is in the question.

**Why it matters:** The retrieval practice effect (g=0.50) only works when questions are meaningful. Trivial recall questions don't build schema.

**Fix:**
- Use Bloom's taxonomy tags as mandatory parameters with enforced distribution
- No more than 20% Remember-level questions
- Mandate specific question types: case-based vignettes, compare/contrast, error identification, application to novel scenarios
- Require plausible distractors with explanations for why each incorrect option is wrong
- Post-process: flag questions where the answer contains the question text, or where distractors are obviously wrong

## 10. Missing Connections Between Concepts — MITIGATED

> **Status:** `prior_concepts` list passed to every section prompt, instructing "reference without re-explaining." Concept graph tracks cross-chapter dependencies. Deep reader detects `external_prerequisites` per chapter. Curriculum planner uses topological order to sequence modules. No wiki-link rendering yet. See: `prompts.py` ("What the Learner Already Knows"), `concept_consolidator.py`.

**The problem:** LLMs present concepts in isolation because each generation call has no memory of the full curriculum. Chapter 3's material doesn't reference chapter 1's concepts.

**Why it matters:** Expert knowledge is densely interconnected. Novice knowledge is fragmented. The goal of education is to help novices build expert-like knowledge structures. Isolated concepts don't achieve this.

**Fix:**
- Use the knowledge graph to inject cross-references into every content generation: "This concept relates to [previously learned X] because..."
- Generate explicit linking questions: "How does concept X relate to concept Y from chapter 3?"
- Wiki-link style connections in the rendered output
- After a set of related concepts are taught, include a synthesis element that explicitly connects them

---

## Summary: The Anti-LLM Prompt Checklist

Every content generation prompt should include these guardrails:

| # | Guardrail | Status |
|---|-----------|--------|
| 1 | Specify the prerequisite concepts the learner has already mastered | Done — `prior_concepts` in every prompt |
| 2 | Specify the target Bloom's level for each element | Done — `bloom_target` from curriculum planner |
| 3 | Require at least one analogy per abstract concept | Done — `analogy_first` template + system prompt |
| 4 | Require narrative prose, not bullet point soup | Done — system prompt enforces |
| 5 | Specify which content template to use | Done — 11 templates, planner assigns per section |
| 6 | Include cross-references to previously learned concepts | Done — prior_concepts context block |
| 7 | Require source attribution (page numbers) | Done — `source_pages` on slides |
| 8 | Specify difficulty calibration based on learner proficiency | Done — Bloom's-level-specific prompt supplements (BLOOM_PROMPT_SUPPLEMENTS), no runtime adaptation yet |
| 9 | Forbid questions where the answer is in the question | Done — system prompt enforces |
| 10 | Require explanations for every quiz answer | Done — `explanation` field + graduated hints |
