# Product Vision: General-Purpose Learning Experience Engine

## What We're Building

A general-purpose, customizable learning experience platform. You feed in any material (PDF, docs, notes, whatever) and it generates a pedagogically sound, interactive training course — automatically applying evidence-based learning science.

Think NotebookLM meets Brilliant.org meets Anki — but as an open, self-hosted tool you own.

## Core Principle

The system is **material-agnostic**. The learning science and pedagogy are baked into the engine. The user provides the domain-specific content and any customization they want. The engine handles:

- Structuring the content into a proper learning path
- Generating diverse interactive elements (not just slides and quizzes)
- Applying spaced repetition, interleaving, retrieval practice
- Building a knowledge graph of concepts and prerequisites
- Adapting to the learner's progress

## User Requirements (from Matthieu)

### The Learning Experience

- **Chapter → subchapter navigation** — not a flat deck of 168 elements. Structured, browsable course with clear hierarchy.
- **Diverse element types** — slides, quizzes, flashcards, fill-in-the-blank, matching, vignettes, mind maps, infographics, worked examples.
- **Bloom's Taxonomy progression** — exercises that range in difficulty from basic recall to synthesis and evaluation.
- **CFA-style vignettes** — a realistic scenario paragraph with multiple interrelated questions that test application and analysis.
- **Knowledge graph with wiki-links** — concepts linked across chapters. A global graph of subjects showing how everything connects.
- **Bite-sized slides** — small, digestible pieces of content. Not walls of text.
- **Storytelling** — humans learn through narrative. Include history, anecdotes, "why it was made this way," fun facts. Logical connections between items matter more than isolated facts.
- **Analogies** — bridge abstract concepts with concrete, relatable comparisons.

### The Editor / Authoring Experience

- **Editor mode** — an easy frontend where the course creator can:
  - Rephrase or rewrite any generated content
  - Add custom images, diagrams, or screenshots
  - Reorder elements within a section
  - Add, remove, or modify individual elements
  - Override the LLM's choices (change a quiz to a flashcard, etc.)
  - Add custom notes, tips, or warnings
  - Insert their own vignettes or case studies
- **Preview mode** — see exactly what the learner will see
- **Re-generate** — regenerate a single element or section without re-running the whole pipeline
- **Template customization** — choose the overall style, branding, colors
- **Export options** — standalone HTML, SCORM package, or embedded in an LMS

### The Learner Experience

- **Progress tracking** — remember where the learner left off, show completion percentage
- **Spaced repetition scheduling** — flashcards and review questions surfaced at optimal intervals
- **Adaptive difficulty** — track accuracy per topic, adjust difficulty to stay in the zone of proximal development
- **Cross-concept navigation** — wiki-links let the learner jump to related concepts
- **Review mode** — surface due cards and questions from previous sections

## Additional Features (Claude's Suggestions)

### Pedagogical Safeguards

- **Prerequisite enforcement** — the knowledge graph determines what concepts must be mastered before advancing. Don't let the learner skip ahead to derivatives without understanding limits.
- **Anti-LLM-pattern prompting** — explicit system prompt instructions to counteract LLM failure modes:
  - Force analogies for every abstract concept
  - Require narrative prose, not bullet point soup
  - Mandate logical bottom-up ordering
  - Rotate content templates to prevent slide-quiz-slide-quiz monotony
  - Require cross-references to previously learned concepts
- **Factual validation pipeline** — a second LLM pass or rule-based check to catch hallucinations, especially in quantitative domains.

### Content Variety

- **Worked example fading** — start with fully worked examples, progressively remove steps as the learner gains proficiency. (Backed by Sweller & Cooper 1985, one of the strongest effects in instructional design research.)
- **Socratic dialogue elements** — instead of explaining, ask guiding questions that lead the learner to discover the answer.
- **Compare/contrast exercises** — "What's the difference between A and B?" forces analysis-level thinking.
- **Error identification** — present a flawed solution and ask the learner to find and fix the mistake.
- **Concept linking questions** — "How does concept X relate to concept Y that you learned in chapter 3?"

### Structural

- **Course outline view** — a table of contents the learner can browse, showing progress per section
- **Mind map / knowledge graph view** — visual representation of all concepts and their connections
- **Search** — find any concept across the entire course
- **Bookmarks** — learner can bookmark elements for later review

## What This Is NOT

- Not an LMS (no user management, grading, enrollment)
- Not a video platform (text/interactive-first)
- Not a chatbot tutor (structured course, not freeform Q&A)
- Not a note-taking app (the system generates the content, the author refines it)

## Target Users

1. **Course creators** — people with domain expertise who want to turn their material into interactive training without manual authoring
2. **Learners** — anyone going through the generated course, whether self-directed or assigned
3. **Organizations** — teams that want to onboard employees on internal tools, processes, or domain knowledge

## Current State (v0.3)

Working end-to-end pipeline with deep reading, concept graph, FSRS spaced repetition, and multi-document support:

**Pipeline stages:**
1. **Extraction** — PDF → structured Book (PyMuPDF + pdfplumber, LLM-assisted TOC detection)
2. **Deep reading** — LLM analyzes each chapter for concepts, prerequisites, content types (+ regex pre-analysis fallback)
3. **Concept consolidation** — entity resolution across chapters, dependency graph via topological sort
4. **Curriculum planning** — LLM plans module order, template assignments, Bloom's targets using the concept graph
5. **Content transformation** — Two-phase LLM generation: Phase 1 identifies reinforcement targets (mechanisms, connections, edge cases), Phase 2 generates elements with Bloom's-level-specific prompting and explicit targets. Post-generation source verification flags potential hallucinations.
6. **HTML rendering** — self-contained interactive course with chapter/section navigation, FSRS review pages, progress tracking

**Element types (9):** slides, quizzes, flashcards, fill-in-the-blank, matching, Mermaid diagrams, concept maps, self-explanation, interactive essays.

**Content templates (11):** analogy_first, narrative, worked_example, compare_contrast, problem_first, socratic, visual_walkthrough, error_identification, vignette, visual_summary, milestone_assessment (interactive essay).

**Learner-facing features:**
- FSRS-5 spaced repetition with dedicated review pages (review.html, mixed_review.html)
- Client-side progress tracking (localStorage per element, quiz attempts, confidence ratings)
- Dark/light theme toggle (persisted per course)
- Cross-chapter navigation (prev/next, breadcrumbs, sidebar)
- Bloom's Taxonomy badges on all elements, KaTeX math rendering
- Graduated quiz hints (metacognitive → strategic → eliminator)

**Pipeline features:**
- Multi-document input (multiple PDFs → unified course)
- Concept dependency graph drives ordering and cross-references
- Prior concept context passed to every section ("learner already knows X")
- Render-only mode (re-render from training_modules.json without re-running LLM)
- Tiered model routing — cheaper model for low-complexity tasks (TOC detection), primary model for content generation. Opt-in via `LLM_MODEL_LIGHT` env var, 30-50% cost savings.
- Pipeline checkpointing and resume — `--resume` flag skips stages whose output already exists. Partial transformation resume (only missing chapters). Incremental save after each chapter protects against mid-stage crashes.
- Streamlit GUI (`app.py`) — drag-and-drop PDFs, exercise type toggles, LLM config
- CLI with single-doc, multi-doc, directory, single-chapter, resume, and force modes

**Architecture:**
- Functional style: Pydantic v2 discriminated unions, structural pattern matching (match/case), reduce/accumulate folds, Protocol-based LLM client
- 321 tests (pytest), zero external JS dependencies in output (CDN only for KaTeX + Mermaid)

**Supported LLM providers:** OpenAI, OpenRouter (configurable via .env or GUI). Two-tier model support (primary + light).

Tested on "Quantitative Finance with Python" (698 pages, 21 chapters) and multi-document corpora.
