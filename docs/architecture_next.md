# Architecture: Next Version

Where the system needs to go based on the research and requirements.

## Current Architecture (v0.3)

```
PDF(s) → Extract → Book(s)  ←── checkpoint: book_structure.json
                      │
              Deep Reader (LLM per chapter + regex pre-analysis)
                      │                                         ←── checkpoint: chapter_analyses.json
              Concept Consolidator (entity resolution + embedding similarity, topological sort)
                      │
              Document Type Detector (regex heuristics, zero LLM cost)
                      │
              Curriculum Planner (LLM, uses concept graph + chapter analyses + document type)
                      │                                         ←── checkpoint: curriculum_blueprint.json
              Content Designer (2-phase LLM: target selection → element generation)
                      │          [Bloom's prompt supplements + source verification]
                      │          [incremental save after each chapter]
              training_modules.json (editable intermediate)     ←── checkpoint: training_modules.json
                      │
              HTML Renderer → self-contained interactive course
                                ├── chapter pages (sidebar nav, progress tracking, learner model, review buttons)
                                ├── index page (mastery dashboard, concept graph viz, due card counts)
                                ├── review.html (FSRS-5 spaced repetition, concept-priority sorting)
                                └── mixed_review.html (cross-chapter practice, learner model tracking)

LLM Client: two-tier model routing
  ├── Primary model (gpt-5.2) — deep reading, planning, content generation
  └── Light model (gpt-4.1-mini, opt-in via LLM_MODEL_LIGHT) — TOC detection, target selection
```

Multi-stage pipeline with deep reading and concept graph. Multi-document support. Embedding-based concept deduplication (optional `sentence-transformers` dep, graceful fallback to heuristics). Auto-detected document type (quantitative, narrative, procedural, analytical, regulatory, mixed) biases template selection in the curriculum planner. Two-phase assessment generation (Phase 1 reinforcement target selection → Phase 2 element generation with explicit targets). Bloom's-level-specific prompt supplements (progressively complex prompting per cognitive level). Post-generation source verification (rule-based claim extraction + string similarity, zero LLM cost, warnings in `TrainingSection.verification_notes`). FSRS-5 spaced repetition with review pages. Client-side progress tracking and theme toggle (localStorage). Client-side learner model with concept-level mastery tracking, mastery dashboard, and concept-based review prioritization. Interactive concept graph visualization (vis-network.js) on index page with mastery overlay. Pipeline checkpointing with `--resume` (per-stage skip + partial transformation resume + incremental save after each chapter). Tiered model routing (primary + light model, opt-in via `LLM_MODEL_LIGHT`). Functional architecture (Protocol-based DI, reduce/accumulate, discriminated unions, match/case). Streamlit GUI and CLI. No editor mode yet.

## Target Architecture (v1.0)

```
                    ┌──────────────┐
                    │  Source Docs  │  PDF, DOCX, Markdown, URLs, notes
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Extraction  │  Text, images, tables, structure
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Knowledge   │  Concepts, prerequisites, relationships
                    │    Graph     │  (drives everything downstream)
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐ ┌──▼──────┐ ┌──▼──────────┐
       │  Content    │ │ Learning│ │  Assessment  │
       │  Generator  │ │ Path    │ │  Generator   │
       │  (LLM)     │ │ Planner │ │  (LLM)       │
       └──────┬──────┘ └──┬──────┘ └──┬───────────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼───────┐
                    │  Course JSON │  The complete intermediate representation
                    │  (editable)  │  ← Editor mode operates here
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐ ┌──▼──────┐ ┌──▼──────────┐
       │  HTML       │ │ SCORM   │ │  Other       │
       │  Renderer   │ │ Export  │ │  Renderers   │
       └─────────────┘ └─────────┘ └──────────────┘
```

## Key Components

### 1. Knowledge Graph — IMPLEMENTED

The concept graph drives curriculum planning and content generation. Built by the deep reader and concept consolidator.

**Current implementation** (`analysis_types.py`, `concept_consolidator.py`):

```
ConsolidatedConcept: {
    canonical_name: str,
    aliases: list[str],
    best_definition: str,
    concept_type: str,   # definition | formula | process | comparison | principle | ...
    chapter_mentions: list[int],
    importance: str      # core | supporting | peripheral
}

ConceptEdge: {
    source -> target,    # source depends on target
    relationship: str    # requires | builds_on | contrasts_with | applies
}

ConceptGraph: {
    concepts: list[ConsolidatedConcept],
    edges: list[ConceptEdge],
    topological_order: list[str],
    foundation_concepts: list[str],  # in-degree 0
    advanced_concepts: list[str]     # highest in-degree
}
```

**How it's built:**
1. Deep reader analyzes each chapter: extracts concepts, definitions, types, prerequisites, section characterizations
2. Concept consolidator performs entity resolution across all chapters using 4 strategies:
   - Exact name match (case-insensitive)
   - Substring containment
   - Key term overlap (>50% threshold)
   - Embedding cosine similarity (>0.75 threshold, via optional `sentence-transformers` dep)
3. Prerequisite edges collected and deduplicated, canonical names resolved
4. Kahn's algorithm produces topological learning order
5. Foundation and advanced concept layers identified

**Still needed for v1.0:**
- Human review in editor mode (validate/adjust the graph)
- Learner model that uses the graph for adaptive path personalization — PARTIALLY DONE (client-side concept mastery tracking implemented; adaptive path personalization not yet)
- ~~Visual graph exploration in the rendered HTML output~~ — DONE (vis-network.js interactive graph on index page, mastery-colored node borders)

### 2. Course JSON (The Editable Intermediate Representation)

This is the central data model. Everything upstream produces it, everything downstream consumes it. The editor mode operates directly on it.

```json
{
  "course": {
    "title": "Fixed Income",
    "description": "...",
    "knowledge_graph": { ... },
    "modules": [
      {
        "id": "mod_01",
        "title": "Bond Fundamentals",
        "learning_outcomes": [
          "Calculate the price of a bond given yield and coupon",
          "Explain the relationship between price and yield"
        ],
        "sections": [
          {
            "id": "sec_01_01",
            "title": "What is a Bond?",
            "concepts": ["bond", "coupon", "maturity", "face_value"],
            "elements": [
              {
                "type": "slide",
                "bloom_level": "understand",
                "template": "analogy_first",
                "content": { ... },
                "source_pages": [42, 43],
                "editable": true
              },
              {
                "type": "vignette",
                "bloom_level": "analyze",
                "content": { ... }
              }
            ]
          }
        ]
      }
    ]
  }
}
```

### 3. Content Templates

Instead of letting the LLM freestyle, we define templates and rotate through them:

| Template | Description | When to Use |
|----------|-------------|-------------|
| `analogy_first` | Real-world analogy → formal definition → worked example | First introduction of an abstract concept |
| `narrative` | Story-based explanation with history/context/anecdotes | Concepts with interesting origins |
| `worked_example` | Full solution walkthrough with cognitive load fading (4 stages) | Procedural/calculation topics |
| `problem_first` | Paradox/question before explanation (leverages generation effect) | Review topics, building confidence |
| `compare_contrast` | Side-by-side comparison of similar concepts | Confusable concepts |
| `socratic` | Question-guided discovery approach | When the learner has enough background |
| `vignette` | Realistic scenario with embedded concepts | Application and analysis testing |
| `visual_walkthrough` | Mental visualization of diagrams/charts | Concepts with strong visual components |
| `error_identification` | Plausible wrong solution → learner finds the mistake | After the learner has seen correct examples |
| `visual_summary` | Concept map showing relationships + blanks to fill | End-of-section synthesis |
| `milestone_assessment` | Chapter-end interactive essay with AI tutor evaluation | Module-level mastery verification |

The LLM is told which template to use. The system rotates templates to prevent monotony.

### 4. Learner Model — IMPLEMENTED (client-side)

Stored client-side (localStorage). Drives concept-based review prioritization and mastery dashboard. Implemented in `_learner_model.js` as a factory function `__lxpCreateLearnerModel(courseSlug)`, included in all templates via `{% include '_learner_model.js' %}`.

```javascript
// Key: lxp_${course_slug}_learner
{
  "concepts": {
    "bond": {
      "attempts": 5, "correct": 4, "accuracy": 0.80,
      "bloom_levels": {
        "remember": { "attempts": 2, "correct": 2 },
        "apply": { "attempts": 1, "correct": 0 }
      },
      "last_seen": "2024-03-15T14:30:00Z",
      "mastery": "progressing"  // new | progressing | mastered | struggling
    }
  },
  "sessions": [
    { "date": "2024-03-10", "elements_completed": 12, "accuracy": 0.75 }
  ],
  "overall": {
    "total_attempts": 45, "total_correct": 36, "accuracy": 0.80,
    "concepts_mastered": 12, "concepts_struggling": 3
  }
}
```

**Mastery classification:** `new` (0 attempts) → `mastered` (>=85% accuracy & >=3 attempts) | `struggling` (<50% & >=2 attempts) | `progressing` (default).

**Integration points:**
- `base.html`: `recordAnswer()` hooked into quiz/flashcard/fill-in-blank handlers
- `index.html`: mastery dashboard (stat cards + per-chapter bars), graph node mastery overlay
- `review.html`: concept-priority sorting (struggling first)
- `mixed_review.html`: session recording on completion

### 5. Editor Mode (Frontend)

A web-based editor that operates on the Course JSON. Features:

- **Content editing** — click any element to rephrase, rewrite, or expand
- **Image insertion** — drag-and-drop images, screenshots, diagrams into any element
- **Element management** — add, remove, reorder, change type (turn a slide into a flashcard)
- **Knowledge graph editor** — visual graph where you can add/remove/adjust concept relationships
- **Preview** — toggle between editor view and learner view
- **Selective regeneration** — right-click a section → "Regenerate with template: vignette"
- **Customization** — course branding, color themes, logo
- **Export** — HTML, SCORM, PDF summary

### 6. Navigation Structure

Instead of a flat 168-element slide deck:

```
Course: Fixed Income
├── Module 1: Bond Fundamentals
│   ├── Section 1.1: What is a Bond? (8 elements)
│   ├── Section 1.2: Bond Pricing (12 elements)
│   ├── Section 1.3: Yield Measures (10 elements)
│   └── Module Quiz (vignette-based)
├── Module 2: Term Structure
│   ├── Section 2.1: Spot Rates (6 elements)
│   ├── ...
│   └── Module Quiz
└── Review: Spaced Repetition Session (due cards from all modules)
```

The learner sees a sidebar with the course structure. They can:
- Navigate to any section
- See completion status per section (green checkmark, yellow in-progress, gray locked)
- See prerequisite locks (section 2.1 locked until 1.2 is complete)
- Access the review session for spaced repetition

## Migration Path

### v0.1 → v0.2 — DONE

- ~~Add chapter → section navigation (sidebar with sections)~~
- ~~Save intermediate JSON to disk~~
- ~~Add source page citations to generated elements~~
- ~~Build concept graph (LLM-assisted concept extraction + prerequisite detection)~~
- ~~Add content template rotation (11 templates)~~
- ~~Deep reading stage (LLM + regex pre-analysis per chapter)~~
- ~~Concept consolidation (entity resolution, topological sort, dependency graph)~~
- ~~Curriculum planner (LLM-powered module ordering with concept context)~~
- ~~Multi-document input (multiple PDFs → unified course)~~
- ~~Streamlit GUI (drag-and-drop PDFs, exercise toggles, LLM config)~~
- ~~9 element types (slides, quizzes, flashcards, fill-in-the-blank, matching, mermaid, concept maps, self-explain, interactive essays)~~

### v0.2 → v0.3 — DONE

- ~~FSRS-5 spaced repetition (review.html + mixed_review.html with inline FSRS algorithm)~~
- ~~Client-side progress tracking (localStorage per element, quiz attempts, confidence ratings)~~
- ~~Dark/light theme toggle (persisted to localStorage)~~
- ~~Cross-chapter navigation (prev/next buttons in chapter headers)~~
- ~~Course header with breadcrumbs and home link~~
- ~~Per-course localStorage keying (slug-based isolation)~~
- ~~Render-only mode (re-render from training_modules.json without LLM)~~
- ~~Functional refactoring (Pydantic v2 discriminated unions, match/case dispatch, reduce/accumulate folds, Protocol-based LLM client)~~
- ~~Two-phase assessment generation (Phase 1 reinforcement target selection → Phase 2 element generation with explicit targets)~~
- ~~Bloom's-level-specific prompt supplements (progressively complex prompting per cognitive level)~~
- ~~Post-generation source verification (rule-based claim extraction + string similarity, zero LLM cost)~~

### v0.3 → v0.5

- ~~Add the learner model (accuracy per concept per Bloom's level)~~ — DONE (_learner_model.js, mastery dashboard, concept-based review prioritization)
- ~~Concept graph visualization in HTML output~~ — DONE (vis-network.js on index page with chapter colors, click-to-navigate, mastery overlay)
- ~~Element-level concept tagging~~ — DONE (concepts_tested threaded from chapter analyses to rendered elements)
- ~~Deterministic element IDs for FSRS tracking~~ — DONE (card_ch01_s00_e00 format)
- ~~"Add to Review" buttons on interactive elements~~ — DONE (base.html)
- ~~Tiered model routing~~ — DONE (Task 05: primary + light model, `LLM_MODEL_LIGHT` env var, `complete_light()` / `complete_structured_light()`)
- ~~Pipeline checkpointing and resume~~ — DONE (Task 10: `--resume` flag, per-stage checkpoints, partial transformation resume, incremental save)
- Adaptive difficulty (adjust Bloom's distribution based on learner model accuracy)
- Implement the editor mode (basic: edit text, add images, reorder elements)
- Wiki-link cross-references between concepts in rendered output
- Reduce element count per section (cap at ~8-10 elements per section)

### v0.5 → v1.0

- Full editor with knowledge graph visualization
- Adaptive difficulty driven by learner model
- Multiple export formats (HTML, SCORM)
- Additional source formats (DOCX, Markdown, URLs)
- Pacing suggestions and study schedules
