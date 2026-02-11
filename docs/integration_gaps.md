# Integration Gaps

Post-implementation audit. The 10 features were built independently and several don't wire into each other. This document catalogs every place where data is computed but not consumed, where namespaces diverge silently, or where a feature covers only a subset of what it should.

### Status Summary (last audited 2026-02-11)

| Gap | Title | Status |
|-----|-------|--------|
| 1 | Concept Identity Is Fractured | Open |
| 2 | Verification Notes Are Write-Only | Open |
| 3 | Mixed Review Only Handles 2 of 7 Element Types | **Done** — matching + self-explain now cached and rendered |
| 4 | FSRS Ignores In-Chapter Performance | **Done** — `recordScore()` now updates FSRS cards; `_fsrs.js` shared partial created |
| 5 | Bloom's Supplement Missing from Phase 1 | Open |
| 6 | `document_type` Influence Stops at the Planner | Open |
| 7 | `concepts_tested` Is Section-Granularity | Open |
| 8 | Cross-Navigation Between Review Pages | **Done** — Mixed Practice link added to chapter nav; cross-links added to review pages |
| 9 | `chapter_analyses` Is Optional with Silent Degradation | Open |
| 10 | Target Selection Uses Expensive Model | **Done** — now uses `complete_structured_light()` |
| 11 | Streamlit Never Resumes from Checkpoints | **Done** — deterministic temp dirs + resume by default + force-rerun checkbox |
| 12 | CurriculumBlueprint Metadata Ignored by Renderer | **Done** — `course_title`, `course_summary`, `learner_journey` threaded to renderer and displayed on index page |
| 13 | Planner Blueprint Fields Computed but Never Consumed | Open |
| 14 | Table Content Extracted but Only Count Reaches LLM | Open |
| 15 | Image Extraction Disconnected from Content Generation | Open |
| 16 | Passthrough Blueprint Ignores Doc Type Weights | Open |
| 17 | Pre-Analyzer Key Terms Don't Reach Content Designer | Open |
| 18 | `httpx` Is a Dead Dependency | **Done** — removed from `pyproject.toml` |
| 19 | Concept Maps Don't Record Scores | **Done** — `checkConceptMap()` now calls `recordScore()` |
| 20 | Confidence Data Never Reaches the Learner Model | **Done** — `recordAnswer()` accepts confidence + score params |
| 21 | Flashcards Missing Bloom Level at Creation | **Done** — `addToReview()` now stores `bloom` from `getSlideConceptsAndBloom()` |
| 22 | Self-Explain and Milestone Rubric Scores Reduced to Binary | **Done** — continuous `score` param added to `recordAnswer()` |
| 23 | Mixed Practice Exercises Are Stateless | Open |
| 24 | Partially Completed Chapters Excluded from Mixed Practice | Open |
| 25 | FSRS State and Learner Model Are Parallel Systems | **Done** — FSRS stability data merged into mastery display on index page |
| 26 | Matching Exercise Hardcodes Attempts to 1 | **Done** — `wrongAttempts` tracked on container and passed to `recordScore()` |
| 27 | Reinforcement Targets Lost After Phase 2 | Open |
| 28 | `prior_concepts` Threading Loses Structure | Open |
| 29 | All Test Mock Clients Missing Light Methods | **Done** — mocks now implement `complete_light()` + `complete_structured_light()` |
| 30 | `--force` Flag Parsed but Never Wired | **Done** — flag removed from CLI |
| 31 | `app.py` Imports Private `_resolve_llm_provider` | **Done** — private import removed |
| 32 | `.env.example` Incomplete | Open |
| 33 | `app.py` Hardcodes `embed_images=True` | **Done** — no longer hardcoded |
| 34 | `source_book_index` Set but Never Rendered | **Done** — source badge rendered in chapter nav |
| 35 | Passthrough Blueprint Hardcodes `bloom_target` | Open |
| 36 | Multi-Document Type Detection Only Analyzes First Book | Open |
| 37 | Streamlit Missing `document_type` UI Control | Open |
| 38 | `learning_objectives` Not Displayed to Learners | **Done** — rendered as collapsible details block per section |
| 39 | Concept Graph Nodes Link to Chapters, Not Review | **Done** — click popup with "Go to Chapter" + "Review this Concept" actions |

---

## 1. Concept Identity Is Fractured Across the Pipeline

### Problem

Two concept namespaces exist and are never reconciled:

- **Raw names** (`ConceptEntry.name`) — used by: content designer, `prior_concepts` list, `concepts_tested` on HTML elements, learner model localStorage keys, FSRS card concepts
- **Canonical names** (`ResolvedConcept.canonical_name`) — used by: concept graph nodes, topological order in curriculum planner summary

The consolidator (`concept_consolidator.py:49`) deduplicates "Sharpe Ratio" and "SR" into canonical "Sharpe ratio" via `_find_canonical()` (line 229). But the canonical map (`_build_canonical_map()`, line 286) is private and never surfaces outside the module. `consolidate_concepts()` returns a `ConceptGraph` with `ResolvedConcept` objects, but no raw-to-canonical lookup.

Downstream consequences:

- **Learner model** tracks mastery under `"Sharpe Ratio"` (raw, from `concepts_tested`)
- **Concept graph** displays node ID `"Sharpe ratio"` (canonical)
- **Mastery overlay** on `index.html` (line 321) does `mastery[n.id]` where `n.id` is canonical — silent miss for any concept whose raw name differs from its canonical name
- **Curriculum planner** `_build_rich_content_summary()` (line 519) mixes `concept_graph.topological_order` (canonical) with `analysis.concepts[].name` (raw) in the same prompt
- **Pipeline** builds `prior_concepts` from raw names (`pipeline.py:605-606`)

### Fix

Expose the canonical map from `consolidate_concepts()`. Either:
- Add a `canonical_map: dict[str, str]` field to `ConceptGraph` (mapping `raw_name.lower()` → `canonical_name`), or
- Add a `resolve(name: str) -> str` method to `ConceptGraph`

Then use it in:
- `pipeline.py` when building `prior_concepts` — resolve raw names to canonical before threading
- `html_generator.py:228-231` when building `section_concepts` — resolve `concept.name` to canonical before tagging elements
- `curriculum_planner.py:538-593` when building the rich summary — use canonical names consistently

This makes the learner model, concept graph, FSRS cards, and planner all speak the same language.

### Affected Files

- `src/transformation/concept_consolidator.py` — expose `canonical_map` on `ConceptGraph` or make `_build_canonical_map()` public
- `src/transformation/analysis_types.py` — add `canonical_map` field to `ConceptGraph` if going that route
- `src/pipeline.py` — resolve `prior_concepts` through the map
- `src/rendering/html_generator.py` — resolve `section_concepts` through the map
- `src/transformation/curriculum_planner.py` — use canonical names in `_build_rich_content_summary()`

---

## 2. Verification Notes Are Write-Only

### Problem

`_verify_elements()` (`content_designer.py:225`) runs on every section, extracts claims, checks them against source text, and stores warnings in `TrainingSection.verification_notes` (`types.py:408`). These serialize correctly to `training_modules.json`.

But `_build_sections_data()` (`html_generator.py:253-258`) builds section dicts with only: `title`, `source_pages`, `element_count`, `start_index`. It never reads `section.verification_notes`. No template renders them. A course author has zero visibility into flagged content in the actual HTML output.

### Fix

1. In `_build_sections_data()` (`html_generator.py:253-258`), add `verification_notes` to the section dict:
   ```python
   section_data = {
       "title": section.title,
       "source_pages": section.source_pages,
       "element_count": len(section_elements),
       "start_index": offset,
       "verification_notes": section.verification_notes,
   }
   ```

2. In `base.html`, render verification notes as a collapsible warning banner at the top of each section that has them. Only visible if `section.verification_notes` is non-empty. Style as a subtle amber/yellow notice — these are for course authors reviewing the output, not for learners.

### Affected Files

- `src/rendering/html_generator.py` — pass `verification_notes` in section dict
- `src/rendering/templates/base.html` — render warning banner per section
- `src/rendering/templates/styles.css` — style the verification banner

---

## 3. Mixed Review Only Handles 2 of 7 Element Types

### Problem

`cacheExercisesForReview()` in `base.html` (lines 1102-1173) caches three types:
- Flashcards → stored in flashcards localStorage (used by FSRS `review.html`)
- Quizzes → stored in exercises array as `type: 'quiz'`
- Fill-in-the-blank → stored in exercises array as `type: 'fitb'`

`mixed_review.html` (lines 189-193) renders only `quiz` and `fitb`.

Absent from mixed practice: **self-explain, matching, concept map, interactive essay**. Self-explain and matching are interactive assessment types that are pedagogically valuable in review. The "Add to Review" button in `base.html` is also missing from matching, self-explain, and concept map elements.

### Fix

1. In `cacheExercisesForReview()` (`base.html:1102-1173`), add caching for:
   - **Matching** — extract pairs data, store as `type: 'matching'`
   - **Self-explain** — extract prompt and key points, store as `type: 'self_explain'`

2. In `mixed_review.html`, add render functions:
   - `renderMatching(card, ex)` — drag-and-drop or click-to-match UI
   - `renderSelfExplain(card, ex)` — prompt + textarea + reveal key points

3. Add "Add to Review" buttons to matching and self-explain elements in `base.html`.

Concept maps, mermaid diagrams, interactive essays, and slides are structural/visual — reasonable to exclude from mixed review.

### Affected Files

- `src/rendering/templates/base.html` — extend `cacheExercisesForReview()`, add review buttons
- `src/rendering/templates/mixed_review.html` — add `renderMatching()` and `renderSelfExplain()`
- `src/rendering/templates/styles.css` — styles for new review element types

---

## 4. FSRS Ignores In-Chapter Performance

### Problem

When a learner answers a quiz in the chapter view, `recordScore()` (`base.html:590`) updates:
1. The learner model (`learnerModel.recordAnswer()`, line 595)
2. Section mastery progress (lines 598-631)

But it does NOT update FSRS scheduling data. The FSRS card's `difficulty`, `stability`, `lastReview`, `nextReview`, `reps`, `lapses` are only modified in `review.html` via `FSRS.scheduleCard()` (line 298). So:

- Learner nails a concept in-chapter → FSRS still shows it as "due" in review
- The bridge is one-way: `review.html` → learner model (yes), chapter quiz → FSRS (no)

### Fix

When `recordScore()` fires for a quiz/flashcard/fill-in-blank in the chapter view, also update the corresponding FSRS card if it exists in localStorage:

```javascript
// In recordScore(), after learnerModel.recordAnswer():
var elementId = slideEl.getAttribute('data-element-id');
var flashcards = JSON.parse(localStorage.getItem(flashcardsKey) || '{}');
if (flashcards[elementId]) {
    var grade = isCorrect ? 3 : 1;  // "good" or "again"
    flashcards[elementId] = FSRS.scheduleCard(flashcards[elementId], grade);
    localStorage.setItem(flashcardsKey, JSON.stringify(flashcards));
}
```

This requires the FSRS scheduling function to be available in `base.html`. Currently it's only defined in `review.html`. Extract it into a shared includable partial (like `_learner_model.js`).

### Affected Files

- `src/rendering/templates/base.html` — update FSRS card in `recordScore()`
- `src/rendering/templates/review.html` — extract FSRS scheduling into `_fsrs.js` partial
- `src/rendering/templates/_fsrs.js` — new shared partial with `FSRS.scheduleCard()`
- `src/rendering/templates/mixed_review.html` — include `_fsrs.js` if not already

---

## 5. Bloom's Supplement Missing from Phase 1 Target Selection

### Problem

Two-phase assessment (Task 01) runs:
- **Phase 1** — target selection using `TARGET_SELECTION_PROMPT` (`content_designer.py:267`), a static prompt with no Bloom's awareness
- **Phase 2** — element generation using `SYSTEM_PROMPT + BLOOM_PROMPT_SUPPLEMENTS[bloom_target]` (`content_designer.py:328-329`)

Phase 1 identifies reinforcement targets with self-assigned bloom levels (`ReinforcementTarget.bloom_level`), but the LLM is not primed with the planner's `bloom_target`. A section targeted at "analyze" may get "remember"-level targets from Phase 1. Phase 2 compensates partially through the `bloom_target` in the user prompt, but the targets themselves may already be misaligned.

### Fix

Pass `bloom_target` to `_select_reinforcement_targets()` and append a short Bloom's focus hint to `TARGET_SELECTION_PROMPT`:

```python
def _select_reinforcement_targets(
    section, chapter_title, client, section_concepts=None,
    bloom_target=None,  # NEW
):
    bloom_hint = ""
    if bloom_target:
        bloom_hint = f"\n\nThe curriculum targets Bloom's level: {bloom_target}. "
        "Prioritize targets that naturally test at this level or above."
    effective_prompt = TARGET_SELECTION_PROMPT + bloom_hint
    ...
```

Then in `_transform_section()` (line 305-307), pass `bloom_target` through.

### Affected Files

- `src/transformation/content_designer.py` — add `bloom_target` param to `_select_reinforcement_targets()`, append hint

---

## 6. `document_type` Influence Stops at the Planner

### Problem

`detect_document_type()` (`content_pre_analyzer.py`) runs, produces `"quantitative"` / `"narrative"` / etc., and passes it to `plan_curriculum()` / `plan_multi_document_curriculum()` (`curriculum_planner.py:166-174, 321-329`), which adjusts template weight guidance.

But:
- `transform_chapter()` (`pipeline.py:592-597`) never receives `document_type`
- `_transform_section()` (`content_designer.py:283-294`) has no `document_type` parameter
- The content generation system prompt does not adapt for quantitative vs narrative documents
- `document_type` is not persisted to any checkpoint file — recomputed every run

The planner indirectly encodes document_type influence via template choices in the blueprint, but the content designer cannot adapt its prose style (e.g., "emphasize worked examples" for quantitative, "use case studies" for narrative).

### Fix

1. Thread `document_type` from `pipeline.py` through `transform_chapter()` → `_transform_section()`.
2. In `_transform_section()`, append a short document-type hint to the system prompt (similar to Bloom's supplements):
   ```python
   DOC_TYPE_HINTS = {
       "quantitative": "\nThis is quantitative/mathematical content. Prioritize worked numerical examples, step-by-step calculations, and formula derivations.",
       "narrative": "\nThis is narrative/conceptual content. Prioritize analogies, case studies, and compare/contrast exercises.",
       ...
   }
   ```
3. Save `document_type` to `curriculum_blueprint.json` so it survives checkpoint resume.

### Affected Files

- `src/pipeline.py` — thread `document_type` to `transform_chapter()`, persist in blueprint checkpoint
- `src/transformation/content_designer.py` — accept and use `document_type` in `_transform_section()`
- `src/transformation/prompts.py` — add `DOC_TYPE_HINTS` dict

---

## 7. `concepts_tested` Is Section-Granularity, Not Element-Granularity

### Problem

In `_build_sections_data()` (`html_generator.py:227-248`), concept tagging works like this:

```python
# line 228-231: build section-level concept map
for concept in chapter_analysis.concepts:
    section_concepts.setdefault(concept.section_title, []).append(concept.name)

# line 247-248: all elements in a section get the same list
"concepts_tested": concepts,  # where concepts = section_concepts.get(section.title, [])
```

A quiz testing "Bond Duration" and a flashcard testing "Yield Curve" in the same section both get tagged `["Bond Duration", "Yield Curve"]`. Answering the duration quiz incorrectly marks *both* concepts as having a wrong answer in the learner model.

### Fix

Element-level concept tagging requires the LLM to tag each element during generation (which it already partially does — `InteractiveEssayElement` has `concepts_tested`, `QuizElement` explanations reference concepts). A non-LLM approach:

1. For each element, extract key terms from its text content (question text, answer text, flashcard front/back).
2. Match extracted terms against the section's concept list using simple substring/keyword overlap.
3. Assign only the matched concepts to that element's `concepts_tested`.

This is a heuristic but strictly better than the current section-level assignment. Implement as a `_tag_element_concepts(element: dict, section_concepts: list[str]) -> list[str]` function in `html_generator.py`.

### Affected Files

- `src/rendering/html_generator.py` — add `_tag_element_concepts()`, call it per-element instead of per-section assignment

---

## 8. Cross-Navigation Between Review Pages Missing

### Problem

- Review (FSRS) has no link to Mixed Review
- Mixed Review has no link to Review (FSRS)
- Chapter pages link to Review but not Mixed Review
- Review/Mixed Review show "Chapter N" as text, not as a navigable link

### Fix

1. Add a secondary nav link in `review.html` header area: "Switch to Mixed Practice →" linking to `mixed_review.html`
2. Add a secondary nav link in `mixed_review.html` header area: "Switch to Spaced Review →" linking to `review.html`
3. In `base.html`, add a mixed review link near the existing review badge (line 36)
4. In review pages, make "Chapter N" text a link to `chapter_NN.html`

### Affected Files

- `src/rendering/templates/review.html` — add cross-link to mixed review
- `src/rendering/templates/mixed_review.html` — add cross-link to FSRS review
- `src/rendering/templates/base.html` — add mixed review nav link

---

## 9. `chapter_analyses` Is Optional with Silent Degradation

### Problem

`render_course()` (`html_generator.py:55-62`) accepts `chapter_analyses: list[ChapterAnalysis] | None = None`. When `None`:
- All `concepts_tested` arrays are empty `[]`
- Learner model `recordAnswer()` becomes a no-op (line 47 of `_learner_model.js`: `if (!concepts || concepts.length === 0) return;`)
- Mastery dashboard stays hidden (never has data)
- Concept graph mastery overlay does nothing
- Review prioritization has no concept data to prioritize on

The `rerender_from_json()` path does load analyses, but any other caller could easily forget. There is no warning when this happens.

**Known caller that forgets:** `app.py:244-249` — the post-filter re-render path. When a user toggles exercise types off in the Streamlit GUI, `app.py` re-calls `render_course()` without `concept_graph` or `chapter_analyses`. The concept graph visualization, mastery dashboard, and concept-based review are silently lost in the re-rendered output.

### Fix

Log a warning when `chapter_analyses` is `None` or empty in `render_course()`:

```python
if not chapter_analyses:
    logger.warning(
        "render_course() called without chapter_analyses — "
        "learner model, mastery dashboard, and concept-based review will be inert"
    )
```

Consider making it a required parameter (remove the default `None`) to force callers to be explicit. The `rerender_from_json()` path already provides it; `run_pipeline()` already provides it. Any new caller should too.

### Affected Files

- `src/rendering/html_generator.py` — add warning log or make parameter required

---

## 10. ~~Target Selection Uses Expensive Model Instead of Light Model~~ — FIXED

Target selection now uses `client.complete_structured_light()` (`content_designer.py:354`).

---

## 11. Streamlit Never Resumes from Checkpoints

### Problem

`run_pipeline()` accepts `resume: bool = False` (`pipeline.py:99`). The CLI exposes `--resume` (`main.py:44`). Streamlit always calls:

```python
index_path = run_pipeline(config, chapter_number=ch_num)
```

No `resume=True`. Worse, Streamlit writes uploads to `tempfile.mkdtemp()` (`app.py:190`), so even if resume were passed, the checkpoint files from a previous run are in a different temp directory and unreachable.

If a Streamlit run fails at chapter 8 of 15, the user starts over from scratch — re-extracting, re-analyzing, re-planning, and re-transforming all 7 completed chapters.

### Fix

1. Use a deterministic output directory keyed on uploaded file names (e.g., hash of sorted filenames) instead of a random temp dir. This makes checkpoints persist across runs.
2. Pass `resume=True` to `run_pipeline()` by default in the Streamlit path.
3. Add a "Force Re-run" checkbox in the sidebar that passes `resume=False` when checked.

```python
# Deterministic working dir based on uploaded filenames
import hashlib
slug = hashlib.md5("_".join(sorted(uf.name for uf in uploaded_files)).encode()).hexdigest()[:12]
tmp_dir = Path(tempfile.gettempdir()) / f"lxpgen_{slug}"
tmp_dir.mkdir(exist_ok=True)
```

### Affected Files

- `app.py` — deterministic temp dir, pass `resume=True`, add force re-run option

---

## 12. ~~CurriculumBlueprint Metadata Ignored by Renderer~~ — FIXED

`render_course()` now accepts `course_title`, `course_summary`, and `learner_journey` parameters. Blueprint values are used when available, with `_derive_course_title()` as fallback. Course metadata is persisted to `course_meta.json` and rendered on the index page.

---

## 13. Planner Blueprint Fields Computed but Never Consumed

### Problem

The curriculum planner produces per-section and per-module metadata that the content designer never reads:

- **`SectionBlueprint.prerequisites`** (`types.py:355-358`): The planner marks "section B requires section A". `content_designer.py` receives the `ModuleBlueprint` but never inspects `prerequisites` on any section. No ordering enforcement, no cross-reference hints to the LLM.

- **`SectionBlueprint.rationale`** (`types.py:359-362`): The planner explains why it chose each template (e.g., "This section introduces key terminology, so analogy_first helps build intuition."). Never passed to the content generation prompt. The LLM that generates content doesn't know *why* it was given a particular template.

- **`ModuleBlueprint.summary`** (`types.py:377`): A 2-3 sentence module summary. Never used in the content designer prompt or the HTML output.

These fields cost LLM tokens to produce and represent genuine pedagogical reasoning that's discarded.

### Fix

1. Pass `section_bp.rationale` to `build_section_prompt()` as a template rationale hint:
   ```
   ### Why This Template:
   {rationale}
   ```
2. Pass `module_bp.summary` to `transform_chapter()` and include it as chapter context in each section prompt.
3. For `prerequisites`: when building `prior_sections` in `_fold_transform_sections()`, check if the current section has prerequisites that haven't been covered yet — log a warning if ordering is violated.

### Affected Files

- `src/transformation/content_designer.py` — thread `rationale` and `summary` into prompts
- `src/transformation/prompts.py` — add rationale block to `build_section_prompt()`

---

## 14. Table Content Extracted but Only Count Reaches the LLM

### Problem

The extraction pipeline stores full table data — headers, rows, cell values — in `Table` dataclass (`extraction/types.py:34-46`). Tables are attached to sections. A financial textbook table with returns, volatilities, and correlations is fully parsed.

But `content_designer.py:315` passes only the count:

```python
table_count=len(section.tables),
```

And `build_section_prompt()` (`prompts.py:542`) renders it as:

```
This section contains 3 table(s).
```

The LLM knows tables exist but cannot see their content. For quantitative documents, tables often contain the most important data — the worked examples, comparison matrices, and reference values that the LLM needs to generate accurate quizzes and worked examples.

### Fix

1. Pass `section.tables` to `build_section_prompt()`.
2. Format tables as markdown in the prompt (headers + rows), capped at a reasonable size:
   ```python
   if tables:
       for i, table in enumerate(tables[:3]):  # Cap at 3 tables
           lines = ["| " + " | ".join(table.headers) + " |"]
           lines.append("| " + " | ".join("---" for _ in table.headers) + " |")
           for row in table.rows[:10]:  # Cap at 10 rows
               lines.append("| " + " | ".join(row) + " |")
           table_block += f"\n\nTable {i+1} (page {table.page}):\n" + "\n".join(lines)
   ```

This is particularly high-value for quantitative documents where table data directly feeds into quiz questions and worked examples.

### Affected Files

- `src/transformation/content_designer.py` — pass `section.tables` through
- `src/transformation/prompts.py` — format tables as markdown in `build_section_prompt()`

---

## 15. Image Extraction Disconnected from Content Generation

### Problem

The PDF parser extracts images with full metadata — `ImageRef` has `path`, `page`, `caption`, `bbox` (`extraction/types.py:17-31`). Images are saved to disk in the `extracted/images/` directory.

The content designer tells the LLM how many images exist (`image_count`), but:

- The LLM cannot see images (text-only API calls)
- `Slide.image_path` is populated by the LLM, which **guesses** at paths it cannot verify
- The actual extracted `ImageRef.path` values from the parser never connect to what the LLM writes in `image_path`
- `ImageRef.caption` (detected from the PDF) is never sent to the LLM as context

The result: the LLM may generate `image_path: "images/figure_3.png"` while the actual extracted file is `images/img_p42_0.png`. The base64 encoding in `_encode_image_base64()` then fails silently (`html_generator.py:546-547` logs a warning and returns `None`).

### Fix

1. Pass `section.images` metadata to `build_section_prompt()` — not the image bytes, but the captions and paths:
   ```
   ### Images in this section:
   - images/img_p42_0.png (page 42): "Figure 3.1: Efficient Frontier"
   - images/img_p43_0.png (page 43): "Figure 3.2: Capital Market Line"
   ```
2. The LLM can then reference actual paths and use captions as context for generating slide content.
3. Alternatively, assign images to slides post-generation by matching page ranges — a slide about pages 42-43 gets images extracted from those pages, regardless of what the LLM wrote.

### Affected Files

- `src/transformation/prompts.py` — include image metadata in `build_section_prompt()`
- `src/transformation/content_designer.py` — pass `section.images` through
- Optionally: `src/rendering/html_generator.py` — post-hoc image assignment by page range

---

## 16. Passthrough Blueprint Ignores Document Type Template Weights

### Problem

`content_pre_analyzer.py:44-95` defines `DOCUMENT_TYPE_TEMPLATE_WEIGHTS` — precise per-document-type distributions (e.g., quantitative → 30% worked_example, 20% problem_first). These weights are formatted as guidance text for the LLM planner prompt (`format_document_type_guidance()`).

When the LLM planner **fails**, both `_passthrough_blueprint()` (`curriculum_planner.py:664-700`) and `_passthrough_multi_doc_blueprint()` (`curriculum_planner.py:415-454`) fall back to blind template rotation:

```python
template = _AVAILABLE_TEMPLATES[template_idx % len(_AVAILABLE_TEMPLATES)]
template_idx += 1
```

A math textbook gets the same template cycle as a history book. The document type information is available (it's computed before planning) but never reaches the fallback path.

Additionally, `DOCUMENT_TYPE_TEMPLATE_WEIGHTS` itself is incomplete: `visual_summary` and `milestone_assessment` exist in `TEMPLATE_DESCRIPTIONS` (valid templates the planner can assign) but are **absent from every entry** in the weights dict. No document type will ever weight toward these templates in the planner prompt guidance. They can still be assigned by the LLM, but the document profile is blind to them.

### Fix

Accept `document_type` in the passthrough functions and sample templates from the weight distribution:

```python
import random

def _passthrough_blueprint(book: Book, document_type: str = "mixed") -> CurriculumBlueprint:
    weights = DOCUMENT_TYPE_TEMPLATE_WEIGHTS.get(document_type, DOCUMENT_TYPE_TEMPLATE_WEIGHTS["mixed"])
    templates = list(weights.keys())
    probs = list(weights.values())
    rng = random.Random(book.title)  # deterministic
    ...
    for section in chapter.sections:
        template = rng.choices(templates, weights=probs, k=1)[0]
```

### Affected Files

- `src/transformation/curriculum_planner.py` — add `document_type` param to `_passthrough_blueprint()` and `_passthrough_multi_doc_blueprint()`, sample from weights

---

## 17. Pre-Analyzer Key Terms Don't Reach Content Designer

### Problem

`content_pre_analyzer.py:377-402` extracts key terms from section text via regex (bold markers, capitalized phrases). These terms are:

1. Stored in `SectionSignals.key_terms`
2. Passed to the deep reader prompt (`deep_reader.py:257`) to help it focus
3. Used in concept consolidator for deduplication overlap (`concept_consolidator.py:256-266`)

But when deep reading **succeeds**, the content designer receives concepts from `ChapterAnalysis` (which has its own `ConceptEntry.key_terms` from the LLM). The pre-analyzer's regex-extracted terms are discarded. If the LLM deep reader misses a term that regex caught, it's lost.

More importantly, when deep reading **fails**, the fallback `ChapterAnalysis` (`deep_reader.py:270-297`) populates `SectionCharacterization` but creates **zero concepts** — the key terms from pre-analysis don't survive into `ConceptEntry` objects. The content designer then has no concept context for that chapter.

### Fix

In `_fallback_analysis()` (`deep_reader.py:270-297`), create `ConceptEntry` objects from the pre-analyzer's key terms:

```python
from src.transformation.analysis_types import ConceptEntry

concepts = []
for s in signals.sections:
    for term in s.key_terms:
        concepts.append(ConceptEntry(
            name=term,
            definition="",
            concept_type="definition",
            section_title=s.section_title,
            importance="supporting",
        ))
```

This ensures the content designer always has at least regex-level concept context, even when the LLM fails.

### Affected Files

- `src/transformation/deep_reader.py` — populate concepts from key terms in `_fallback_analysis()`

---

## 18. ~~`httpx` Is a Dead Dependency~~ — FIXED

Removed from `pyproject.toml`.

---

## 19. Concept Maps Don't Record Scores

> **Status: DONE** — `checkConceptMap()` now calls `recordScore()` at the end of the function.

### Problem (historical)

`checkConceptMap()` previously validated user answers and provided visual feedback (green/red borders), but never called `recordScore()`. This made concept maps invisible to section mastery, learner model tracking, review prioritization, and mixed practice eligibility.

### Fix (applied)

`recordScore()` call added at the end of `checkConceptMap()`:

```javascript
var slideIdx = parseInt(container.closest('.slide').dataset.index);
recordScore(slideIdx, allCorrect, 1, 0, '');
```

### Additional note (2026-02-11)

The concept map SVG renderer was also rewritten: the old circular layout (fixed 400px height, straight-line edges, character-count node sizing) was replaced with a hierarchical layered layout using longest-path layer assignment, barycenter crossing minimization, auto-sized nodes with text wrapping, cubic bezier curved edges, edge labels with background rects, drop shadows, and responsive viewBox-based sizing.

### Affected Files

- `src/rendering/templates/base.html` — `recordScore()` call in `checkConceptMap()`; `renderConceptMap()` rewritten with hierarchical layout

---

## 20. Confidence Data Never Reaches the Learner Model

### Problem

`recordScore()` (`base.html:590`) accepts a `confidence` parameter (1-5 scale from learner). It stores confidence in `ch.scores[slideIndex].confidence` (line 606). But when it calls the learner model:

```javascript
learnerModel.recordAnswer(info.concepts, info.bloom || bloomLevel, isCorrect);
```

Confidence is not passed. `recordAnswer()` (`_learner_model.js:46`) only accepts `(concepts, bloomLevel, isCorrect)` — three parameters, no confidence.

Additionally, hypercorrection detection (`base.html:747-753`) shows a one-time warning when the learner is confident but wrong. This event is never persisted — the learner model can't surface "concepts you were confidently wrong about" for targeted re-testing in review pages.

### Fix

1. Add `confidence` parameter to `recordAnswer()` in `_learner_model.js`:
   ```javascript
   function recordAnswer(concepts, bloomLevel, isCorrect, confidence) {
       // ...
       c.attempts++;
       if (isCorrect) c.correct++;
       if (confidence && confidence >= 4 && !isCorrect) {
           c.hypercorrections = (c.hypercorrections || 0) + 1;
       }
       // ...
   }
   ```

2. Pass confidence from `recordScore()`:
   ```javascript
   learnerModel.recordAnswer(info.concepts, info.bloom || bloomLevel, isCorrect, confidence);
   ```

3. In `review.html`, use `hypercorrections` count in `cardPriority()` to surface high-confidence errors for re-testing.

### Affected Files

- `src/rendering/templates/_learner_model.js` — add `confidence` param, track `hypercorrections`
- `src/rendering/templates/base.html` — pass confidence to `recordAnswer()`
- `src/rendering/templates/review.html` — factor hypercorrections into card priority

---

## 21. Flashcards Missing Bloom Level at Creation

### Problem

`addToReview()` (`base.html:1044-1099`) creates flashcard entries in localStorage. The stored fields (lines 1087-1093):

```javascript
flashcards[elementId] = {
    difficulty: 0, stability: 0, lastReview: null, nextReview: null,
    reps: 0, lapses: 0,
    front: front, back: back,
    chapterNum: parseInt(CHAPTER_NUM), sectionTitle: sectionTitle,
    concepts: concepts
};
```

`bloomLevel` is missing, even though `getSlideConceptsAndBloom()` (which extracts it from slide data attributes) is available in the same scope. `cacheExercisesForReview()` (line 1101) does store `bloom` on cached exercises, making this an inconsistency.

In `review.html`, `rateCard()` (line 307) passes an empty string for bloom level:

```javascript
learnerModel.recordAnswer(concepts, '', isCorrect);
```

This means every flashcard review is recorded without Bloom's level — the per-Bloom breakdown in the learner model is never populated from review sessions.

### Fix

1. In `addToReview()`, extract and store bloom level:
   ```javascript
   var info = getSlideConceptsAndBloom(slideEl);
   flashcards[elementId] = {
       // ...existing fields...
       bloom: info.bloom || ''
   };
   ```

2. In `review.html` `rateCard()`, pass the stored bloom level:
   ```javascript
   learnerModel.recordAnswer(concepts, card.bloom || '', isCorrect);
   ```

### Affected Files

- `src/rendering/templates/base.html` — store `bloom` in `addToReview()`
- `src/rendering/templates/review.html` — pass `card.bloom` in `rateCard()`

---

## 22. Self-Explain and Interactive Essay Rubric Scores Reduced to Binary

### Problem

Self-explain and interactive essay assessments calculate a rubric fraction — e.g., 4 of 6 key points checked = 0.667. This is converted to binary before reaching the learner model:

```javascript
// Self-explain (line 1270)
recordScore(slideIndex, fraction >= 0.7, 1, 0, '');

// Milestone (line 1413)
recordScore(slideIdx, fraction >= 0.7, 1, confidence, bloom);
```

A learner progressing from 40% → 65% → 75% across sessions shows as: wrong → wrong → right. The progression is erased. The learner model's `recordAnswer()` only accepts boolean `isCorrect`, so partial mastery can't be tracked.

### Fix

1. Add an optional `score` parameter (0.0-1.0) to `recordAnswer()` in `_learner_model.js`:
   ```javascript
   function recordAnswer(concepts, bloomLevel, isCorrect, confidence, score) {
       // ...
       c.attempts++;
       if (isCorrect) c.correct++;
       if (typeof score === 'number') {
           c.total_score = (c.total_score || 0) + score;
           c.avg_score = c.total_score / c.attempts;
       }
   }
   ```

2. Pass the fraction from self-explain and interactive essay:
   ```javascript
   recordScore(slideIndex, fraction >= 0.7, 1, 0, '', fraction);
   ```

3. Use `avg_score` in mastery classification as an alternative to pure binary accuracy — a learner averaging 0.65 is meaningfully different from one averaging 0.20.

### Affected Files

- `src/rendering/templates/_learner_model.js` — add optional `score` parameter
- `src/rendering/templates/base.html` — pass fraction through `recordScore()` to learner model

---

## 23. Mixed Practice Exercises Are Stateless Across Sessions

### Problem

`lxp_*_exercises` is a write-once cache populated by `cacheExercisesForReview()` (`base.html:1101-1173`). After a mixed practice session where the learner answers 20 exercises, those exercises are never updated. `mixed_review.html` (lines 172-181) records a session summary to the learner model but never writes back to the exercises array:

```javascript
// Session completion — only learner model updated
learnerModel.recordSession(0, interleaved.length, pct / 100);
return;
```

Next visit, the same exercises appear fresh with no indication they've been practiced before. The Bloom-level sorting and interleaving are identical because no state tracks what was previously shown.

### Fix

After each exercise is answered in `mixed_review.html`, update the exercise object in the cached array:

```javascript
// After checkMixedQuiz or checkMixedFitb:
var ex = interleaved[currentIdx];
ex.lastPracticed = Date.now();
ex.practiceCount = (ex.practiceCount || 0) + 1;
ex.lastResult = isCorrect;
try { localStorage.setItem(EXERCISES_KEY, JSON.stringify(exercises)); } catch(e) {}
```

Then use `practiceCount` and `lastResult` when building the interleaved queue — deprioritize recently-practiced exercises, surface ones never practiced or previously failed.

### Affected Files

- `src/rendering/templates/mixed_review.html` — update exercises after answers, use history in queue building

---

## 24. Partially Completed Chapters Excluded from Mixed Practice

### Problem

`mixed_review.html` (lines 96-107) filters exercises to fully completed chapters only:

```javascript
Object.keys(chapters).forEach(function(k) {
    if (chapters[k].completed) completedChapters[parseInt(k)] = true;
});

exercises = exercises.filter(function(ex) {
    return completedChapters[ex.chapter];
});
```

A learner who finished 4 of 5 sections in a chapter gets zero exercises from that chapter in mixed practice. The exercises from completed *sections* within partial chapters are valid review material, but they're excluded.

### Fix

Include exercises from sections the learner has visited, not just fully completed chapters. The progress data already tracks section mastery (`ch.sectionMastery`):

```javascript
var eligibleChapterSections = {};
Object.keys(chapters).forEach(function(k) {
    var ch = chapters[k];
    if (ch.completed) {
        eligibleChapterSections[parseInt(k)] = true;  // all sections
    } else if (ch.sectionMastery) {
        // Include sections that have been attempted
        var attempted = Object.keys(ch.sectionMastery).filter(function(s) {
            return ch.sectionMastery[s].attempted > 0;
        });
        if (attempted.length > 0) eligibleChapterSections[parseInt(k)] = true;
    }
});
```

### Affected Files

- `src/rendering/templates/mixed_review.html` — relax chapter filter to include partially completed chapters

---

## 25. FSRS State and Learner Model Are Parallel Systems

### Problem

Two independent tracking systems exist for the same learning events:

- **FSRS** (`lxp_*_flashcards`): per-card scheduling state — stability, difficulty, next review date, reps, lapses. Only updated in `review.html`.
- **Learner model** (`lxp_*_learner`): per-concept accuracy — attempts, correct, accuracy, bloom breakdown, mastery. Updated from all pages.

They never cross-reference:
- FSRS doesn't know "this card tests a struggling concept — schedule more aggressively"
- Learner model doesn't know "this concept was reviewed 10 times over 2 months with consistent recall"
- The temporal dimension of spaced repetition (stability, interval history) is invisible to the mastery dashboard

### Fix

When the mastery dashboard (`index.html`) displays concept status, augment it with FSRS data:

```javascript
// In index.html, after loading learner model:
var flashcards = JSON.parse(localStorage.getItem(flashcardKey) || '{}');
Object.keys(flashcards).forEach(function(key) {
    var card = flashcards[key];
    (card.concepts || []).forEach(function(concept) {
        var cm = conceptMastery[concept];
        if (cm) {
            cm.fsrs_stability = Math.max(cm.fsrs_stability || 0, card.stability || 0);
            cm.fsrs_reps = (cm.fsrs_reps || 0) + (card.reps || 0);
        }
    });
});
```

In `review.html`, factor learner model mastery into FSRS grade interpretation — e.g., if a concept is "struggling" in the learner model, treat a "Good" rating more conservatively (use grade 2 instead of 3 for FSRS scheduling).

### Affected Files

- `src/rendering/templates/index.html` — merge FSRS data into mastery display
- `src/rendering/templates/review.html` — optionally adjust FSRS grading based on concept mastery

---

## 26. Matching Exercise Hardcodes Attempts to 1

### Problem

`selectMatchItem()` (`base.html:950-1007`) records the matching exercise result only when all pairs are matched (line 991):

```javascript
recordScore(slideIdx, true, 1, 0, '');
```

The `attempts` parameter is hardcoded to `1`, regardless of how many wrong pair selections preceded success. Wrong pair attempts (lines 994-1002) are handled visually (red flash, then reset) but never counted. A learner who tries 8 wrong pairs before getting all 4 correct has `attempts: 1` — identical to a learner who nailed it first try.

Additionally, `isCorrect` is always `true` — the exercise only records when fully completed, never on failure/abandonment.

### Fix

Track wrong attempts and pass accurate count:

```javascript
// At the top of the matching exercise scope:
var matchAttempts = 0;

// In the wrong-pair handler (line 994):
matchAttempts++;

// In the all-matched handler (line 991):
recordScore(slideIdx, true, matchAttempts + 1, 0, '');
```

Store `matchAttempts` as a data attribute on the container so it persists across pair selections.

### Affected Files

- `src/rendering/templates/base.html` — track and pass accurate attempts count in matching exercise

---

## 27. Reinforcement Targets Lost After Phase 2

### Problem

Phase 1 of content design (`content_designer.py:_select_reinforcement_targets()`) identifies `ReinforcementTarget` objects — "this section should reinforce concept X at Bloom level Y, focusing on insight Z." Phase 2 uses these targets in the generation prompt, then discards them.

No field on `TrainingElement` or `TrainingSection` carries the target data through to the renderer. This means:
- The renderer can't know *why* an element was generated
- Element-level concept tagging (Gap 7) must reverse-engineer from section titles instead of using the generation-time mapping
- No audit trail from "planner intended this" to "designer generated this"

This is related to but distinct from Gap 7 — Gap 7 is about the section-level granularity of `concepts_tested` at render time. This gap is about the generation-time intent data being discarded before it can even reach the renderer.

### Fix

1. Add an optional `reinforcement_targets` field to `TrainingSection` in `types.py`:
   ```python
   reinforcement_targets: list[dict] = Field(default_factory=list)
   ```

2. In `_transform_section()`, store the selected targets on the section after generation:
   ```python
   section.reinforcement_targets = [t.model_dump() for t in targets]
   ```

3. In `_build_sections_data()`, use reinforcement targets for finer-grained concept tagging per element (replacing or augmenting the section-title-matching approach from Gap 7).

### Affected Files

- `src/transformation/types.py` — add `reinforcement_targets` field to `TrainingSection`
- `src/transformation/content_designer.py` — store targets on section
- `src/rendering/html_generator.py` — use targets for element-level concept tagging

---

## 28. `prior_concepts` Threading Loses Structure

### Problem

`pipeline._transform_modules()` (line 566-617) builds `prior_concepts` as a flat `list[str]` — just concept names:

```python
new_concepts = [c.name for c in analysis.concepts] if analysis else []
prior_concepts = prior_concepts + new_concepts
```

Only the names flow through. `ConceptEntry` has `definition`, `concept_type`, `importance`, `section_title`, and `key_terms` — all stripped. The content designer receives `prior_concepts` and passes it to `build_section_prompt()` where it's rendered as a plain list in the prompt. The LLM generating content for Chapter 5 knows that "duration" was introduced earlier, but not that it's a core formula concept or what its definition is.

### Fix

Thread `prior_concepts` as `list[dict]` or `list[ConceptEntry]` instead of `list[str]`:

```python
new_concepts = [{"name": c.name, "type": c.concept_type, "importance": c.importance}
                for c in analysis.concepts] if analysis else []
prior_concepts = prior_concepts + new_concepts
```

Update `build_section_prompt()` to render structured prior concepts:
```
### Prior concepts (already taught):
- **Duration** (formula, core): The sensitivity of bond price to yield changes
- **Yield Curve** (definition, supporting): The relationship between yields and maturities
```

### Affected Files

- `src/pipeline.py` — thread structured concept data instead of flat names
- `src/transformation/content_designer.py` — accept structured prior_concepts
- `src/transformation/prompts.py` — render structured prior concepts in prompt

---

## 29. ~~All Test Mock Clients Missing `complete_light()` / `complete_structured_light()`~~ — FIXED

All test mocks now implement `complete_light()` and `complete_structured_light()`, delegating to primary methods.

---

## 30. ~~`--force` Flag Parsed but Never Wired Through~~ — FIXED

The `--force` flag has been removed from `main.py`. Default behavior (no `--resume`) is a full run.

---

## 31. ~~`app.py` Imports Private `_resolve_llm_provider`~~ — FIXED

`app.py` no longer imports the private function.

---

## 32. ~~`.env.example` Incomplete — Documents 3 of 8 Environment Variables~~ — FIXED

`.env.example` now documents all 8 environment variables with defaults and descriptions.

---

## 33. ~~`app.py` Hardcodes `embed_images=True`, Ignores Environment Variable~~ — FIXED

`app.py` no longer hardcodes `embed_images`.

---

## 34. `source_book_index` Set but Never Rendered

### Problem

For multi-document courses, `_passthrough_multi_doc_blueprint()` (`curriculum_planner.py:434,444`) sets `source_book_index` on both `SectionBlueprint` and `ModuleBlueprint` to indicate which PDF each module originated from. The field is defined in `types.py:339-342` (section) and `types.py:373-376` (module).

The pipeline reads it — `_transform_chapter_with_state()` (`pipeline.py:572`) uses `module_bp.source_book_index or 0` to select the correct book for content extraction. But `html_generator.py` never reads `source_book_index` from modules or sections. The rendered HTML has no indication of which source document content came from.

For a multi-document course (e.g., "Quantitative Methods" from PDF 1, "Ethics" from PDF 2), the learner sees a unified course with no source attribution. A course author reviewing the output can't tell which PDF generated which module without inspecting the JSON.

### Fix

1. In `_build_sections_data()` (`html_generator.py`), pass `source_book_index` through to the template context.
2. In `base.html`, render a subtle source attribution badge per module (e.g., "Source: Quantitative Methods Vol. 2") when the course has multiple source documents.
3. The source document title can be derived from `config.input_sources[source_book_index].path.stem`.

### Affected Files

- `src/rendering/html_generator.py` — pass `source_book_index` and source document name to template context
- `src/rendering/templates/base.html` — render source attribution when multi-doc

---

## 35. Passthrough Blueprint Hardcodes `bloom_target="understand"`

### Problem

When the LLM planner fails, both fallback functions assign the same Bloom's level to every section:

- `_passthrough_blueprint()` (`curriculum_planner.py:684`): `bloom_target="understand"`
- `_passthrough_multi_doc_blueprint()` (`curriculum_planner.py:436`): `bloom_target="understand"`

A quantitative section full of formulas and derivations gets "understand" — it should be "apply" or "analyze". A definition-heavy introductory section gets "understand" — which happens to be correct, but only by coincidence.

The deep reader's `SectionCharacterization.dominant_content_type` is available at this point (the analysis runs before planning). A section characterized as `"quantitative"` or `"procedural"` naturally maps to higher Bloom's levels than one characterized as `"conceptual"`.

### Fix

Add a simple mapping from `dominant_content_type` to `bloom_target` in the passthrough functions:

```python
_CONTENT_TYPE_TO_BLOOM = {
    "conceptual": "understand",
    "quantitative": "apply",
    "procedural": "apply",
    "comparative": "analyze",
    "case_study": "analyze",
    "mixed": "understand",
}
```

Accept `chapter_analyses` in the passthrough functions (they're already computed), look up each section's characterization, and assign bloom_target from the mapping.

### Affected Files

- `src/transformation/curriculum_planner.py` — add content-type-to-bloom mapping, accept `chapter_analyses` in passthrough functions

---

## 36. Multi-Document Type Detection Only Analyzes First Book

### Problem

`pipeline.py:148` calls:

```python
document_type = detect_document_type(books[0])
```

For a single-document pipeline this is correct. For multi-document (e.g., a quantitative methods PDF + a narrative ethics PDF), only the first book's type is detected. The second book's characteristics are ignored entirely.

`detect_document_type()` (`content_pre_analyzer.py:437`) accepts a single `Book`. There is no multi-book variant. The detected type feeds into the curriculum planner's `### Document Profile` prompt section, which then biases template selection for the *entire* course — including modules from the second book.

A multi-doc course with "Quantitative Methods" (book 1) + "Ethics and Professional Standards" (book 2) would be typed as `"quantitative"`, biasing ethics modules toward `worked_example` and `problem_first` templates instead of `narrative` and `vignette`.

### Fix

Two options:

1. **Per-book detection**: Call `detect_document_type()` for each book. Pass `document_type` per-module to the planner (keyed by `source_book_index`). The planner prompt gets a per-module document profile instead of a global one.

2. **Aggregate detection**: Create `detect_multi_document_type(books: list[Book]) -> DocumentType` that aggregates scores across all books. If scores are spread across types, return `"mixed"`. Simpler but less precise.

Option 1 is better for heterogeneous corpora. Option 2 is simpler and adequate when multi-doc inputs are typically from the same domain.

### Affected Files

- `src/pipeline.py` — call `detect_document_type()` per book or aggregate
- `src/transformation/content_pre_analyzer.py` — optionally add `detect_multi_document_type()`
- `src/transformation/curriculum_planner.py` — accept per-module document type if using option 1

---

## 37. Streamlit Missing `document_type` UI Control

### Problem

`app.py:216` reads `document_type` from the environment variable:

```python
document_type=os.getenv("DOCUMENT_TYPE", "auto").lower(),
```

There is no Streamlit sidebar widget for it. The user cannot select or override the document type from the GUI. The CLI path (`main.py`) also doesn't expose it as a flag — it's env-var-only.

The Streamlit app has sidebar controls for temperature (`st.slider`), chapter selection, and element type filtering. Document type is a natural fit for a sidebar dropdown — it directly affects template selection and content generation style.

### Fix

Add a `st.selectbox` in the Streamlit sidebar:

```python
document_type = st.sidebar.selectbox(
    "Document Type",
    ["auto", "quantitative", "narrative", "procedural", "analytical", "regulatory", "mixed"],
    index=0,
    help="Auto-detect from content, or manually override to bias template selection.",
)
```

Pass the selected value to the `Config` constructor instead of reading from env var.

### Affected Files

- `app.py` — add `st.selectbox` for document type, wire to Config

---

## 38. `learning_objectives` Not Displayed to Learners

### Problem

The curriculum planner generates per-section `learning_objectives` on `SectionBlueprint` (`types.py:343-346`). These are passed to the content designer (`content_designer.py:170`) and used in the LLM generation prompt (`prompts.py:557-559`) to guide what content is produced.

But the rendered HTML never shows objectives to the learner. `html_generator.py` does not read `learning_objectives` from blueprint data — no grep match in the renderer or any template file.

Instructional design research consistently shows that displaying learning objectives upfront improves retention (advance organizers). The data is generated, the LLM cost is already paid, and the objectives describe exactly what each section teaches — they just aren't surfaced.

### Fix

1. Thread `learning_objectives` from `SectionBlueprint` through the content designer to `TrainingSection` (add a field) or pass them directly from the blueprint to the renderer.
2. In `base.html`, render objectives as a collapsible "Learning Objectives" block at the top of each section:
   ```html
   {% if section.learning_objectives %}
   <details class="learning-objectives" open>
     <summary>Learning Objectives</summary>
     <ul>
       {% for obj in section.learning_objectives %}
       <li>{{ obj }}</li>
       {% endfor %}
     </ul>
   </details>
   {% endif %}
   ```

### Affected Files

- `src/transformation/types.py` — add `learning_objectives: list[str]` to `TrainingSection`
- `src/transformation/content_designer.py` — pass objectives through to `TrainingSection`
- `src/rendering/html_generator.py` — include objectives in section template context
- `src/rendering/templates/base.html` — render objectives block
- `src/rendering/templates/styles.css` — style objectives block

---

## 39. Concept Graph Nodes Link to Chapters, Not Review

### Problem

The interactive concept graph on `index.html` (`line 304-313`) handles node clicks:

```javascript
network.on('click', function(params) {
    if (params.nodes.length > 0) {
        var nodeId = params.nodes[0];
        var node = graphData.nodes.find(function(n) { return n.id === nodeId; });
        if (node && node.group) {
            window.location.href = 'chapter_' + String(node.group).padStart(2, '0') + '.html';
        }
    }
});
```

Clicking a concept node navigates to the chapter where it's introduced. This makes sense for initial learning, but there's no way to trigger a review session focused on a specific concept from the graph. A learner who sees a "struggling" (red-bordered) concept node on the mastery dashboard has to:
1. Navigate to `review.html` or `mixed_review.html`
2. Hope the concept shows up in the queue
3. No guarantee — review order is global, not concept-filtered

The concept graph knows mastery status (node borders colored by mastery), and the review pages know concept-based prioritization. But there's no bridge from "click a struggling concept" to "practice that concept."

### Fix

Add a context menu or secondary action (e.g., right-click, or a popup with two options) on graph nodes:

```javascript
network.on('click', function(params) {
    if (params.nodes.length > 0) {
        var nodeId = params.nodes[0];
        var node = graphData.nodes.find(function(n) { return n.id === nodeId; });
        if (node) {
            showNodePopup(node, {
                "Go to Chapter": 'chapter_' + String(node.group).padStart(2, '0') + '.html',
                "Review this Concept": 'review.html?concept=' + encodeURIComponent(nodeId)
            });
        }
    }
});
```

In `review.html`, read the `concept` query parameter and filter/prioritize cards tagged with that concept.

### Affected Files

- `src/rendering/templates/index.html` — add popup/context menu on graph node click
- `src/rendering/templates/review.html` — read `?concept=` query param, filter cards

---

## Execution Order

These are largely independent, but some have natural dependencies:

1. **Gap 1 (concept identity)** should be done first — it affects gaps 7, 9, and 28.
2. **Gap 4 (FSRS in chapter)** requires extracting FSRS into a shared partial, which gap 3 (mixed review element types) also benefits from.
3. ~~**Gap 10 (light model)**~~ — FIXED
4. ~~**Gap 18 (dead dependency)**~~ — FIXED
5. **Gap 19 (concept map scores)** is a one-line fix — do it immediately.
6. **Gap 26 (matching attempts)** is a small fix — do it immediately.
7. ~~**Gap 12 (blueprint metadata)**~~ — FIXED. **Gap 13 (planner fields)** remains — planner `rationale`, `prerequisites`, and `summary` still unused by content designer.
8. **Gap 14 (tables)** and **Gap 15 (images)** are both about extraction data not reaching the LLM prompt — can be done together.
9. **Gap 20 (confidence)**, **Gap 21 (bloom on flashcards)**, and **Gap 22 (rubric scores)** all modify `_learner_model.js` `recordAnswer()` — do them together to avoid repeated refactoring.
10. **Gap 27 (reinforcement targets)** feeds into **Gap 7 (element-level concepts)** — if targets are preserved, element-level tagging becomes much more accurate.
11. **Gap 23 (exercise state)** and **Gap 24 (partial chapters)** both modify `mixed_review.html` — do them together.
12. **Gap 25 (FSRS ↔ learner model)** is a design decision that should come after gaps 20-22 stabilize the learner model interface.
13. ~~**Gap 29 (test mocks)**~~ — FIXED
14. ~~**Gap 30 (--force)**~~ — FIXED, ~~**Gap 31 (private import)**~~ — FIXED, **Gap 32 (.env.example)**, ~~**Gap 33 (embed_images)**~~ — FIXED — remaining infra hygiene is independent, do anytime.
15. **Gap 35 (passthrough bloom)** benefits from the deep reader's analyses being available — pair with Gap 17 (pre-analyzer key terms) since both thread analysis data into the fallback path.
16. **Gap 36 (multi-doc type)** is independent but naturally pairs with Gap 16 (passthrough template weights) — both deal with document type not reaching all code paths.
17. **Gap 37 (Streamlit document_type)** — remaining Streamlit UI gap (~~Gap 33 (embed_images)~~ FIXED).
18. **Gap 38 (learning objectives display)** requires threading data from blueprint → TrainingSection → renderer. Similar pattern to Gap 2 (verification_notes) and Gap 12 (blueprint metadata). Do them together.
19. **Gap 39 (graph → review)** depends on concept identity being consistent (Gap 1) — the concept name in the graph node must match the concept name on review cards.
20. Everything else is orthogonal.

Suggested grouping for parallel work:

- **Agent A (concept pipeline)**: Gap 1, Gap 7, Gap 9, Gap 27, Gap 28
- **Agent B (rendering/templates)**: Gap 2, Gap 3, Gap 8, Gap 34, Gap 38
- **Agent C (FSRS + scheduling)**: Gap 4, Gap 25, Gap 39
- **Agent D (prompting)**: Gap 5, Gap 6, Gap 13, Gap 16, Gap 17, Gap 35
- **Agent E (extraction → LLM wiring)**: Gap 14, Gap 15
- **Agent F (pipeline/infra)**: ~~Gap 10~~ FIXED, Gap 11, ~~Gap 12~~ FIXED, ~~Gap 18~~ FIXED, Gap 36
- **Agent G (learner model + scoring)**: Gap 19, Gap 20, Gap 21, Gap 22, Gap 26
- **Agent H (mixed practice)**: Gap 23, Gap 24
- **Agent I (infra hygiene)**: ~~Gap 29~~ FIXED, ~~Gap 30~~ FIXED, ~~Gap 31~~ FIXED, Gap 32, ~~Gap 33~~ FIXED, Gap 37
- **Quick wins** (do first, <5 min each): ~~Gap 10~~ FIXED, ~~Gap 18~~ FIXED, Gap 19, Gap 26, ~~Gap 29~~ FIXED, ~~Gap 30~~ FIXED, Gap 32, ~~Gap 33~~ FIXED
