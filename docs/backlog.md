# Backlog / Wishlist

Ideas and future features for the learningxp generator.

---

## ~~GUI Launch Menu~~ — DONE (app.py)

Implemented as a Streamlit app. Launch with `streamlit run app.py`.

**Implemented:**
- ~~Drag-and-drop PDF upload (single or multiple files)~~
- ~~Checkboxes to select exercise types (all 9 types)~~
- ~~LLM provider selection (OpenAI / OpenRouter) with API key input~~
- ~~Model picker and temperature slider~~
- ~~"Generate" button with live pipeline progress~~
- ~~Open in browser + ZIP download when done~~
- ~~Chapter selection (process only specific chapters)~~
- ~~Embed images toggle~~

**Remaining nice-to-haves:**
- Template preference overrides (e.g., "prefer worked_example for math-heavy content")
- Bloom's level distribution sliders
- Output directory picker
- Course history — list of previously generated courses with quick re-render

---

## ~~Learner Model~~ — DONE (_learner_model.js + all templates)

Client-side (localStorage) learner state tracking.

**Implemented:**
- ~~Mastered concepts per topic~~ — concept-level mastery classification (new/progressing/mastered/struggling)
- ~~Accuracy history per Bloom's level~~ — per-concept per-Bloom tracking in `_learner_model.js`
- ~~Review schedule (FSRS-5 state per flashcard)~~ — DONE (review.html + mixed_review.html)
- ~~Element-level progress tracking~~ — DONE (localStorage per course slug)
- ~~Session history (duration, elements completed)~~ — `recordSession()` in learner model
- ~~Mastery dashboard on index page~~ — stat cards + per-chapter mastery bars
- ~~Concept-based review prioritization~~ — struggling concepts surfaced first in review.html
- ~~"Add to Review" buttons~~ — on quiz, flashcard, fill-in-blank elements in base.html
- ~~Deterministic element IDs~~ — `card_ch01_s00_e00` format for stable FSRS tracking
- ~~Review nav badge~~ — due card count shown in chapter page header

**Remaining nice-to-haves:**
- Adaptive difficulty (adjust Bloom's distribution based on accuracy per topic)

## Editor Mode

Web-based editor operating on `training_modules.json`:
- Click any element to edit text
- Drag-and-drop image insertion
- Reorder elements within a section
- Change element types (turn a slide into a flashcard)
- Selective regeneration (regenerate one section with a different template)
- Preview mode (learner view vs editor view)

## Wiki-Link Cross-References

Concept names in rendered HTML become clickable links to their first introduction. Uses the concept graph from the deep reader to identify linkable terms.

## SCORM Export

Package the generated course as a SCORM-compliant archive for LMS integration.
