# Platform Analysis: What the Best Learning Tools Do Right

Lessons from existing platforms that should inform our design. For each, the 1-2 features that actually matter.

## NotebookLM (Google)

**What it does:** Generates mind maps, quizzes, flashcards, study guides, audio overviews, and briefing docs from uploaded source documents.

**What makes it effective:**
1. **Source grounding** — all generated content cites back to specific source material. You can click any claim and see where it came from in the original document. This dramatically reduces hallucination risk.
2. **Multi-format output** — same content available as audio, mind map, quiz, or study guide. Enables dual coding (verbal + visual) and varied retrieval practice without the user having to do anything.

**What we should steal:** Source grounding with page citations. Every generated element should link back to the original material.

---

## Anki

**What it does:** Spaced repetition flashcard system. Open source. Supports community-shared decks, customizable card templates. Now uses the FSRS algorithm (as of v23.10).

**Evidence:** Medical students using Anki show higher performance on standardized exams (USMLE Step 1). Greater number of matured cards correlates with increased exam performance.

**What makes it effective:**
1. **Algorithm-driven scheduling** — FSRS ensures reviews happen at mathematically optimal intervals, not arbitrary ones. This is the difference between "I'll review this sometime" and "review this in 3 days, then 8 days, then 22 days."
2. **Active recall by default** — every single interaction is a retrieval practice event. There's no passive re-reading mode.

**What we should steal:** FSRS scheduling for all reviewable elements (flashcards, quiz questions, fill-in-the-blank). Use the py-fsrs library.

---

## Duolingo

**What it does:** Language learning with gamification, adaptive difficulty, bite-sized lessons, streaks.

**Evidence:** Users are 3x more likely to return daily with active streaks. Streak wagers produce 14% boost in 14-day retention. Adaptive difficulty achieves 30% improvement in learning outcomes.

**What makes it effective:**
1. **Habit formation through streaks** — daily engagement is the single strongest predictor of learning outcomes. The streak mechanic creates loss aversion (don't want to break the streak).
2. **Adaptive difficulty** — ML-based error prediction keeps learners in their zone of proximal development. If you're getting everything right, it gets harder. Getting things wrong, it gets easier.

**What we should steal:** Adaptive difficulty tracking (accuracy per topic → adjust question difficulty). The streak/gamification stuff is optional but effective for engagement.

---

## Khan Academy + Khanmigo

**What it does:** Structured courses with a visual knowledge map, mastery-based progression, AI tutor (Khanmigo) that guides without giving direct answers.

**Evidence:** Each additional skill practiced to proficiency produces ~0.5 percentage points in learning gains. Students who increase mastered skills by 60+ see ~30 percentage point increases.

**What makes it effective:**
1. **Mastery-based progression** — you cannot advance to derivatives until you've demonstrated proficiency in limits. The knowledge map enforces prerequisites visually and structurally.
2. **Socratic tutoring** — Khanmigo asks guiding questions rather than giving answers. "What do you think the first step would be?" This leverages the generation effect and retrieval practice simultaneously.

**What we should steal:** Prerequisite-enforced knowledge graph with visual display. The learner sees the map, sees where they are, sees what's unlocked and what's locked.

---

## Brilliant.org

**What it does:** Interactive problem-solving courses with visual explanations, simulations, and hands-on exercises.

**What makes it effective:**
1. **Problem-first approach** — learners engage in problem-solving from the very first interaction, before receiving formal instruction. This leverages the generation effect and desirable difficulties.
2. **Interactive visualizations** — manipulable visual elements (drag a curve, adjust parameters, see the result change in real time). This is dual coding at its most effective.

**What we should steal:** The problem-first philosophy. Don't always explain then quiz. Sometimes pose the challenge first, let the learner struggle, then explain.

---

## CFA Institute Learning Ecosystem

**What it does:** Structured curriculum for the CFA exams. Learning outcome statements, vignette-based item sets (Level II: 22 vignettes, 88 MCQs), constructed response (Level III).

**What makes it effective:**
1. **Vignette-based assessment** — a 1-2 page scenario describing a realistic investment situation, followed by 4-6 related questions. Tests application and analysis in context, not isolated recall. Questions require pulling data from different parts of the vignette, demanding synthesis.
2. **Learning outcome statements** — every module starts with explicit, testable objectives. "After completing this section, you should be able to: calculate the forward rate, explain the term structure, compare spot rates and forward rates."

**What we should steal:** Both. Vignette generation as an element type. Learning outcome statements at the start of every section (generated from the content, used to drive exercise generation).

---

## Coursera / edX

**What it does:** Structured weekly modules with video, readings, quizzes, assignments, peer assessment.

**What makes it effective:**
1. **Structured progression with deadlines** — external accountability and pacing prevents procrastination and ensures distributed practice (you can't cram a 10-week course into one night).
2. **Peer assessment** — evaluating others' work is a high-level Bloom's activity (Evaluate) that deepens understanding.

**What we should steal:** Structured pacing suggestions. Not mandatory deadlines, but suggested timelines: "This module is designed for ~2 hours of study. Consider spreading it over 3-4 sessions."

---

## Obsidian

**What it does:** Personal knowledge management with bidirectional wiki-links, graph visualization, local-first markdown files.

**What makes it effective:**
1. **Bidirectional linking** — `[[concept]]` syntax creates automatic backlinks. Every place a concept is mentioned becomes navigable. This mirrors how experts organize knowledge (dense interconnected networks).
2. **Graph visualization** — the visual graph reveals isolated nodes (poorly connected concepts) and clusters. This is metacognitive awareness of your own knowledge structure.

**What we should steal:** Wiki-link style cross-references in generated content. A concept graph view showing all concepts and their connections. Orphan detection (concepts with zero connections = content gaps).

---

## RemNote

**What it does:** Notes with inline flashcard creation, spaced repetition scheduling, knowledge graph.

**What makes it effective:**
1. **Zero-friction flashcard creation** — type `==answer==` inside a note and it becomes a flashcard automatically. No context switching between "note-taking mode" and "flashcard-creation mode."
2. **Unified knowledge base** — notes, flashcards, and knowledge graph are the same data structure. Reviewing a flashcard shows the surrounding note context.

**What we should steal:** The idea that content elements and review elements are the same thing. A slide with a key term should automatically become a flashcard. A quiz question should automatically enter the spaced repetition schedule.

---

## Synthesis: The Feature Stack

Combining the best of each platform, our engine should have:

| Feature | Inspired By | Learning Principle |
|---------|------------|-------------------|
| Source grounding (page citations) | NotebookLM | Anti-hallucination |
| FSRS spaced repetition | Anki | Distributed practice |
| Adaptive difficulty | Duolingo | Zone of proximal development |
| Prerequisite knowledge graph | Khan Academy | Cognitive load management |
| Problem-first exercises | Brilliant.org | Generation effect, desirable difficulties |
| Vignette-based assessment | CFA Institute | Application, analysis (Bloom's 3-4) |
| Learning outcome statements | CFA Institute | Clear objectives drive focused learning |
| Pacing suggestions | Coursera | Distributed practice |
| Wiki-link cross-references | Obsidian | Expert knowledge structure |
| Concept graph visualization | Obsidian | Metacognitive awareness |
| Automatic flashcard creation | RemNote | Zero-friction retrieval practice |
| Editor mode for refinement | Original requirement | Human-in-the-loop quality |

---

## Open-Source Landscape (February 2026)

The following projects were surveyed for reusable patterns, architectural ideas, and gaps our engine can fill. None of them combine the full pipeline (PDF → concept graph → curriculum planning → diverse interactive elements → self-contained HTML) that we do.

### Direct Competitors

**Automated Course Content Generator (ACCG)** — https://github.com/pramodkoujalagi/Automated-Course-Content-Generator
Streamlit app that generates course outlines, content, and quizzes from topic inputs (not PDFs). Uses GPT-3.5 Turbo / GPT-4 Turbo / Meta's LLaMA-3-70B. Outputs PDF and PPT. Useful reference for: Streamlit UI patterns, model selection UI, export formats. Gap: no PDF extraction, no concept graph, no interactive HTML, no pedagogical framework.

**AI-Course Generator (Ashis-Mishra07)** — https://github.com/Ashis-Mishra07/AI-Course_Generator
Full-stack Next.js 14 app with LlamaIndex + vector storage (Pinecone/FAISS). Generates multi-week learning plans with topics, resources, tasks. Gap: web-app-first (not self-contained), no deep content generation, no Bloom's taxonomy, no spaced repetition.

**Open TutorAI** — https://arxiv.org/html/2602.07176 / https://github.com/Open-TutorAi/open-tutor-ai-CE/
Open-source LLM-powered tutoring platform. Key lessons extracted below (see "Engineering Patterns Worth Stealing"). Gap: focused on real-time tutoring, not batch course generation from documents.

**OpenBookLM** — https://github.com/open-biz/OpenBookLM
Open-source NotebookLM alternative focused on audio-based learning from documents. Interactive, AI-generated courses with customizable 3D avatars. Gap: audio-first, no structured interactive elements, no concept graph.

### Component-Level Projects

**PDF Extraction:**
- **Docling** (https://github.com/docling-project/docling) — IBM Research, donated to Linux Foundation AI & Data in 2025. Best-in-class document layout analysis. Worth monitoring as a potential replacement for our PyMuPDF + pdfplumber stack if we need better multi-column or complex layout handling.
- **GROBID** (https://github.com/kermitt2/grobid) — ML-based scholarly document parsing. Strong on academic papers but less general-purpose than our approach.
- **deepdoctection** (https://github.com/deepdoctection/deepdoctection) — Orchestrates layout analysis, OCR, and classification pipelines. More modular than Docling but heavier setup.

**Knowledge Graphs:**
- **knowledge_graph (rahulnyk)** (https://github.com/rahulnyk/knowledge_graph) — Extracts concepts (not entities) from text chunks using a local LLM (Mistral 7B via Ollama). Builds graphs with explicit relationships (weighted W1) and implicit co-occurrence connections (weighted W2). Uses NetworkX for graph algorithms and PyVis for interactive JavaScript visualization. Key insight: **concepts make more meaningful knowledge graphs than named entities** — "pleasant weather in Bangalore" is more useful than just "Bangalore." Uses community detection to identify thematic clusters and node degree for importance scoring.
- **AI Knowledge Graph Generator** (https://github.com/robert-mcdermott/ai-knowledge-graph) — Extracts Subject-Predicate-Object triples from unstructured text using any LLM. Visualizes as interactive graph. Simpler than rahulnyk's approach but the SPO extraction pattern is clean.
- **Mindmap Generator** (https://github.com/Dicklesworthstone/mindmap-generator) — The most architecturally sophisticated of the group. See "Engineering Patterns Worth Stealing" below.

**Bloom's Taxonomy & Question Generation:**
- **AEQG_Blooms_Evaluation_LLMs** (https://github.com/nicyscaria/AEQG_Blooms_Evaluation_LLMs) — Research implementation testing LLM ability to generate questions at different Bloom's levels. See "Research Findings" below.
- **EduQG Dataset** (https://hf.co/papers/2210.06104) — 3,397 MCQs from educational domain with source documents, distractor annotations, and Bloom's level tags (903 questions). Useful as a benchmark for our quiz generation quality.
- **BloomLLM** (https://link.springer.com/chapter/10.1007/978-3-031-72312-4_11) — Fine-tuned LLM (ChatGPT-3.5-turbo on 1,026 questions across 29 topics) specifically for Bloom's-aligned question generation. Demonstrates that fine-tuning on Bloom's-tagged data significantly outperforms prompting alone for higher cognitive levels.

**Spaced Repetition:**
- **Open Spaced Repetition / FSRS** (https://github.com/open-spaced-repetition) — Community behind the FSRS algorithm. Ecosystem is mature: implementations in Python, Rust, TypeScript, Go, Java, C, Swift, Kotlin, Dart. FSRS v5/v6 are the current versions. The `ts-fsrs` TypeScript implementation is what we'd inline in our HTML output. Already integrated into Anki (opt-in since v23.10), Mochi Cards, Logseq, Obsidian (via plugins), RemNote, and others. The `awesome-fsrs` repo (https://github.com/open-spaced-repetition/awesome-fsrs) catalogs all implementations, apps, papers, and datasets.

**NotebookLM Alternatives:**
- **Open Notebook** (https://github.com/lfnovo/open-notebook) — Open-source NotebookLM alternative with 16+ LLM providers (OpenAI, Anthropic, Ollama, Google, LM Studio). MIT license. Note-taking + AI interaction, not course generation.
- **SurfSense** (https://github.com/Decentralised-AI/SurfSense-Open-Source-Alternative-to-NotebookLM) — "NotebookLM + Perplexity combined." 7,600+ GitHub stars. Connected to external sources (search engines, Slack, Notion, YouTube, GitHub). Works with Ollama local LLMs.

### Commercial Competitors (Closed Source)

**Nolej** (https://nolej.io) — Closest commercial analogue to our project. AI engine that automatically generates interactive courses AND builds a global knowledge graph from uploaded documents. Decentralized skills platform. Key differentiator: the knowledge graph is global (shared across all users' content), not per-course. Gap: closed source, SaaS-only.

**Coursebox** (https://www.coursebox.ai/) — AI course creator from files with SCORM export. Converts uploaded files to eLearning with assessments. Free tier available. More polished UI but less pedagogically sophisticated than our approach.

**MiniCourseGenerator** (https://minicoursegenerator.com/) — PDF-to-course with AI. Focuses on "micro-courses" (5-15 minute modules). Simpler than our multi-chapter approach.

---

## Research Findings Relevant to Our Implementation

### LLM Question Generation at Different Bloom's Levels (arXiv:2408.04394)

Tested five LLMs with five prompting strategies (PS1-PS5) of increasing complexity:

| Strategy | Description | When It's Enough |
|----------|-------------|-----------------|
| PS1 | Simple prompt, no extra instructions | Remember-level questions only |
| PS2 | Chain-of-thought with skill definitions | Understand-level questions |
| PS3 | CoT + example questions at that Bloom's level | Apply-level questions |
| PS4 | CoT + skill information + examples | Analyze-level questions |
| PS5 | CoT + skill definition + skill explanation + example questions | Evaluate/Create-level questions |

**Key findings:**
1. LLMs CAN generate high-quality questions at all Bloom's levels, but only when prompted with adequate information. PS1 fails above Understand.
2. There is "significant variance in the performance of the five LLMs" — model selection matters as much as prompt engineering for higher levels.
3. **Automated evaluation is NOT on par with human evaluation** for pedagogical quality. LLM-based grading of question quality misses nuanced issues that human experts catch. Implication: we should not rely solely on LLM self-evaluation for quality assurance.
4. Past systems had "limited abilities to generate questions at higher cognitive levels." The PS5 strategy partially addresses this, but it requires substantially more prompt engineering effort per level.

**What this means for us:** Our `SYSTEM_PROMPT` should not be uniform. Higher Bloom's levels need chain-of-thought reasoning, skill definitions, and example questions in the prompt. This is implemented in Task 02.

### Andy Matuschak's Research on ML-Generated Spaced Repetition Prompts

Source: https://notes.andymatuschak.org/Using_machine_learning_to_generate_good_spaced_repetition_prompts_from_explanatory_text

**What works:**
- GPT-4 can generate usable prompts for **simple declarative knowledge** (facts, definitions, formulas) "usually on the first try" when given four key strategies:
  1. **Separate targeting from prompt writing**: The human (or a separate LLM call) identifies the specific phrases/insights worth reinforcing. The LLM then writes prompts for those pre-identified targets.
  2. **Provide writing principles**: Give the LLM explicit prompt-writing guidelines (what makes a good flashcard).
  3. **Supply reinforcement angle hints**: Tell the LLM what angle to reinforce — "test the mechanism" vs. "test the definition" vs. "test the edge case."
  4. **Include ample context**: More surrounding context always helps.

**What fails:**
- LLMs **lack a "pattern language" for complex conceptual material**. They don't know how to decompose concepts into learnable components through flashcards.
- Generated prompts reinforce **surface-level facts** ("what is said") rather than **meaning** ("what it means, why it matters"). A flashcard testing "What is the Sharpe ratio?" is far less valuable than "Why does the Sharpe ratio penalize upside volatility equally, and when is that a problem?"
- Specific failure modes:
  - **Surface-level reinforcement only**: Definitions and terminology instead of mechanisms and connections.
  - **Missing connection strategies**: Doesn't link new concepts to related ideas or prior knowledge, even when connections are obvious in the source.
  - **Insufficient example generation**: Vague "generate an example" instructions fail. Must be specific: "Generate an example linear system and ask whether it's in echelon form."
  - **Absent purpose-oriented questions**: Doesn't naturally ask WHY a concept matters or what problems it solves.
  - **Weak guidance responsiveness**: "A simple hint doesn't suffice." Models need extensive, detailed guidance rather than high-level strategic hints.

**What this means for us:** Single-call generation (section text → elements) conflates two distinct tasks: identifying what's worth testing and writing the test. Separating them (Task 01) with explicit angle hints addresses Matuschak's core finding. His suggestion of a "pattern language of prompt-writing" maps directly to our template system.

### EduBench: Evaluating LLMs in Education (HuggingFace papers/2505.16160)

A benchmark with 9 educational scenarios and 4,000+ contexts, with 12 evaluation dimensions. Key finding: a relatively small-scale model fine-tuned on educational data can match state-of-the-art large models (DeepSeek V3, Qwen Max) on educational tasks. Implication: if we ever fine-tune a model for our pipeline, the EduBench dataset is a useful training/evaluation resource.

---

## Engineering Patterns Worth Stealing

### From the Mindmap Generator (https://github.com/Dicklesworthstone/mindmap-generator)

The most architecturally relevant open-source project we found. Key patterns:

**1. Document Type Detection System**
Auto-detects whether content is technical, scientific, narrative, business, legal, academic, or instructional. Each type has specialized prompt templates optimized for that domain. Our pipeline currently uses the same prompts regardless of document type. Implemented in Task 08.

**2. Reality Check System**
Verifies every generated node against source material to prevent confabulation. Each claim is checked for grounding in the original text. Our system relies entirely on prompt-level instructions ("Source attribution required") but has no post-generation verification. Implemented in Task 04.

**3. Tiered Processing**
Uses cheaper models for simpler tasks (pre-filtering, structure detection) and expensive models only where quality demands it (deep analysis, synthesis). Our pipeline uses the same model for everything. Implemented in Task 05.

**4. Overlapping Chunk Creation with Boundary Optimization**
Chunks deliberately overlap to preserve context at edges. Boundaries align with natural breaks (sentence endings) rather than fixed character counts. Our `_smart_truncate()` in prompts.py uses an 80/20 head/tail split which is decent but doesn't handle section boundaries as carefully.

**5. Multi-Layered Semantic Redundancy Detection**
Combines textual matching, fuzzy algorithms, token analysis, and LLM-based semantic comparison for deduplication. Our concept consolidator uses exact match, substring, and key-term overlap — missing the semantic layer. Implemented in Task 06.

**6. Non-Linear Exploration Model**
Instead of a linear pipeline, uses parallel processes with feedback loops and heuristic-guided decisions about when to explore deeper vs. halt. Early stopping once sufficient quality emerges. Our pipeline is strictly linear (extract → analyze → plan → transform → render). A non-linear approach could improve quality by allowing the planner to request re-analysis of specific chapters.

**7. Cost Optimization via Similarity Pre-Filtering**
Before expensive LLM-based comparisons, uses computational methods (embedding similarity, Jaccard distance) to filter candidates. Only pairs that pass the cheap filter go through the expensive LLM comparison. Applicable to our concept deduplication (Task 06) and verification (Task 04).

### From Open TutorAI (https://arxiv.org/html/2602.07176)

**1. Four-Layer Prompt Architecture**
Their conversational engine uses a layered system:
- **Layer 1 — Global context**: Overall role, pedagogical philosophy, tone.
- **Layer 2 — Instructional logic**: Session structure, concept scaffolding, reasoning framework selection (deductive, inductive, analogical, causal, abductive).
- **Layer 3 — Adaptive variables**: Learner profile, current concept, mastery level, session position.
- **Layer 4 — Post-interaction management**: Summary generation, progress recording, next-step planning.

Our `SYSTEM_PROMPT` effectively combines layers 1 and 2. Layers 3 and 4 are what the learner model (Task 09) and pipeline checkpointing (Task 10) would add.

**2. Reasoning Framework Selection**
The system selects between deductive (general → specific), inductive (examples → rule), analogical (known → unknown), causal (cause → effect), and abductive (observation → best explanation) reasoning approaches based on the content type. Our template system (analogy_first, worked_example, socratic, etc.) is a similar concept but at a coarser granularity. Their approach suggests we could make our templates more precise about the reasoning mode, not just the structural format.

**3. Structured Learner Profiling**
Onboarding captures goals, preferences, and support needs upfront. Enables individualized AI assistants rather than one-size-fits-all. For our batch-generation context, this maps to: document type detection (Task 08) customizes the pipeline, and the learner model (Task 09) personalizes the experience.

**4. Multi-Dimensional Engagement Tracking**
Tracks behavioral (time per step, completion), emotional (inferred from interaction patterns), and cognitive (accuracy, error patterns) engagement. Their "cognitive engagement score" combines these dimensions. For our client-side learner model, tracking time-per-element + accuracy gives us two of the three dimensions without requiring real-time interaction.

### From the Knowledge Graph Project (https://github.com/rahulnyk/knowledge_graph)

**1. Concepts Over Entities**
Uses LLM to extract "concepts" — contextual phrases like "Pleasant weather in Bangalore" — rather than traditional Named Entity Recognition ("Bangalore"). The author argues concepts create more meaningful graphs because they capture relationships and context, not just proper nouns. Our deep reader already extracts concepts with definitions and key terms, which aligns with this philosophy.

**2. Implicit + Explicit Relationships**
Assigns two weight types: W1 for explicit relationships extracted by the LLM ("A depends on B"), and W2 for implicit connections based on co-occurrence within the same text chunk ("A and B appear together"). Summing and consolidating weights produces a richer graph than explicit relationships alone. Our concept consolidator only uses explicit prerequisite links — adding co-occurrence weighting could reveal connections the LLM missed.

**3. Community Detection for Thematic Clustering**
Uses NetworkX community detection algorithms to identify clusters of related concepts. These clusters map naturally to course modules. Our curriculum planner could use community structure from the concept graph to validate or improve its module groupings.

**4. PyVis for Zero-Dependency Interactive Graphs**
Generates standalone HTML files with interactive force-directed graph layouts. No server required. Directly applicable to our concept graph visualization (Task 07), though we chose vis-network.js for lighter weight.

---

## Gap Analysis: Where We're Unique

No open-source project we found combines all of these in one pipeline:

| Capability | Us | ACCG | Open TutorAI | Mindmap Gen | knowledge_graph |
|-----------|-----|------|-------------|-------------|----------------|
| PDF → structured extraction | Yes | No | No | Yes (text only) | No |
| Concept graph with prerequisites | Yes | No | No | No | Yes |
| Curriculum planning with Bloom's | Yes | No | Partial | No | No |
| 9+ diverse interactive element types | Yes | Quizzes only | Chat only | Mindmaps only | Graph only |
| Evidence-based pedagogy in prompts | Yes | No | Yes | No | No |
| Self-contained HTML output | Yes | PDF/PPT | Web app | HTML | HTML |
| Multi-document unified courses | Yes | No | No | No | No |
| Spaced repetition scheduling | Yes (FSRS-5) | No | No | No | No |
| Learner model with adaptive difficulty | Partial (tracking done, adaptive not yet) | No | Yes | No | No |

The closest commercial product is **Nolej** (concept graph + interactive course generation from documents), but it is closed-source and SaaS-only.

Our primary differentiation as an open-source project: **evidence-based pedagogy baked into the system architecture** (not just prompt engineering), with a complete pipeline from raw documents to self-contained interactive courses.
| Storytelling / anecdotes | Original requirement | Narrative memory, germane load |
