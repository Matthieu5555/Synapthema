# Learning Experience Generator

PDF-to-interactive-training pipeline. Feed in PDFs, get a self-contained HTML course with spaced repetition, concept graphs, and progress tracking.

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Set your LLM provider (pick one)
echo 'OPENAI_KEY=sk-...' > .env
# or
echo 'OPENROUTER_KEY=sk-or-...' > .env

# 3. Run on a PDF
uv run main.py yourbook.pdf
```

Output lands in `output/<slug>/index.html`. Open it in a browser.

## CLI Usage

```bash
# Single PDF
uv run main.py doc.pdf

# Multiple PDFs → unified course
uv run main.py doc1.pdf doc2.pdf doc3.pdf

# Directory of PDFs
uv run main.py --input-dir ./materials/

# Process only chapter 3
uv run main.py doc.pdf --chapter 3

# Resume from last checkpoint (skip completed stages)
uv run main.py doc.pdf --resume

# Re-render HTML from existing JSON (no LLM needed)
uv run main.py doc.pdf --render-only

# Exclude element types from output
uv run main.py doc.pdf --exclude interactive_essay self_explain
```

## Streamlit GUI

```bash
streamlit run app.py
```

Drag-and-drop PDFs, toggle element types, configure LLM settings, and download the result as a ZIP.

## Configuration

All settings via environment variables or `.env` file. See `.env.example` for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_KEY` | — | OpenAI API key (mutually exclusive with `OPENROUTER_KEY`) |
| `OPENROUTER_KEY` | — | OpenRouter API key |
| `LLM_MODEL` | `gpt-5.2` | Primary model for content generation |
| `LLM_MODEL_LIGHT` | same as `LLM_MODEL` | Cheaper model for TOC detection, target selection |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature (0.0-1.0) |
| `LLM_MAX_TOKENS` | `16384` | Max output tokens per LLM call |
| `EMBED_IMAGES` | `true` | Base64-embed images in HTML for portability |
| `DOCUMENT_TYPE` | `auto` | Override auto-detection: `quantitative`, `narrative`, `procedural`, `analytical`, `regulatory`, `mixed` |

## Pipeline Stages

```
PDF(s) → Extract → Book(s)                         ← book_structure.json
                     │
             Deep Reader (LLM per chapter)          ← chapter_analyses.json
                     │
             Concept Consolidator (entity resolution, topological sort)
                     │
             Document Type Detector (regex heuristics)
                     │
             Curriculum Planner (LLM)               ← curriculum_blueprint.json
                     │
             Content Designer (2-phase LLM)         ← training_modules.json
                     │
             HTML Renderer → self-contained course
                               ├── chapter pages
                               ├── index (mastery dashboard, concept graph)
                               ├── review.html (FSRS-5 spaced repetition)
                               └── mixed_review.html (cross-chapter practice)
```

Each stage checkpoints to disk. Use `--resume` to skip completed stages after a crash or interruption.

## Element Types

| Type | Bloom Level | Description |
|------|-------------|-------------|
| `slide` | understand | Teaching content — narrative prose, analogies, worked examples |
| `flashcard` | remember | Key term/definition pairs for spaced repetition |
| `quiz` | apply | MCQs requiring application, not just recall |
| `fill_in_the_blank` | apply | Contextual recall with scaffolding |
| `matching` | apply | Pair related items (concept ↔ example) |
| `mermaid` | understand | Diagrams for processes, workflows, hierarchies |
| `concept_map` | analyze | Relationship decomposition between 5+ concepts |
| `self_explain` | evaluate | Learner generates original explanation |
| `interactive_essay` | evaluate | Chapter-end checkpoint with AI tutor |

Bloom levels are assigned deterministically by element type — the LLM does not choose them.

## Deploying with a Custom LLM Provider

The pipeline uses a `LLMClient` protocol with four methods. Any object implementing these methods can replace the default `OpenAIClient`.

### The Protocol

```python
# src/transformation/llm_client.py

class LLMClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...
    def complete_light(self, system_prompt: str, user_prompt: str) -> str: ...
    def complete_structured(
        self, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> T: ...
    def complete_structured_light(
        self, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> T: ...
```

**`complete`** / **`complete_light`**: Send a system prompt + user prompt, get back raw text. `complete_light` routes to a cheaper model for simple tasks (TOC detection, target selection). Can delegate to `complete` if you only have one model.

**`complete_structured`** / **`complete_structured_light`**: Same as above, but the response must be parsed into a Pydantic model specified by `response_model`. The default implementation uses [Instructor](https://github.com/jxnl/instructor) to constrain the LLM to output valid JSON matching the schema, with automatic retry on validation failure.

### Option A: OpenAI-Compatible Endpoint

If your infrastructure exposes an OpenAI-compatible API (Azure OpenAI, vLLM, LiteLLM, etc.), use the built-in `OpenAIClient` with a custom `base_url`:

```python
from src.transformation.llm_client import OpenAIClient

client = OpenAIClient(
    api_key="your-key",
    model="your-model-id",
    max_tokens=16384,
    temperature=0.3,
    base_url="https://your-internal-endpoint/v1",
    model_light="your-cheaper-model",  # optional
)
```

Or just set environment variables and the pipeline handles it:

```bash
OPENAI_KEY=your-key
LLM_MODEL=your-model-id
# Set the base URL in src/config.py or override resolve_llm_provider()
```

### Option B: Fully Custom Client

If your infrastructure doesn't speak the OpenAI protocol, implement the four methods directly:

```python
from pydantic import TypeAdapter

class YourClient:
    def __init__(self, internal_client):
        self._client = internal_client

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._client.chat(system=system_prompt, user=user_prompt)

    def complete_light(self, system_prompt: str, user_prompt: str) -> str:
        return self.complete(system_prompt, user_prompt)  # or route to cheaper model

    def complete_structured(self, system_prompt, user_prompt, response_model):
        # Ask the LLM to return JSON matching the Pydantic schema
        schema = response_model.model_json_schema()
        prompt = f"{user_prompt}\n\nRespond with JSON matching this schema:\n{schema}"
        raw = self._client.chat(system=system_prompt, user=prompt)
        return TypeAdapter(response_model).validate_json(raw)

    def complete_structured_light(self, system_prompt, user_prompt, response_model):
        return self.complete_structured(system_prompt, user_prompt, response_model)
```

### Wiring It In

The client is created in one place — `src/pipeline.py:run_pipeline()`. Replace the `create_llm_client()` call:

```python
# Before (default OpenAI/OpenRouter):
client = create_llm_client(
    api_key=config.llm_api_key,
    model=config.llm_model,
    ...
)

# After (your client):
client = YourClient(your_internal_llm)
```

No other file needs to change. The client is passed via dependency injection throughout the pipeline.

### Structured Output Requirements

The `complete_structured` method is the critical one. The pipeline sends Pydantic model classes as `response_model` and expects back a validated instance. The models use:

- Discriminated unions (element types dispatched by `element_type` field)
- Nested models (e.g., `QuizElement` contains `Quiz` which contains `list[QuizQuestion]`)
- Field validators (e.g., quiz must have >= 2 options)
- Model validators (e.g., section must have >= 1 slide and >= 1 assessment)

If your LLM doesn't reliably output valid JSON, you'll need retry logic in `complete_structured`. The default `OpenAIClient` uses Instructor for this (sends Pydantic validation errors back to the LLM for self-correction, up to 2 retries).

## Project Structure

```
src/
  config.py                          # Environment-based configuration
  pipeline.py                        # Orchestrator — wires all stages together
  extraction/
    pdf_parser.py                    # PDF → structured Book
    structure_detector.py            # Chapter/section boundary detection
    multi_doc.py                     # Multi-PDF corpus extraction
    types.py                         # Book, Chapter, Section, ImageRef, Table
  transformation/
    llm_client.py                    # LLMClient protocol + OpenAIClient
    deep_reader.py                   # Stage: concept extraction per chapter
    concept_consolidator.py          # Stage: cross-chapter entity resolution
    content_pre_analyzer.py          # Stage: document type detection
    curriculum_planner.py            # Stage: module ordering + template assignment
    content_designer.py              # Stage: 2-phase element generation
    prompts.py                       # All LLM prompt templates
    types.py                         # TrainingModule, TrainingElement, etc.
    analysis_types.py                # ChapterAnalysis, ConceptGraph, etc.
  rendering/
    html_generator.py                # Course JSON → HTML
    templates/                       # Jinja2 templates + CSS + JS
main.py                              # CLI entry point
app.py                               # Streamlit GUI
```

## Testing

```bash
uv run pytest tests/                 # All tests
uv run pytest tests/ -x --tb=short   # Stop on first failure
uv run pytest tests/test_types.py    # Specific test file
```

321 tests, no external services required (all LLM calls mocked).

## Checkpoints and Re-rendering

The pipeline saves intermediate JSON at each stage under `extracted/<slug>/`:

| File | Stage | Contents |
|------|-------|----------|
| `book_structure.json` | Extraction | Chapters, sections, page ranges |
| `chapter_analyses.json` | Deep reading | Concepts, prerequisites, section characterizations |
| `curriculum_blueprint.json` | Planning | Module order, templates, Bloom targets |
| `training_modules.json` | Transformation | All generated training elements |

Use `--render-only` to re-render HTML from `training_modules.json` without any LLM calls. Use `--resume` to skip stages whose checkpoint already exists.

You can manually edit `training_modules.json` to fix content, then re-render.
