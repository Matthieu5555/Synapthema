"""Visualization opportunity detection and generation.

Provides a three-layer pipeline for deciding when a section benefits from
an interactive visualization (explorable explanation) and generating the
HTML/JS code for it:

1. Rule-based pre-filter (free) — keyword/pattern scoring
2. Light model triage (cheap) — structured assessment via Instructor
3. Creative model generation (expensive) — HTML/JS code generation

The pipeline is opt-in (VIZ_ENABLED=true) and designed to fail gracefully:
if any layer fails, the section keeps its regular elements unchanged.
"""

from __future__ import annotations

import logging
import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Discriminator, Field

logger = logging.getLogger(__name__)


# ── Layer 1: Rule-based pre-filter ──────────────────────────────────────────

# Positive signal patterns — each hit scores +1. Need ≥2 to pass.
_VIZ_SIGNAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"as\s+\w+\s+(increases?|decreases?|grows?|shrinks?|rises?|falls?)", re.I),
    re.compile(r"(formula|equation|function)\b", re.I),
    re.compile(r"step\s+\d", re.I),
    re.compile(r"(equilibrium|feedback|stable|unstable)\b", re.I),
    re.compile(r"(compared?\s+to|versus|in\s+contrast)\b", re.I),
    re.compile(r"(trade-?off|relationship\s+between)\b", re.I),
    re.compile(r"(proportional|inversely|exponential|logarithmic)\b", re.I),
    re.compile(r"(curve|graph|plot|distribution|histogram)\b", re.I),
]

_MIN_WORDS_FOR_VIZ = 200


def viz_prefilter(section_text: str) -> bool:
    """Quick heuristic: does this section have visualization potential?

    Returns True if the section has enough length and enough positive
    signal patterns to justify sending to the light model for triage.
    """
    if len(section_text.split()) < _MIN_WORDS_FOR_VIZ:
        return False

    score = sum(1 for pat in _VIZ_SIGNAL_PATTERNS if pat.search(section_text))
    return score >= 2


# ── Layer 2: Light model triage models ──────────────────────────────────────

class NoVisualization(BaseModel):
    """The section does not benefit from an interactive visualization."""

    decision: Literal["skip"] = "skip"
    reason: str = Field(description="Brief reason why visualization is not helpful here")


class VisualizationOpportunity(BaseModel):
    """The section would benefit from an interactive visualization."""

    decision: Literal["visualize"] = "visualize"
    viz_type: Literal[
        "parameter_explorer",
        "process_stepper",
        "comparison",
        "system_dynamics",
        "data_explorer",
    ] = Field(description="Type of visualization that best fits the content")
    concept: str = Field(description="The core concept this visualization illustrates")
    variables: list[str] = Field(
        min_length=2,
        description="Interactive parameters the learner can adjust (min 2)",
    )
    learning_goal: str = Field(
        description="What intuition the learner builds by interacting with this visualization",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence that this visualization would significantly aid learning (0-1)",
    )


VizTriageResult = Annotated[
    Union[NoVisualization, VisualizationOpportunity],
    Discriminator("decision"),
]


# Confidence threshold — only generate visualizations above this score.
VIZ_CONFIDENCE_THRESHOLD = 0.7


# ── Layer 2: Triage prompt ──────────────────────────────────────────────────

VIZ_TRIAGE_SYSTEM_PROMPT = """\
You are an expert instructional designer assessing whether a section of \
educational content would significantly benefit from an interactive \
visualization (explorable explanation).

An interactive visualization is an HTML widget where the learner manipulates \
parameters (sliders, toggles) and observes how a system responds — building \
intuition that text alone cannot deliver.

## When to recommend visualization

GOOD candidates (decision: "visualize"):
- Content with a formula or equation where changing variables produces \
non-obvious effects (e.g. compound interest, price/yield relationship)
- Multi-step processes where the learner benefits from stepping through states
- Systems with feedback loops or equilibrium behavior
- Comparisons where side-by-side parameter exploration reveals trade-offs
- Statistical distributions or data patterns that become clear through interaction

BAD candidates (decision: "skip"):
- Simple definitions or terminology
- Narrative or historical content without quantitative relationships
- Content where a static diagram or bullet list suffices
- Very short or shallow treatments of a topic

## Important constraints

- Generating a visualization is expensive. Only flag content where interaction \
would produce an insight that text cannot achieve.
- You MUST name at least 2 real variables from the source text in the \
"variables" field. If you cannot identify 2+ interactive parameters, choose "skip".
- Set confidence conservatively — 0.7+ means you are quite sure this would help.\
"""


def build_viz_triage_prompt(
    section_title: str,
    section_text: str,
    key_terms: list[str] | None = None,
    concepts: list[str] | None = None,
) -> str:
    """Build the user prompt for visualization opportunity triage."""
    parts = [
        f"## Section: {section_title}\n",
    ]
    if key_terms:
        parts.append(f"**Key terms:** {', '.join(key_terms[:10])}\n")
    if concepts:
        parts.append(f"**Concepts covered:** {', '.join(concepts[:8])}\n")

    # Truncate section text to avoid blowing up context
    text = section_text[:3000]
    if len(section_text) > 3000:
        text += "\n[...truncated...]"
    parts.append(f"\n{text}")
    parts.append(
        "\n\nAssess whether this section would benefit from an interactive "
        "visualization. If yes, specify the type, variables, and learning goal."
    )
    return "\n".join(parts)


# ── Layer 3: Design tokens & generation prompt ────────────────────────────

# CSS snippet the LLM must embed in every generated visualization so it
# matches the course look and feel.  Dark is the default; light overrides
# via prefers-color-scheme media query.
_DESIGN_TOKENS_CSS = """\
:root {
  --bg: #152041;  --surface: #1A2747;  --border: #263359;
  --fg: #F8FAFC;  --fg-muted: #94A3B8;
  --accent: #6DA2F8;  --accent-hover: #5B8FE5;
  --success: #86EFAC;  --error: #F87171;  --warning: #FBBF24;
  --font-body: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --radius: 8px;
}
@media (prefers-color-scheme: light) {
  :root {
    --bg: #F6F7F9;  --surface: #FFFFFF;  --border: #D7DFEA;
    --fg: #152041;  --fg-muted: #64748B;
    --accent: #3C83F6;  --accent-hover: #2563EB;
    --success: #15803D;  --error: #DC2626;  --warning: #D97706;
  }
}\
"""

VIZ_GENERATION_SYSTEM_PROMPT = """\
You are an expert educational visualization developer. You create interactive \
HTML visualizations (explorable explanations) that help learners build intuition \
through parameter manipulation.

## Output requirements

Generate a COMPLETE, SELF-CONTAINED HTML file. The output must be:
- A single HTML document with embedded CSS and JavaScript
- Uses p5.js loaded from CDN: https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.11.3/p5.min.js
- No other external dependencies (no images, no other libraries, no fetch calls)
- No localStorage or sessionStorage access
- Mobile-friendly (touch events via p5.js touch API)

## Error handling

Include this at the top of your script:
```javascript
window.onerror = function(msg, src, line) {
    window.parent.postMessage({type: 'viz-error', message: msg, line: line}, '*');
};
```

And signal ready state after setup:
```javascript
window.parent.postMessage({type: 'viz-ready'}, '*');
```

## Design principles

1. Every interactive parameter should produce an "aha moment" — not just a number changing
2. Annotate directly on the canvas: labels, threshold lines, zone fills, value callouts
3. Hover should show crosshairs and snap to curves with exact value tooltips
4. Use smooth animations for transitions (lerp, not instant jumps)
5. Include a brief title and 1-2 sentence explanation at the top of the visualization
6. Default parameter values should show the most instructive starting state
7. Sliders should have labeled min/max and current value display

## Style — MANDATORY design tokens

You MUST embed the following CSS variables in your HTML and use them for ALL \
colors. This ensures the visualization matches the course design system. \
Do NOT use hardcoded colors — always reference these variables.

```css
{design_tokens}
```

Apply them like this:
- `body {{ background: var(--bg); color: var(--fg); font-family: var(--font-body); }}`
- Canvas background: use the raw hex from `--bg` (p5.js `background()` needs numeric values). \
Read it at runtime: `getComputedStyle(document.documentElement).getPropertyValue('--bg')`
- Curves/data: use `--accent` for primary, `--success`/`--warning`/`--error` for secondary series
- Grid lines / axis: use `--border`
- Labels / annotations: use `--fg` or `--fg-muted`
- Slider controls: style with `--surface`, `--border`, `--accent`
- Canvas should be responsive (use windowWidth/windowHeight from p5.js)

Output ONLY the complete HTML file. No markdown fences, no explanation.\
""".replace("{design_tokens}", _DESIGN_TOKENS_CSS)

# Few-shot example embedded in generation prompts (~600 tokens)
_VIZ_FEW_SHOT_EXAMPLE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sine Wave Explorer</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.11.3/p5.min.js"></script>
<style>
  """ + _DESIGN_TOKENS_CSS + """
  * { margin: 0; box-sizing: border-box; }
  body { font-family: var(--font-body); background: var(--bg); color: var(--fg); padding: 16px; }
  h3 { margin-bottom: 4px; font-size: 1.1em; }
  p { font-size: 0.85em; color: var(--fg-muted); margin-bottom: 12px; }
  .controls { display: flex; gap: 24px; flex-wrap: wrap; margin-top: 12px; }
  .control label { display: block; font-size: 0.8em; margin-bottom: 2px; }
  .control input[type=range] { width: 180px; accent-color: var(--accent); }
  .control .val { font-size: 0.75em; color: var(--fg-muted); }
</style>
</head>
<body>
<h3>Sine Wave Explorer</h3>
<p>Drag the sliders to see how frequency and amplitude shape the wave.</p>
<div id="sketch-container"></div>
<div class="controls">
  <div class="control">
    <label>Frequency: <span id="freq-val" class="val">2.0</span></label>
    <input type="range" id="freq" min="0.5" max="8" step="0.1" value="2">
  </div>
  <div class="control">
    <label>Amplitude: <span id="amp-val" class="val">80</span></label>
    <input type="range" id="amp" min="10" max="150" step="1" value="80">
  </div>
</div>
<script>
window.onerror = function(msg, src, line) {
    window.parent.postMessage({type: 'viz-error', message: msg, line: line}, '*');
};
// Helper: read CSS custom property as p5-compatible color
function cssColor(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
let freq = 2, amp = 80;
function setup() {
    let c = createCanvas(min(windowWidth - 32, 600), 250);
    c.parent('sketch-container');
    document.getElementById('freq').oninput = function() {
        freq = parseFloat(this.value);
        document.getElementById('freq-val').textContent = freq.toFixed(1);
    };
    document.getElementById('amp').oninput = function() {
        amp = parseFloat(this.value);
        document.getElementById('amp-val').textContent = amp.toFixed(0);
    };
    window.parent.postMessage({type: 'viz-ready'}, '*');
}
function draw() {
    background(cssColor('--bg'));
    let midY = height / 2;
    stroke(cssColor('--border')); strokeWeight(1); line(0, midY, width, midY);
    stroke(cssColor('--accent')); strokeWeight(2); noFill();
    beginShape();
    for (let x = 0; x < width; x++) {
        let y = midY + amp * sin(TWO_PI * freq * x / width);
        vertex(x, y);
    }
    endShape();
    if (mouseX >= 0 && mouseX < width && mouseY >= 0 && mouseY < height) {
        let snapY = midY + amp * sin(TWO_PI * freq * mouseX / width);
        stroke(cssColor('--warning')); strokeWeight(1);
        line(mouseX, 0, mouseX, height); line(0, snapY, width, snapY);
        fill(cssColor('--warning')); noStroke(); ellipse(mouseX, snapY, 8);
        fill(cssColor('--fg')); textSize(11); textAlign(LEFT, BOTTOM);
        text('y=' + (amp * sin(TWO_PI * freq * mouseX / width)).toFixed(1), mouseX + 8, snapY - 4);
    }
}
function windowResized() { resizeCanvas(min(windowWidth - 32, 600), 250); }
</script>
</body>
</html>\
"""


def build_viz_generation_prompt(
    opportunity: VisualizationOpportunity,
    section_title: str,
    section_text: str,
) -> str:
    """Build the user prompt for visualization code generation."""
    # Truncate source text — the model knows the domain; it needs scope, not full text
    text_excerpt = section_text[:2000]
    if len(section_text) > 2000:
        text_excerpt += "\n[...truncated...]"

    return f"""\
## Task

Create an interactive visualization for the following educational concept.

**Section:** {section_title}
**Concept:** {opportunity.concept}
**Visualization type:** {opportunity.viz_type}
**Interactive variables:** {', '.join(opportunity.variables)}
**Learning goal:** {opportunity.learning_goal}

## Source content (for context)

{text_excerpt}

## Example of expected output format

The following is a simple example showing the expected HTML structure, p5.js usage, \
slider controls, hover interactivity, and error handling. Your visualization should \
follow this pattern but be specific to the concept above.

```html
{_VIZ_FEW_SHOT_EXAMPLE}
```

Generate a complete HTML file for the concept described above. Make it visually \
polished, educational, and interactive. The default parameter values should show \
the most instructive starting state.\
"""
