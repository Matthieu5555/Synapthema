"""Scientific/math/physics content profile.

Overrides the general-purpose defaults for STEM-heavy content:
- Heavier worked_example and derivation-step emphasis
- Math-notation-specific prompt rules
- Dimensional analysis and proof-structure guidance
- Adjusted template weights favouring quantitative templates
"""

from src.profiles.base import ContentProfile

SCIENTIFIC_PROFILE = ContentProfile(
    name="scientific",

    domain_rules="""\
## Domain-Specific Rules: Scientific / Math / Physics

- **Derivation steps**: Every worked_example MUST show full derivation with
  intermediate algebraic steps.  Never skip steps — the learner needs to see
  *how* to get from A to B, not just A and B.
- **Dimensional analysis**: For any formula, state units explicitly.
  Exercises should include at least one distractor with the wrong units.
- **Notation precision**: Use LaTeX ($...$) for ALL mathematical expressions,
  even simple ones like $x = 3$.  Never write math in plain text.
- **Proof structure**: For theorem-based content, use the pattern:
  Statement → Intuition → Formal proof → Worked example.
- **Physical intuition**: Always provide a physical interpretation before
  formal mathematical treatment.  "What does this *mean* in the real world?"
- **Significant figures**: Exercises should respect significant figures in
  numerical answers.  Flag when precision matters.
- **Variable naming**: Use standard discipline notation (e.g. $F$ for force,
  $v$ for velocity, $\\lambda$ for wavelength).  Define every symbol on first use.
""",

    bloom_prompt_supplements={
        "apply": (
            "For scientific content at the Apply level:\n"
            "- Include at least one numerical problem with full worked solution\n"
            "- Change physical context, not just numbers (e.g. from springs to pendulums)\n"
            "- Distractors should represent common algebraic mistakes\n"
            "- Require unit conversion in at least one exercise\n"
        ),
        "analyze": (
            "For scientific content at the Analyze level:\n"
            "- Include limiting-case analysis (what happens as x → 0 or x → ∞?)\n"
            "- Ask learners to identify which variables matter most (sensitivity)\n"
            "- Use error_detection exercises with plausible sign/unit errors\n"
            "- Include graph interpretation exercises where applicable\n"
        ),
    },

    template_weight_overrides={
        "quantitative": {
            "worked_example": 0.35,
            "problem_first": 0.25,
            "error_identification": 0.15,
            "analogy_first": 0.10,
            "compare_contrast": 0.10,
            "visual_walkthrough": 0.05,
        },
    },

    extra_concept_types=("derivation", "unit", "constant", "law"),
)
