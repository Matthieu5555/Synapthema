# Evidence-Based Learning Science

Everything here is backed by empirical research — meta-analyses, randomized controlled trials, or well-replicated findings. These are the principles the engine should implement.

## The Two Highest-Impact Techniques

Dunlosky et al. (2013) reviewed 10 learning strategies across hundreds of studies and rated only TWO as "high utility":

### 1. Retrieval Practice (Testing Effect)

**The finding:** Testing yourself on material is dramatically more effective than re-reading it. Students who took a single practice test remembered 50% more than students who studied four times (Roediger & Karpicke, 2006).

**Meta-analytic effect size:** g = 0.50 (Rowland, 2014, dozens of studies). Adesope et al. (2017) examined 217 studies and confirmed the effect is most robust when initial retrieval success exceeds 75%.

**Implementation:**
- Testing should be the primary learning activity, not a secondary assessment
- Every content section must end with retrieval questions
- Target ~75-85% initial success rate (the "desirable difficulty" sweet spot)
- Provide feedback AFTER retrieval attempts, not before
- Alternate study and test trials rather than study-only blocks

### 2. Spaced Repetition (Distributed Practice)

**The finding:** Ebbinghaus showed memory decays exponentially but can be rescued through strategically timed reviews. The optimal gap is ~20% of the desired retention interval for short delays, falling to ~5-10% for yearly retention (Cepeda et al., 2008, n=1,350).

**The FSRS algorithm** (Free Spaced Repetition Scheduler) is the state of the art, replacing SM-2. It uses a Three Component Model of Memory (Retrievability, Stability, Difficulty) with 21 machine-learned parameters. Achieves 20-30% fewer reviews than SM-2 for the same retention level.

**Implementation:**
- Use the py-fsrs library (MIT license, production-ready): https://github.com/open-spaced-repetition/py-fsrs
- Track per-item: difficulty (D), stability (S), retrievability (R)
- Schedule reviews when R drops below the target retention (default 0.9 = 90%)
- Store review history in localStorage for client-side scheduling, or server-side for multi-device

---

## Other High-Value Principles

### Interleaving (g = 0.42)

**The finding:** Mixing different problem types in practice sessions is more effective than blocking by type (Brunmair & Richter, 2019 meta-analysis). The effect is strongest when concepts are confusable (similar on the surface but different underneath).

**Caveat:** Blocking is better for initial learning of brand-new concepts. Interleaving kicks in during review.

**Implementation:**
- After teaching concepts A, B, C separately, present mixed practice sets
- Specifically interleave confusable concepts (e.g., similar formulas, related but distinct principles)
- Tag exercises with topic metadata to enable intelligent mixing

### Elaborative Interrogation

**The finding:** Learners prompted to explain "why" and "how" things work generate deeper processing than those who simply receive explanations (Dunlosky, 2013 — "moderate utility").

**Implementation:**
- Generate "why" and "how" questions, not just "what is" questions
- After presenting a fact: "Why does X lead to Y?" or "How would this change if Z were different?"
- Use elaborative interrogation as its own exercise type

### Concrete Examples and Worked Examples

**The finding:** Studying worked examples leads to better learning outcomes than unguided problem-solving for novices (Sweller & Cooper, 1985). Sweller calls this "the best known and most widely studied of the cognitive load effects."

**The fading technique** (Renkl & Atkinson): Start with fully worked examples, then progressively remove steps until only the problem statement remains. Superior to alternating complete examples and complete problems.

**The expertise reversal effect:** Worked examples help novices but become redundant for advanced learners, who benefit more from problem-solving.

**Implementation:**
- For new concepts: Full worked example → Example with 1 step missing → Example with 2 steps missing → Full problem
- Track learner proficiency per topic; shift from worked examples to problems as proficiency increases
- Use concrete analogies to bridge abstract concepts before the formal definition

### Desirable Difficulties (Bjork, 1994)

**The finding:** "Conditions that create short-term challenges enhance long-term retention." The paradox: easy learning feels good but doesn't stick. Difficult learning feels frustrating but lasts.

Key desirable difficulties: spacing (vs. massing), interleaving (vs. blocking), retrieval practice (vs. re-reading), generation (vs. reading).

**Implementation:**
- Do NOT optimize for immediate session accuracy. Optimize for delayed test performance.
- If everything is easy, the learner is not learning. Introduce deliberate friction.
- Require generation before showing answers. Use varied contexts for the same concept.

### Cognitive Load Theory (Sweller)

Three types of load:
- **Intrinsic** — the inherent complexity of the topic (driven by how many elements interact)
- **Extraneous** — caused by poor instructional design (split attention, redundancy, decorative images)
- **Germane** — productive processing that builds lasting schema (the good kind of load)

**Goal:** Minimize extraneous, manage intrinsic through sequencing, maximize germane.

**Implementation:**
- Clean UI, no unnecessary animations or decorative images
- Keep related info together on screen (no split attention)
- Break complex topics into chunks; teach elements in isolation before combining
- Use the knowledge graph to determine prerequisite ordering
- Exercises requiring schema construction (comparison, categorization) over superficial processing

### Zone of Proximal Development / Scaffolding (Vygotsky)

**The finding:** Present challenges just beyond current ability, achievable with support, then gradually remove support.

**Implementation:**
- Adaptive difficulty: track accuracy per topic, increase difficulty when accuracy > 85%, decrease when < 65%
- Tiered hints: first hint is a nudge, second is specific, third reveals the approach (not the answer)
- Use the knowledge graph to determine which concepts are in the learner's ZPD based on mastered prerequisites

### Bloom's Taxonomy (Revised — Anderson & Krathwohl, 2001)

Six cognitive levels, hierarchical:

| Level | Verb | Exercise Types |
|-------|------|----------------|
| Remember | Recall, list, define | Flashcards, definition matching |
| Understand | Explain, paraphrase, summarize | Slides with analogies, "explain in your own words" |
| Apply | Solve, calculate, use | Fill-in-the-blank, worked problems |
| Analyze | Compare, contrast, distinguish | Compare/contrast questions, error identification |
| Evaluate | Judge, justify, defend | "Is this approach correct? Why or why not?" |
| Create | Design, synthesize, construct | "Design a solution that satisfies X, Y, Z" |

**Implementation:**
- Tag every exercise with a Bloom's level
- Enforce distribution: no more than 20% at Remember level
- Progress from Remember → Create within each topic, gated by mastery at lower levels
- Use Bloom's level as an explicit parameter in every LLM prompt

### Knowledge Graphs / Concept Maps

**The finding:** Expert knowledge is hierarchical and densely interconnected. Novice knowledge is fragmented and spoke-like (isolated facts radiating from a central node). Experts organize around deep structural principles; novices organize around surface features.

**Implementation:**
- Build a concept dependency graph: nodes = concepts, directed edges = "prerequisite of"
- Display the graph visually so learners can see progress and trajectory
- Use the graph for prerequisite enforcement (don't present B until A is mastered)
- Use it for connection building (after teaching related concepts, explicitly link them)

### Dual Coding (Paivio, 1971)

**The finding:** Combining verbal and visual channels creates more interconnected memory traces. Multimedia > single-channel.

**Implementation:**
- Pair text explanations with diagrams, charts, or visual metaphors
- For math: visual representations alongside formulas
- For processes: flowcharts or step-by-step visuals
- Do NOT use decorative images (they add extraneous load without benefit)

### The Generation Effect (g = 0.40)

**The finding:** Self-generated information is remembered better than read information (Slamecka & Graf, 1978; Bertsch et al., 2007 meta-analysis of 86 studies).

**Implementation:**
- Require learners to generate answers before revealing them
- Fill-in-the-blank, cloze deletions, free-response questions all leverage this
- Present incomplete worked examples where learners generate the missing steps

### Storytelling and Narrative

**The finding:** Information embedded in narrative context is remembered significantly better than isolated facts. Stories create causal links between events, which mirrors how human memory works (schema theory). Anecdotes, history, and "why it was made this way" create germane cognitive load — the productive kind that builds lasting understanding.

**Implementation:**
- Include origin stories: who created this concept and why?
- Add anecdotes that make abstract ideas concrete and memorable
- Build logical connections between items through narrative threads
- Present the "why" before the "what" — motivation before definition
- Use the history of a concept to explain its current form

---

## Key References

- Dunlosky et al. (2013) — "Strengthening the Student Toolbox" — meta-review of 10 learning strategies
- Roediger & Karpicke (2006) — "Test-Enhanced Learning" — retrieval practice
- Cepeda et al. (2008) — optimal spacing intervals (n=1,350)
- Brunmair & Richter (2019) — interleaving meta-analysis (g=0.42)
- Bjork & Bjork (2011) — desirable difficulties
- Sweller & Cooper (1985) — worked example effect
- Slamecka & Graf (1978) — generation effect
- Adesope et al. (2017) — retrieval practice meta-analysis (217 studies)
- FSRS: https://github.com/open-spaced-repetition/fsrs4anki/wiki/abc-of-fsrs
