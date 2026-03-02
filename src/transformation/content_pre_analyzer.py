"""Regex-based content signal detection for pre-analysis.

Pure Python, no LLM calls. Runs BEFORE the deep reader to detect
structural signals in section text: formulas, procedural language,
comparisons, definitions, examples, and key terms.

The output signals serve two purposes:
1. Formatted into the deep reader's prompt to help it focus.
2. Fallback when the deep reader LLM fails — content type signals
   still survive for the planner and content designer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from src.extraction.types import Book

logger = logging.getLogger(__name__)

# ── Document type classification ─────────────────────────────────────────────

DocumentType = Literal[
    "quantitative",    # Math-heavy, formulas, proofs
    "narrative",       # History, stories, case studies
    "procedural",      # Step-by-step processes, tutorials
    "analytical",      # Comparative analysis, research
    "regulatory",      # Law, compliance, standards
    "mixed",           # No dominant type
]

SectionQuality = Literal[
    "content",        # Meaningful learnable content
    "structural",     # ToC, index, bibliography, front/back matter
    "insufficient",   # Too little text to generate training content
]

# Minimum total score to confidently assign a document type.
# Below this threshold, the document is classified as "mixed".
_MIN_DETECTION_CONFIDENCE = 3.0

# Maximum characters sampled from the book for heuristic detection.
_MAX_SAMPLE_LENGTH = 10_000

# Recommended template weight distributions per document type.
# Used by the curriculum planner to bias template assignment.
DOCUMENT_TYPE_TEMPLATE_WEIGHTS: dict[str, dict[str, float]] = {
    "quantitative": {
        "worked_example": 0.30,
        "problem_first": 0.20,
        "error_identification": 0.15,
        "analogy_first": 0.15,
        "compare_contrast": 0.10,
        "visual_walkthrough": 0.10,
    },
    "narrative": {
        "narrative": 0.30,
        "analogy_first": 0.25,
        "socratic": 0.15,
        "vignette": 0.15,
        "problem_first": 0.10,
        "compare_contrast": 0.05,
    },
    "procedural": {
        "visual_walkthrough": 0.30,
        "worked_example": 0.25,
        "problem_first": 0.15,
        "vignette": 0.15,
        "error_identification": 0.10,
        "narrative": 0.05,
    },
    "analytical": {
        "compare_contrast": 0.30,
        "socratic": 0.20,
        "error_identification": 0.15,
        "vignette": 0.15,
        "analogy_first": 0.10,
        "problem_first": 0.10,
    },
    "regulatory": {
        "compare_contrast": 0.25,
        "vignette": 0.25,
        "error_identification": 0.20,
        "worked_example": 0.15,
        "narrative": 0.10,
        "socratic": 0.05,
    },
    "mixed": {
        "analogy_first": 0.18,
        "worked_example": 0.17,
        "compare_contrast": 0.15,
        "problem_first": 0.13,
        "socratic": 0.12,
        "narrative": 0.10,
        "vignette": 0.08,
        "error_identification": 0.07,
    },
}


# ── Output types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SectionSignals:
    """Content signals detected in a single section's text."""

    section_title: str
    formula_count: int = 0
    procedural_count: int = 0
    comparison_count: int = 0
    definition_count: int = 0
    example_count: int = 0
    key_terms: tuple[str, ...] = field(default=())

    @property
    def dominant_type(self) -> str:
        """Infer the dominant content type from signal counts.

        Returns one of: conceptual, procedural, comparative, theoretical,
        applied, mixed.
        """
        counts = {
            "procedural": self.procedural_count,
            "comparative": self.comparison_count,
            "conceptual": self.definition_count,
            "applied": self.example_count,
            "theoretical": self.formula_count,
        }
        total = sum(counts.values())
        if total == 0:
            return "mixed"

        top_type, top_count = max(counts.items(), key=lambda x: x[1])
        # If the top type accounts for less than 40% of signals, call it mixed
        if top_count / total < 0.4:
            return "mixed"
        return top_type


@dataclass(frozen=True)
class ChapterSignals:
    """Aggregated content signals across all sections of a chapter."""

    chapter_title: str
    sections: tuple[SectionSignals, ...] = field(default=())

    @property
    def total_formulas(self) -> int:
        return sum(s.formula_count for s in self.sections)

    @property
    def total_procedures(self) -> int:
        return sum(s.procedural_count for s in self.sections)

    @property
    def total_comparisons(self) -> int:
        return sum(s.comparison_count for s in self.sections)

    @property
    def total_definitions(self) -> int:
        return sum(s.definition_count for s in self.sections)

    @property
    def total_examples(self) -> int:
        return sum(s.example_count for s in self.sections)

    @property
    def all_key_terms(self) -> tuple[str, ...]:
        seen: set[str] = set()
        terms: list[str] = []
        for s in self.sections:
            for t in s.key_terms:
                lower = t.lower()
                if lower not in seen:
                    seen.add(lower)
                    terms.append(t)
        return tuple(terms)


# ── Public API ───────────────────────────────────────────────────────────────


def analyze_section(section_title: str, text: str) -> SectionSignals:
    """Detect content signals in a section's text.

    Args:
        section_title: The section heading.
        text: Full text content of the section.

    Returns:
        SectionSignals with counts for each signal type.
    """
    return SectionSignals(
        section_title=section_title,
        formula_count=_count_formulas(text),
        procedural_count=_count_procedural(text),
        comparison_count=_count_comparisons(text),
        definition_count=_count_definitions(text),
        example_count=_count_examples(text),
        key_terms=_extract_key_terms(text),
    )


def analyze_chapter_sections(
    chapter_title: str,
    sections: list[tuple[str, str]],
) -> ChapterSignals:
    """Analyze all sections of a chapter.

    Args:
        chapter_title: The chapter heading.
        sections: List of (section_title, section_text) tuples.

    Returns:
        ChapterSignals aggregating all section signals.
    """
    section_signals = tuple(
        analyze_section(title, text) for title, text in sections
    )
    return ChapterSignals(
        chapter_title=chapter_title,
        sections=section_signals,
    )


# ── Section quality classification ────────────────────────────────────────────

# Minimum stripped text length for a section to be worth transforming.
_MIN_CONTENT_LENGTH = 200

# Titles that indicate structural (non-content) pages.
_STRUCTURAL_TITLE_PATTERNS = re.compile(
    r"(?i)^(?:"
    r"table\s+of\s+contents?"
    r"|contents?"
    r"|index"
    r"|bibliography"
    r"|references"
    r"|glossary"
    r"|appendix\s+[a-z]?\s*$"
    r"|list\s+of\s+(?:figures|tables|abbreviations|symbols)"
    r"|acknowledgements?"
    r"|about\s+the\s+authors?"
    r"|colophon"
    r"|copyright"
    r"|dedication"
    r"|foreword"
    r"|preface"
    r"|further\s+reading"
    r"|suggested\s+reading"
    r"|notes"
    r")$"
)

# Lines ending with page numbers (e.g., "Chapter 1 ..... 5" or "Introduction  12")
_TOC_LINE_PATTERN = re.compile(
    r"^.{5,}\s*\.{2,}\s*\d+\s*$"  # dots + page number
    r"|^.{5,}\s{2,}\d+\s*$",       # spaces + page number
    re.MULTILINE,
)


def classify_section_quality(title: str, text: str) -> SectionQuality:
    """Classify whether a section contains meaningful learnable content.

    Zero LLM cost. Uses heuristics based on text length, title patterns,
    and content structure to filter out structural pages (ToC, index,
    bibliography) and insufficient content before expensive LLM calls.

    Args:
        title: The section heading.
        text: Full text content of the section.

    Returns:
        'content', 'structural', or 'insufficient'.
    """
    stripped = text.strip()

    # Check text length first (cheapest check)
    if len(stripped) < _MIN_CONTENT_LENGTH:
        return "insufficient"

    # Check title against structural patterns
    if _STRUCTURAL_TITLE_PATTERNS.match(title.strip()):
        return "structural"

    # Check for ToC-like content (many lines ending with page numbers)
    lines = stripped.splitlines()
    non_empty_lines = [l for l in lines if l.strip()]
    if non_empty_lines:
        toc_lines = len(_TOC_LINE_PATTERN.findall(stripped))
        if toc_lines / len(non_empty_lines) > 0.5:
            return "structural"

    return "content"


# ── Formula detection ────────────────────────────────────────────────────────

# LaTeX math: $...$ (not currency like $100) and $$...$$
# NOTE: Use [$] instead of \$ — Python 3.13+ treats \$ as end-of-string anchor.
_INLINE_MATH = re.compile(r"(?<!\w)[$](?!\d)(.+?)[$](?!\d)")
_DISPLAY_MATH = re.compile(r"[$][$](.+?)[$][$]", re.DOTALL)

# LaTeX commands commonly used in formulas
_LATEX_COMMANDS = re.compile(
    r"\\(?:frac|sum|int|prod|sqrt|alpha|beta|gamma|delta|sigma|mu|theta|lambda|"
    r"partial|nabla|infty|lim|log|ln|exp|sin|cos|tan|begin\{equation\}|"
    r"begin\{align\}|mathbb|mathcal|overline|underline|hat|bar|vec|dot)"
)

# Common formula patterns without LaTeX
_FORMULA_PATTERNS = re.compile(
    r"(?:"
    r"[A-Z][a-z]?\s*=\s*[A-Z\d]"  # E = mc, PV = FV
    r"|[A-Z]{2,}\s*=\s*"           # NPV = , CAPM =
    r"|\b\w+\s*/\s*\w+"            # return / risk
    r")"
)


def _count_formulas(text: str) -> int:
    """Count formula indicators in text."""
    count = 0
    count += len(_DISPLAY_MATH.findall(text))
    count += len(_INLINE_MATH.findall(text))
    count += len(_LATEX_COMMANDS.findall(text))
    count += len(_FORMULA_PATTERNS.findall(text))
    return count


# ── Procedural language detection ────────────────────────────────────────────

_PROCEDURAL_PATTERNS = re.compile(
    r"(?i)(?:"
    r"\bstep\s+\d+"               # Step 1, Step 2
    r"|\b(?:first|second|third|fourth|fifth)\b(?:ly)?,?\s"  # ordinals
    r"|\bprocedure\b"
    r"|\balgorithm\b"
    r"|\bprocess\b"
    r"|\bfollow(?:ing)?\s+(?:these\s+)?steps\b"
    r"|\bhow\s+to\b"
    r"|\binstructions?\b"
    r"|\bworkflow\b"
    r"|\bsequence\s+of\b"
    r"|\b\d+\)\s+[A-Z]"           # 1) Calculate, 2) Determine
    r")"
)


def _count_procedural(text: str) -> int:
    """Count procedural language indicators."""
    return len(_PROCEDURAL_PATTERNS.findall(text))


# ── Comparison language detection ────────────────────────────────────────────

_COMPARISON_PATTERNS = re.compile(
    r"(?i)(?:"
    r"\bwhereas\b"
    r"|\bin\s+contrast\b"
    r"|\bversus\b"
    r"|\bvs\.?\b"
    r"|\bdiffers?\s+from\b"
    r"|\bunlike\b"
    r"|\bon\s+the\s+other\s+hand\b"
    r"|\bcompare[ds]?\s+(?:to|with)\b"
    r"|\bsimilar(?:ly)?\s+to\b"
    r"|\bdistinguish(?:es|ed)?\s+(?:between|from)\b"
    r"|\badvantages?\s+(?:and|over)\b"
    r"|\bdisadvantages?\b"
    r"|\bpros?\s+and\s+cons?\b"
    r"|\bstrengths?\s+and\s+weaknesses?\b"
    r")"
)


def _count_comparisons(text: str) -> int:
    """Count comparison/contrast language indicators."""
    return len(_COMPARISON_PATTERNS.findall(text))


# ── Definition pattern detection ─────────────────────────────────────────────

_DEFINITION_PATTERNS = re.compile(
    r"(?i)(?:"
    r"\bis\s+defined\s+as\b"
    r"|\brefers?\s+to\b"
    r"|\bknown\s+as\b"
    r"|\bcalled\b"
    r"|\bmeans\s+that\b"
    r"|\bdefinition\b"
    r"|\bthe\s+term\b"
    r"|\bwe\s+define\b"
    r"|\bcan\s+be\s+described\s+as\b"
    r"|\bis\s+(?:a|an|the)\s+(?:measure|concept|process|method|technique|approach|framework)\b"
    r")"
)


def _count_definitions(text: str) -> int:
    """Count definition pattern indicators."""
    return len(_DEFINITION_PATTERNS.findall(text))


# ── Example pattern detection ────────────────────────────────────────────────

_EXAMPLE_PATTERNS = re.compile(
    r"(?i)(?:"
    r"\bfor\s+example\b"
    r"|\bfor\s+instance\b"
    r"|\bconsider\s+(?:a|an|the|this)\b"
    r"|\bsuppose\b"
    r"|\bimagine\b"
    r"|\billustrat(?:e[ds]?|ion)\b"
    r"|\bdemonstrat(?:e[ds]?|ion)\b"
    r"|\bcase\s+study\b"
    r"|\bexample\s*[:]\b"
    r"|\bexample\s+\d+\b"
    r"|\bscenario\b"
    r"|\bin\s+practice\b"
    r"|\breal[\s-]world\b"
    r")"
)


def _count_examples(text: str) -> int:
    """Count example/illustration indicators."""
    return len(_EXAMPLE_PATTERNS.findall(text))


# ── Key term extraction ──────────────────────────────────────────────────────

# Bold markers in PDF-extracted text (often **word** or similar patterns)
_BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*")

# Capitalized multi-word phrases (2-4 words, likely proper nouns / key terms)
_CAPITALIZED_PHRASE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b"
)

# Minimum length for a key term to be meaningful
_MIN_TERM_LENGTH = 4

# Common words to exclude from key terms
_STOP_PHRASES = frozenset({
    "the", "this", "that", "these", "those", "with", "from", "into",
    "have", "been", "will", "would", "could", "should",
})


def _extract_key_terms(text: str) -> tuple[str, ...]:
    """Extract potential key terms from text."""
    terms: list[str] = []
    seen: set[str] = set()

    # Extract bold-marked terms
    for match in _BOLD_PATTERN.finditer(text):
        term = match.group(1).strip()
        if len(term) >= _MIN_TERM_LENGTH and term.lower() not in seen:
            seen.add(term.lower())
            terms.append(term)

    # Extract capitalized phrases (limit to avoid noise)
    for match in _CAPITALIZED_PHRASE.finditer(text):
        term = match.group(1).strip()
        lower = term.lower()
        if (
            len(term) >= _MIN_TERM_LENGTH
            and lower not in seen
            and not any(w in _STOP_PHRASES for w in lower.split())
        ):
            seen.add(lower)
            terms.append(term)

    # Cap at 20 terms to avoid noise
    return tuple(terms[:20])


# ── Document type detection ────────────────────────────────────────────────


_QUANTITATIVE_KEYWORDS = frozenset({
    "theorem", "proof", "equation", "formula", "calculate",
    "derivative", "integral", "matrix", "coefficient", "variance",
})

_NARRATIVE_KEYWORDS = frozenset({
    "history", "story", "century", "developed", "discovered",
    "founded", "era", "period", "revolution", "movement",
})

_PROCEDURAL_KEYWORDS = frozenset({
    "step", "process", "procedure", "algorithm", "implement",
    "execute", "install", "configure", "deploy", "workflow",
})

_ANALYTICAL_KEYWORDS = frozenset({
    "compare", "contrast", "analysis", "evaluate", "assess",
    "criteria", "framework", "methodology", "hypothesis", "findings",
})

_REGULATORY_KEYWORDS = frozenset({
    "regulation", "compliance", "standard", "requirement", "provision",
    "statute", "legislation", "mandate", "ordinance", "enforcement",
})

# Pattern for ordered lists: "1. ", "2) ", etc.
_ORDERED_LIST_PATTERN = re.compile(r"^\s*\d+[.)]\s+[A-Z]", re.MULTILINE)


def detect_document_type(book: Book) -> DocumentType:
    """Classify document type from structural and textual signals.

    Uses a weighted scoring system across heuristic signals:
    formula density, keyword matches, paragraph length, and
    ordered list counts. Zero LLM cost — purely regex-based.

    Args:
        book: Extracted Book from Stage 1.

    Returns:
        The detected DocumentType literal.
    """
    sample = _build_text_sample(book)
    if not sample.strip():
        return "mixed"

    scores: dict[str, float] = {
        "quantitative": (
            _count_formulas(sample) * 0.5
            + _keyword_density(sample, _QUANTITATIVE_KEYWORDS) * 3.0
            + _count_numeric_tables(book) * 0.5
        ),
        "narrative": (
            _keyword_density(sample, _NARRATIVE_KEYWORDS) * 3.0
            + _average_paragraph_length_score(sample) * 1.0
        ),
        "procedural": (
            _keyword_density(sample, _PROCEDURAL_KEYWORDS) * 3.0
            + _count_ordered_lists(sample) * 0.5
        ),
        "analytical": (
            _keyword_density(sample, _ANALYTICAL_KEYWORDS) * 3.0
        ),
        "regulatory": (
            _keyword_density(sample, _REGULATORY_KEYWORDS) * 3.0
        ),
    }

    best_type = max(scores, key=lambda k: scores[k])
    best_score = scores[best_type]

    if best_score < _MIN_DETECTION_CONFIDENCE:
        logger.info("Document type detection: no clear winner (best=%s, score=%.1f), defaulting to mixed", best_type, best_score)
        return "mixed"

    logger.info("Document type detected: %s (score=%.1f)", best_type, best_score)
    return best_type  # type: ignore[return-value]


def _build_text_sample(book: Book) -> str:
    """Build a representative text sample from the book.

    Collects section titles and text from all chapters, capped at
    _MAX_SAMPLE_LENGTH characters.
    """
    parts: list[str] = []
    char_count = 0

    for chapter in book.chapters:
        parts.append(chapter.title)
        char_count += len(chapter.title)
        for section in chapter.sections:
            parts.append(section.title)
            char_count += len(section.title)
            text = section.text.strip()
            if char_count + len(text) > _MAX_SAMPLE_LENGTH:
                remaining = _MAX_SAMPLE_LENGTH - char_count
                if remaining > 0:
                    parts.append(text[:remaining])
                break
            parts.append(text)
            char_count += len(text)
        if char_count >= _MAX_SAMPLE_LENGTH:
            break

    return "\n".join(parts)


def _keyword_density(text: str, keywords: frozenset[str]) -> float:
    """Count how many keywords from the set appear in the text.

    Returns the raw count of keyword matches (case-insensitive).
    """
    text_lower = text.lower()
    return sum(1.0 for kw in keywords if kw in text_lower)


def _count_numeric_tables(book: Book) -> float:
    """Count tables across the book that contain numeric data."""
    count = 0
    for chapter in book.chapters:
        for section in chapter.sections:
            for table in section.tables:
                for row in table.rows:
                    if any(re.search(r"\d+\.?\d*", cell) for cell in row):
                        count += 1
                        break
    return float(count)


def _count_ordered_lists(text: str) -> float:
    """Count ordered list items (e.g., '1. Step', '2) Do') in text."""
    return float(len(_ORDERED_LIST_PATTERN.findall(text)))


def _average_paragraph_length_score(text: str) -> float:
    """Score based on average paragraph length.

    Longer paragraphs suggest narrative content. Returns a score
    between 0 and 3, where 3 means very long paragraphs (>300 chars avg).
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return 0.0
    avg_len = sum(len(p) for p in paragraphs) / len(paragraphs)
    # Scale: 0 at <100 chars, 1 at 150, 2 at 250, 3 at 350+
    return min(max((avg_len - 100) / 80, 0.0), 3.0)


def format_document_type_guidance(document_type: DocumentType) -> str:
    """Format template weight guidance for the curriculum planner prompt.

    Args:
        document_type: The detected or configured document type.

    Returns:
        A formatted string describing recommended template weights.
    """
    weights = DOCUMENT_TYPE_TEMPLATE_WEIGHTS.get(document_type, DOCUMENT_TYPE_TEMPLATE_WEIGHTS["mixed"])
    lines = [f"  - {template}: {weight:.0%}" for template, weight in sorted(weights.items(), key=lambda x: -x[1])]
    return "\n".join(lines)
