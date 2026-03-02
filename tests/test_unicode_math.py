"""Tests for Unicode math → LaTeX normalization in PDF extraction."""

import pytest

from src.extraction.pdf_parser import _normalize_unicode_math


class TestNormalizeUnicodeMath:
    """Tests for _normalize_unicode_math()."""

    def test_greek_lowercase(self) -> None:
        assert _normalize_unicode_math("α") == r"$\alpha$"
        assert _normalize_unicode_math("β") == r"$\beta$"
        assert _normalize_unicode_math("θ") == r"$\theta$"

    def test_greek_uppercase(self) -> None:
        assert _normalize_unicode_math("Σ") == r"$\Sigma$"
        assert _normalize_unicode_math("Ω") == r"$\Omega$"

    def test_operators(self) -> None:
        assert _normalize_unicode_math("∑") == r"$\sum$"
        assert _normalize_unicode_math("∫") == r"$\int$"
        assert _normalize_unicode_math("∂") == r"$\partial$"
        assert _normalize_unicode_math("∇") == r"$\nabla$"

    def test_relations(self) -> None:
        assert _normalize_unicode_math("≤") == r"$\leq$"
        assert _normalize_unicode_math("≥") == r"$\geq$"
        assert _normalize_unicode_math("≠") == r"$\neq$"
        assert _normalize_unicode_math("∈") == r"$\in$"
        assert _normalize_unicode_math("≈") == r"$\approx$"

    def test_arrows(self) -> None:
        assert _normalize_unicode_math("→") == r"$\to$"
        assert _normalize_unicode_math("⇒") == r"$\Rightarrow$"

    def test_special_symbols(self) -> None:
        assert _normalize_unicode_math("∞") == r"$\infty$"
        assert _normalize_unicode_math("×") == r"$\times$"
        assert _normalize_unicode_math("±") == r"$\pm$"

    def test_set_notation(self) -> None:
        assert _normalize_unicode_math("ℝ") == r"$\mathbb{R}$"
        assert _normalize_unicode_math("ℤ") == r"$\mathbb{Z}$"
        assert _normalize_unicode_math("∅") == r"$\emptyset$"

    def test_superscripts(self) -> None:
        assert _normalize_unicode_math("x²") == r"x$^{2}$"
        assert _normalize_unicode_math("x³") == r"x$^{3}$"
        assert _normalize_unicode_math("xⁿ") == r"x$^{n}$"

    def test_subscripts(self) -> None:
        assert _normalize_unicode_math("x₁") == r"x$_{1}$"
        assert _normalize_unicode_math("x₂") == r"x$_{2}$"

    def test_consecutive_superscripts(self) -> None:
        assert _normalize_unicode_math("x¹²") == r"x$^{12}$"

    def test_consecutive_subscripts(self) -> None:
        assert _normalize_unicode_math("a₁₂") == r"a$_{12}$"

    def test_mixed_text_and_math(self) -> None:
        result = _normalize_unicode_math("The value α is between 0 and ∞")
        assert r"$\alpha$" in result
        assert r"$\infty$" in result
        assert "The value" in result

    def test_preserves_plain_text(self) -> None:
        text = "This is plain text with no math symbols."
        assert _normalize_unicode_math(text) == text

    def test_empty_string(self) -> None:
        assert _normalize_unicode_math("") == ""

    def test_skip_inside_dollar_delimiters(self) -> None:
        text = r"The $\alpha$ variable"
        assert _normalize_unicode_math(text) == text

    def test_skip_inside_double_dollar(self) -> None:
        text = r"$$\sum_{i=1}^{n} α_i$$"
        result = _normalize_unicode_math(text)
        # The α inside $$ should NOT be wrapped again
        assert r"$$\sum_{i=1}^{n} α_i$$" == result

    def test_replaces_outside_but_not_inside_delimiters(self) -> None:
        text = r"The variable α appears in $\beta + \gamma$"
        result = _normalize_unicode_math(text)
        assert r"$\alpha$" in result
        assert r"$\beta + \gamma$" in result

    def test_multiple_symbols_in_sentence(self) -> None:
        text = "For all x ∈ ℝ, we have x² ≥ 0"
        result = _normalize_unicode_math(text)
        assert r"$\in$" in result
        assert r"$\mathbb{R}$" in result
        assert r"$^{2}$" in result
        assert r"$\geq$" in result

    def test_logic_symbols(self) -> None:
        assert _normalize_unicode_math("∀") == r"$\forall$"
        assert _normalize_unicode_math("∃") == r"$\exists$"
        assert _normalize_unicode_math("¬") == r"$\neg$"
