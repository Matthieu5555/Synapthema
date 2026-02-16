"""Tests for mermaid diagram validation and LLM-based auto-fixing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.rendering.mermaid_validator import (
    MAX_FIX_ATTEMPTS,
    MermaidFixResult,
    ValidationResult,
    _build_fix_prompt,
    _build_fix_prompt_no_error,
    fix_mermaid_with_llm,
    validate_and_fix_mermaid_diagrams,
    validate_mermaid,
)
from src.transformation.types import (
    MermaidDiagram,
    MermaidElement,
    Slide,
    SlideElement,
    TrainingModule,
    TrainingSection,
)


# ── ValidationResult ─────────────────────────────────────────────────────────


class TestValidationResult:
    def test_valid(self) -> None:
        r = ValidationResult(valid=True)
        assert r.valid
        assert r.error_message == ""

    def test_invalid_with_message(self) -> None:
        r = ValidationResult(valid=False, error_message="Syntax error")
        assert not r.valid
        assert r.error_message == "Syntax error"


# ── validate_mermaid ─────────────────────────────────────────────────────────


class TestValidateMermaid:
    @patch("src.rendering.mermaid_validator._validator_ready", return_value=False)
    @patch("src.rendering.mermaid_validator._node_available", return_value=True)
    def test_returns_valid_when_mermaid_missing(self, _na, _vr) -> None:
        result = validate_mermaid("flowchart TD\n  A --> B")
        assert result.valid

    @patch("src.rendering.mermaid_validator._validator_ready", return_value=True)
    @patch("src.rendering.mermaid_validator._node_available", return_value=False)
    def test_returns_valid_when_node_missing(self, _na, _vr) -> None:
        result = validate_mermaid("flowchart TD\n  A --> B")
        assert result.valid

    @patch("src.rendering.mermaid_validator._validator_ready", return_value=True)
    @patch("src.rendering.mermaid_validator._node_available", return_value=True)
    @patch("src.rendering.mermaid_validator.subprocess")
    def test_valid_diagram(self, mock_sub, _na, _vr) -> None:
        mock_sub.run.return_value = MagicMock(returncode=0)
        result = validate_mermaid("flowchart TD\n  A --> B")
        assert result.valid

    @patch("src.rendering.mermaid_validator._validator_ready", return_value=True)
    @patch("src.rendering.mermaid_validator._node_available", return_value=True)
    @patch("src.rendering.mermaid_validator.subprocess")
    def test_invalid_diagram(self, mock_sub, _na, _vr) -> None:
        mock_sub.run.return_value = MagicMock(
            returncode=1, stderr="Parse error at line 2", stdout="",
        )
        result = validate_mermaid("flowchart TD\n  A ->- B")
        assert not result.valid
        assert "Parse error" in result.error_message

    @patch("src.rendering.mermaid_validator._validator_ready", return_value=True)
    @patch("src.rendering.mermaid_validator._node_available", return_value=True)
    @patch("src.rendering.mermaid_validator.subprocess")
    def test_timeout_returns_valid(self, mock_sub, _na, _vr) -> None:
        import subprocess
        mock_sub.run.side_effect = subprocess.TimeoutExpired("node", 15)
        mock_sub.TimeoutExpired = subprocess.TimeoutExpired
        result = validate_mermaid("flowchart TD\n  A --> B")
        assert result.valid


# ── fix_mermaid_with_llm ─────────────────────────────────────────────────────


class TestFixMermaidWithLlm:
    def test_returns_fixed_code(self) -> None:
        mock_client = MagicMock()
        mock_client.complete_structured_light.return_value = MermaidFixResult(
            fixed_diagram_code="flowchart TD\n  A --> B",
            explanation="Fixed arrow syntax",
        )
        result = fix_mermaid_with_llm("bad", "error msg", mock_client)
        assert result == "flowchart TD\n  A --> B"

    def test_returns_none_on_failure(self) -> None:
        mock_client = MagicMock()
        mock_client.complete_structured_light.side_effect = Exception("API error")
        result = fix_mermaid_with_llm("bad", "error msg", mock_client)
        assert result is None

    def test_no_error_message_uses_different_prompt(self) -> None:
        mock_client = MagicMock()
        mock_client.complete_structured_light.return_value = MermaidFixResult(
            fixed_diagram_code="ok code",
        )
        fix_mermaid_with_llm("code", None, mock_client)
        call_args = mock_client.complete_structured_light.call_args
        # The user_prompt should contain the "Check" phrasing, not the "Fix" phrasing
        user_prompt = call_args[1]["user_prompt"] if "user_prompt" in call_args[1] else call_args[0][1]
        assert "Check this mermaid diagram" in user_prompt


# ── Prompt builders ──────────────────────────────────────────────────────────


class TestPromptBuilders:
    def test_fix_prompt_includes_error(self) -> None:
        prompt = _build_fix_prompt("code here", "bad syntax")
        assert "bad syntax" in prompt
        assert "code here" in prompt

    def test_fix_prompt_no_error(self) -> None:
        prompt = _build_fix_prompt_no_error("code here")
        assert "code here" in prompt
        assert "Check this mermaid diagram" in prompt


# ── validate_and_fix_mermaid_diagrams ────────────────────────────────────────


def _make_module_with_mermaid(diagram_code: str) -> list[TrainingModule]:
    return [
        TrainingModule(
            chapter_number=1,
            title="Test",
            sections=[
                TrainingSection(
                    title="Section",
                    elements=[
                        MermaidElement(
                            bloom_level="understand",
                            mermaid=MermaidDiagram(
                                title="Test Diagram",
                                diagram_code=diagram_code,
                                caption="Test caption",
                                diagram_type="flowchart",
                            ),
                        ),
                    ],
                ),
            ],
        ),
    ]


class TestValidateAndFixDiagrams:
    @patch("src.rendering.mermaid_validator.validate_mermaid")
    def test_valid_diagram_unchanged(self, mock_val) -> None:
        mock_val.return_value = ValidationResult(valid=True)
        modules = _make_module_with_mermaid("flowchart TD\n  A --> B")
        total, fixed, unfixable = validate_and_fix_mermaid_diagrams(modules)
        assert (total, fixed, unfixable) == (1, 0, 0)

    @patch("src.rendering.mermaid_validator.validate_mermaid")
    def test_invalid_without_llm_is_unfixable(self, mock_val) -> None:
        mock_val.return_value = ValidationResult(valid=False, error_message="err")
        modules = _make_module_with_mermaid("bad")
        total, fixed, unfixable = validate_and_fix_mermaid_diagrams(modules, llm_client=None)
        assert (total, fixed, unfixable) == (1, 0, 1)

    @patch("src.rendering.mermaid_validator.validate_mermaid")
    @patch("src.rendering.mermaid_validator.fix_mermaid_with_llm")
    def test_llm_fix_succeeds(self, mock_fix, mock_val) -> None:
        mock_val.side_effect = [
            ValidationResult(valid=False, error_message="err"),
            ValidationResult(valid=True),
        ]
        mock_fix.return_value = "flowchart TD\n  A --> B"

        modules = _make_module_with_mermaid("bad")
        total, fixed, unfixable = validate_and_fix_mermaid_diagrams(modules, MagicMock())
        assert (total, fixed, unfixable) == (1, 1, 0)
        assert modules[0].sections[0].elements[0].mermaid.diagram_code == "flowchart TD\n  A --> B"

    @patch("src.rendering.mermaid_validator.validate_mermaid")
    @patch("src.rendering.mermaid_validator.fix_mermaid_with_llm")
    def test_llm_fix_exhausts_attempts(self, mock_fix, mock_val) -> None:
        mock_val.return_value = ValidationResult(valid=False, error_message="err")
        mock_fix.return_value = "still bad"

        modules = _make_module_with_mermaid("bad")
        total, fixed, unfixable = validate_and_fix_mermaid_diagrams(modules, MagicMock())
        assert (total, fixed, unfixable) == (1, 0, 1)
        assert mock_fix.call_count == MAX_FIX_ATTEMPTS

    @patch("src.rendering.mermaid_validator.validate_mermaid")
    def test_skips_non_mermaid_elements(self, mock_val) -> None:
        modules = [
            TrainingModule(
                chapter_number=1,
                title="Test",
                sections=[
                    TrainingSection(
                        title="Section",
                        elements=[
                            SlideElement(
                                bloom_level="understand",
                                slide=Slide(title="Slide", content="Hello"),
                            ),
                        ],
                    ),
                ],
            ),
        ]
        total, fixed, unfixable = validate_and_fix_mermaid_diagrams(modules)
        assert (total, fixed, unfixable) == (0, 0, 0)
        mock_val.assert_not_called()

    @patch("src.rendering.mermaid_validator.validate_mermaid")
    def test_unicode_escapes_fixed_before_validation(self, mock_val) -> None:
        mock_val.return_value = ValidationResult(valid=True)
        modules = _make_module_with_mermaid("flowchart TD\n  A[\\u0394] --> B")
        validate_and_fix_mermaid_diagrams(modules)
        # Unicode escape should be decoded to actual character
        assert modules[0].sections[0].elements[0].mermaid.diagram_code == "flowchart TD\n  A[Δ] --> B"
