"""Tests for the LLM client — tiered model routing (Task 05)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.transformation.llm_client import create_llm_client


class TestCreateLLMClient:
    """Test the closure-based factory routes models correctly."""

    @patch("src.transformation.llm_client.openai.OpenAI")
    @patch("src.transformation.llm_client.instructor.from_openai")
    def test_light_defaults_to_primary(self, mock_instructor: MagicMock, mock_openai: MagicMock) -> None:
        """When model_light is empty, light methods should use the primary model."""
        mock_raw_client = MagicMock()
        mock_openai.return_value = mock_raw_client
        mock_patched = MagicMock()
        mock_instructor.return_value = mock_patched
        mock_patched.client = mock_raw_client

        # Set up the mock to track which model is used
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response"
        mock_raw_client.chat.completions.create.return_value = mock_response

        client = create_llm_client(
            api_key="test-key",
            model="gpt-5.2",
            max_tokens=4096,
            temperature=0.3,
            model_light="",  # Empty → falls back to primary
        )

        client.complete_light("sys", "user")
        # The model used should be the primary "gpt-5.2"
        call_kwargs = mock_raw_client.chat.completions.create.call_args
        assert call_kwargs[1]["model"] == "gpt-5.2"

    @patch("src.transformation.llm_client.openai.OpenAI")
    @patch("src.transformation.llm_client.instructor.from_openai")
    def test_light_uses_light_model(self, mock_instructor: MagicMock, mock_openai: MagicMock) -> None:
        """When model_light is set, light methods should use it."""
        mock_raw_client = MagicMock()
        mock_openai.return_value = mock_raw_client
        mock_patched = MagicMock()
        mock_instructor.return_value = mock_patched
        mock_patched.client = mock_raw_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response"
        mock_raw_client.chat.completions.create.return_value = mock_response

        client = create_llm_client(
            api_key="test-key",
            model="gpt-5.2",
            max_tokens=4096,
            temperature=0.3,
            model_light="gpt-4.1-mini",
        )

        client.complete_light("sys", "user")
        call_kwargs = mock_raw_client.chat.completions.create.call_args
        assert call_kwargs[1]["model"] == "gpt-4.1-mini"

    @patch("src.transformation.llm_client.openai.OpenAI")
    @patch("src.transformation.llm_client.instructor.from_openai")
    def test_primary_complete_uses_primary_model(self, mock_instructor: MagicMock, mock_openai: MagicMock) -> None:
        """Primary complete() should still use the primary model."""
        mock_raw_client = MagicMock()
        mock_openai.return_value = mock_raw_client
        mock_patched = MagicMock()
        mock_instructor.return_value = mock_patched
        mock_patched.client = mock_raw_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response"
        mock_raw_client.chat.completions.create.return_value = mock_response

        client = create_llm_client(
            api_key="test-key",
            model="gpt-5.2",
            max_tokens=4096,
            temperature=0.3,
            model_light="gpt-4.1-mini",
        )

        client.complete("sys", "user")
        call_kwargs = mock_raw_client.chat.completions.create.call_args
        assert call_kwargs[1]["model"] == "gpt-5.2"
