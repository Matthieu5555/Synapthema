"""Abstracted LLM client layer with Instructor for structured output.

Provides the LLMClient protocol and an OpenAIClient implementation.
Uses Instructor + Pydantic for schema-constrained output — the LLM is
forced to return valid JSON matching our Pydantic models, with automatic
retry on validation failure.

To swap in a custom client, implement the four LLMClient methods and pass
your object wherever the pipeline expects an LLMClient.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol, TypeVar, Union

import instructor
import openai
import pydantic
from instructor.core.exceptions import InstructorError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# User prompt can be a plain string or a list of multimodal content blocks
# (text + image_url) for vision-enabled calls.
UserPrompt = Union[str, list[dict]]

# Maximum number of retry attempts for transient API failures
# (rate limits, 5xx errors). Used by both complete() and complete_structured().
MAX_API_RETRIES = 3

# Base delay in seconds for exponential backoff between retries.
# Actual delays: 2s, 4s, 8s.
BASE_RETRY_DELAY = 2.0

# Number of Instructor-managed retries on Pydantic validation failure.
# Each retry sends the validation error back to the LLM for self-correction.
VALIDATION_RETRIES = 2

# HTTP timeout in seconds for LLM API calls. Prevents indefinite hangs
# when the API is unresponsive. 300s accommodates long structured responses
# for large chapters (e.g. 50+ page chapters with dense content).
REQUEST_TIMEOUT = 300.0


class LLMClient(Protocol):
    """Protocol for LLM completion clients.

    Any implementation must provide complete() for raw text responses,
    complete_structured() for Pydantic-validated responses, and light
    variants of each that route to a cheaper model for simple tasks.
    """

    def complete(self, system_prompt: str, user_prompt: UserPrompt) -> str: ...

    def complete_light(self, system_prompt: str, user_prompt: UserPrompt) -> str: ...

    def complete_structured(
        self, system_prompt: str, user_prompt: UserPrompt, response_model: type[T]
    ) -> T: ...

    def complete_structured_light(
        self, system_prompt: str, user_prompt: UserPrompt, response_model: type[T]
    ) -> T: ...


class LLMError(Exception):
    """Raised when an LLM API call fails."""


class OpenAIClient:
    """LLMClient implementation backed by any OpenAI-compatible API.

    Works with OpenAI, OpenRouter, Azure OpenAI, vLLM, or any endpoint
    that speaks the OpenAI chat completions format. Uses Instructor for
    schema-constrained structured output with automatic validation retry.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int,
        temperature: float,
        base_url: str = "https://openrouter.ai/api/v1",
        model_light: str = "",
    ) -> None:
        self._model = model
        self._model_light = model_light or model
        self._max_tokens = max_tokens
        self._temperature = temperature

        raw_client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=REQUEST_TIMEOUT,
            default_headers={"HTTP-Referer": "https://github.com/learningxp-generator"},
        )
        self._instructor = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)

    def _complete(self, target_model: str, system_prompt: str, user_prompt: UserPrompt) -> str:
        """Raw text completion (shared implementation)."""
        last_error: Exception | None = None

        for attempt in range(MAX_API_RETRIES):
            try:
                response = self._instructor.client.chat.completions.create(  # pyright: ignore[reportOptionalMemberAccess] -- instructor wraps OpenAI client
                    model=target_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_completion_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                content = response.choices[0].message.content
                if not content:
                    raise LLMError("Empty content in API response")
                return content

            except (openai.RateLimitError, openai.APIStatusError) as exc:
                last_error = exc
                if attempt + 1 < MAX_API_RETRIES:
                    delay = BASE_RETRY_DELAY * (2**attempt)
                    logger.warning(
                        "LLM API error, retrying in %.1fs (attempt %d/%d): %s",
                        delay, attempt + 1, MAX_API_RETRIES, exc,
                    )
                    time.sleep(delay)

        raise LLMError(f"Failed after {MAX_API_RETRIES} retries: {last_error}")

    def _structured(
        self, target_model: str, system_prompt: str, user_prompt: UserPrompt, response_model: type[T]
    ) -> T:
        """Schema-constrained completion via Instructor (shared implementation).

        Retries on transient API errors (rate limits, 5xx) with exponential
        backoff. Instructor also retries on Pydantic validation failure by
        sending the error back to the LLM for self-correction.
        """
        last_error: Exception | None = None

        for attempt in range(MAX_API_RETRIES):
            try:
                # Instructor patches the OpenAI client with response_model support;
                # pyright doesn't see the patched signature, so we suppress here.
                create = self._instructor.chat.completions.create  # pyright: ignore[reportOptionalMemberAccess]
                msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                return create(  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue,reportArgumentType]
                    model=target_model,
                    messages=msgs,
                    response_model=response_model,  # pyright: ignore[reportArgumentType]
                    max_completion_tokens=self._max_tokens,
                    temperature=self._temperature,
                    max_retries=VALIDATION_RETRIES,
                )
            except (openai.RateLimitError, openai.APIStatusError) as exc:
                last_error = exc
                if attempt + 1 < MAX_API_RETRIES:
                    delay = BASE_RETRY_DELAY * (2**attempt)
                    logger.warning(
                        "Structured LLM call failed, retrying in %.1fs (attempt %d/%d): %s",
                        delay, attempt + 1, MAX_API_RETRIES, exc,
                    )
                    time.sleep(delay)
            except (InstructorError, openai.APIConnectionError, pydantic.ValidationError) as exc:
                raise LLMError(f"Structured completion failed: {exc}") from exc

        raise LLMError(f"Structured completion failed after {MAX_API_RETRIES} retries: {last_error}")

    def complete(self, system_prompt: str, user_prompt: UserPrompt) -> str:
        """Raw text completion using the primary model."""
        return self._complete(self._model, system_prompt, user_prompt)

    def complete_light(self, system_prompt: str, user_prompt: UserPrompt) -> str:
        """Raw text completion using the light model."""
        logger.debug("Using light model (%s) for raw completion", self._model_light)
        return self._complete(self._model_light, system_prompt, user_prompt)

    def complete_structured(
        self, system_prompt: str, user_prompt: UserPrompt, response_model: type[T]
    ) -> T:
        """Schema-constrained completion using the primary model."""
        return self._structured(self._model, system_prompt, user_prompt, response_model)

    def complete_structured_light(
        self, system_prompt: str, user_prompt: UserPrompt, response_model: type[T]
    ) -> T:
        """Schema-constrained completion using the light model."""
        logger.debug("Using light model (%s) for structured completion", self._model_light)
        return self._structured(self._model_light, system_prompt, user_prompt, response_model)


def create_llm_client(
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    base_url: str = "https://openrouter.ai/api/v1",
    model_light: str = "",
) -> LLMClient:
    """Create an OpenAI-backed LLM client.

    Convenience factory — equivalent to constructing OpenAIClient directly.
    Works with any OpenAI-compatible API (OpenAI, OpenRouter, etc.)
    by configuring the base_url.

    Args:
        api_key: API key for the provider.
        model: Model identifier for complex tasks.
        max_tokens: Max response tokens.
        temperature: Sampling temperature.
        base_url: API base URL.
        model_light: Model identifier for simple tasks. Falls back to
            model if empty.

    Returns:
        An OpenAIClient satisfying the LLMClient protocol.
    """
    logger.info(
        "Creating LLM client: model=%s, model_light=%s, base_url=%s, max_tokens=%d",
        model, model_light or model, base_url, max_tokens,
    )
    return OpenAIClient(
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        base_url=base_url,
        model_light=model_light,
    )
