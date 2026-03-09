"""Shared protocols and base types used across layers.

Houses the LLM abstraction types so that both extraction/ and transformation/
can depend on them without creating a cross-layer import cycle.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, Union

T = TypeVar("T")

# User prompt can be a plain string or a list of multimodal content blocks
# (text + image_url) for vision-enabled calls.
UserPrompt = Union[str, list[dict]]


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

    def complete_creative(self, system_prompt: str, user_prompt: UserPrompt) -> str: ...

    def complete_structured_creative(
        self, system_prompt: str, user_prompt: UserPrompt, response_model: type[T]
    ) -> T: ...


class LLMError(Exception):
    """Raised when an LLM API call fails."""
