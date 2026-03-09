"""Shared test fixtures — mock LLM clients.

Provides a reusable MockLLMClient and FailingLLMClient so test modules
don't each need to define their own 4-method mock from scratch.
If the LLMClient protocol gains a method, only this file needs updating.
"""

from __future__ import annotations

import threading
from typing import TypeVar

from src.protocols import LLMError

T = TypeVar("T")


class MockLLMClient:
    """Configurable test double for the LLMClient protocol.

    Returns a fixed *structured_response* from ``complete_structured``
    (and its light variant).  Thread-safe call counting allows assertions
    in both sequential and parallel test scenarios.

    Args:
        structured_response: Object to return from complete_structured().
            Can be a callable ``(system_prompt, user_prompt, response_model) -> T``
            for polymorphic behavior.
    """

    def __init__(self, structured_response: object = None) -> None:
        self._response = structured_response
        self._lock = threading.Lock()
        self.call_count = 0
        self.last_system_prompt: str = ""
        self.last_user_prompt: object = ""

    def complete(self, system_prompt: str, user_prompt: object) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return "mock response"

    def complete_light(self, system_prompt: str, user_prompt: object) -> str:
        return self.complete(system_prompt, user_prompt)

    def complete_structured(
        self, system_prompt: str, user_prompt: object, response_model: type[T],
    ) -> T:
        with self._lock:
            self.call_count += 1
            self.last_system_prompt = system_prompt
            self.last_user_prompt = user_prompt

        if callable(self._response):
            return self._response(system_prompt, user_prompt, response_model)  # type: ignore[return-value]
        return self._response  # type: ignore[return-value]

    def complete_structured_light(
        self, system_prompt: str, user_prompt: object, response_model: type[T],
    ) -> T:
        return self.complete_structured(system_prompt, user_prompt, response_model)

    def complete_creative(self, system_prompt: str, user_prompt: object) -> str:
        return self.complete(system_prompt, user_prompt)

    def complete_structured_creative(
        self, system_prompt: str, user_prompt: object, response_model: type[T],
    ) -> T:
        return self.complete_structured(system_prompt, user_prompt, response_model)


class FailingLLMClient:
    """Test double that always raises on every call.

    Unstructured methods raise ``RuntimeError``; structured methods raise
    ``LLMError`` (matching the real client's behaviour).
    """

    def __init__(self, message: str = "API failed") -> None:
        self._message = message

    def complete(self, system_prompt: str, user_prompt: object) -> str:
        raise RuntimeError(self._message)

    def complete_light(self, system_prompt: str, user_prompt: object) -> str:
        raise RuntimeError(self._message)

    def complete_structured(
        self, system_prompt: str, user_prompt: object, response_model: type[T],
    ) -> T:
        raise LLMError(self._message)

    def complete_structured_light(
        self, system_prompt: str, user_prompt: object, response_model: type[T],
    ) -> T:
        raise LLMError(self._message)

    def complete_creative(self, system_prompt: str, user_prompt: object) -> str:
        raise RuntimeError(self._message)

    def complete_structured_creative(
        self, system_prompt: str, user_prompt: object, response_model: type[T],
    ) -> T:
        raise LLMError(self._message)
