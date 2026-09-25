"""LLM client exceptions.

The split that matters is retryable versus not. A dropped connection or a 503
from an overloaded vLLM will likely succeed on a second attempt; a response
that fails schema validation will fail again identically, so retrying it just
burns tokens and latency.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for every failure raised by the LLM layer."""


class LLMTransportError(LLMError):
    """The request never produced a usable response: timeout, 5xx, connection.

    Retryable.
    """


class LLMResponseError(LLMError):
    """A response arrived but could not be turned into the requested schema.

    Not retryable by itself: the same prompt and parameters will produce the
    same failure. Callers route these to ``needs_review``.
    """

    def __init__(self, message: str, *, raw_output: str | None = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output


class LLMSchemaError(LLMResponseError):
    """Valid JSON that does not satisfy the Pydantic model."""


class LLMEmptyResponseError(LLMResponseError):
    """The model returned no content at all."""
