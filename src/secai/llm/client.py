"""The vLLM client.

One entry point for every LLM call the platform makes, so the guarantees hold
everywhere rather than at each call site:

* extraction is schema-constrained (CLAUDE.md §2.4);
* transport faults retry with backoff, schema failures do not;
* each call is a Langfuse *generation* carrying model, parameters, token usage
  and latency;
* W3C trace context rides along, so vLLM's server spans join our trace.
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypeVar

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from secai.config import Settings, get_settings
from secai.llm.errors import LLMTransportError
from secai.llm.guided import parse_structured, response_format_for
from secai.telemetry.attributes import SPAN_LLM_EXTRACT
from secai.telemetry.http import inject_trace_context
from secai.telemetry.tracing import llm_generation

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

Message = dict[str, str]

# Faults worth a second attempt: the request never produced a usable answer.
_RETRYABLE = (APIConnectionError, APITimeoutError)


def _is_retryable_status(exc: BaseException) -> bool:
    """5xx and 429 are transient; 4xx means the request itself is wrong."""
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500 or exc.status_code == 429
    return False


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    logger.warning("vLLM call failed (attempt %d), retrying: %s", state.attempt_number, exc)


class LLMClient:
    """Wrapper around the OpenAI SDK pointed at vLLM."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        openai_client: OpenAI | None = None,
    ) -> None:
        resolved = settings if settings is not None else get_settings()
        self._settings = resolved
        self._vllm = resolved.vllm
        # Trace context is injected per request via extra_headers rather than
        # through a custom http_client: the OpenAI SDK is built on httpx2 while
        # the rest of the codebase uses httpx, and coupling to whichever the SDK
        # vendors today would break on its next major release.
        self._client = openai_client or OpenAI(
            base_url=self._vllm.base_url,
            api_key=self._vllm.api_key,
            timeout=self._vllm.timeout_s,
            max_retries=0,  # tenacity owns retries; double-retrying multiplies latency
        )

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- calls -------------------------------------------------------------
    def generate_structured(
        self,
        messages: list[Message],
        schema: type[ModelT],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        span_name: str = SPAN_LLM_EXTRACT,
        prompt_version: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ModelT:
        """Call vLLM under a JSON-schema constraint and return the parsed model.

        Raises:
            LLMTransportError: the call could not be completed after retries.
            LLMSchemaError: the response did not satisfy ``schema``.
        """
        # Only a test may switch this off; production extraction is always
        # schema-constrained (CLAUDE.md 2.4).
        response_format = (
            response_format_for(schema) if self._settings.guided_decoding_required else None
        )
        body = dict(extra_body) if extra_body else {}

        resolved_model = model or self._vllm.model
        resolved_temperature = temperature if temperature is not None else self._vllm.temperature
        resolved_max_tokens = max_tokens or self._vllm.max_tokens

        parameters = {
            "temperature": resolved_temperature,
            "max_tokens": resolved_max_tokens,
            "structured_output": response_format is not None,
        }

        with llm_generation(
            span_name,
            model=resolved_model,
            model_parameters=parameters,
            input_data=messages,
            metadata={"schema": schema.__name__, "prompt_version": prompt_version},
        ) as span:
            started = time.perf_counter()
            completion = self._create_with_retries(
                messages=messages,
                model=resolved_model,
                temperature=resolved_temperature,
                max_tokens=resolved_max_tokens,
                extra_body=body,
                response_format=response_format,
            )
            latency_s = time.perf_counter() - started

            content = completion.choices[0].message.content if completion.choices else None
            usage = _usage_details(completion)

            # Attach usage before validating: a schema failure is still a call
            # that cost tokens, and the trace should say so.
            span.update(output=content, usage_details=usage, metadata={"latency_s": latency_s})

            return parse_structured(content, schema)

    def _create_with_retries(
        self,
        *,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        extra_body: dict[str, Any],
        response_format: dict[str, Any] | None,
    ) -> ChatCompletion:
        @retry(
            retry=(retry_if_exception_type(_RETRYABLE) | retry_if_exception_type(APIStatusError)),
            stop=stop_after_attempt(self._vllm.max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            before_sleep=_log_retry,
            reraise=True,
        )
        def _call() -> ChatCompletion:
            create_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "extra_body": extra_body,
                # W3C traceparent, so vLLM's server spans nest under ours.
                "extra_headers": inject_trace_context(),
            }
            # Omitted entirely rather than passed as None: the SDK treats an
            # explicit None as a value and rejects it.
            if response_format is not None:
                create_kwargs["response_format"] = response_format

            try:
                # **kwargs defeats the SDK's overloads, so the result is Any.
                completion: ChatCompletion = self._client.chat.completions.create(**create_kwargs)
                return completion
            except APIStatusError as exc:
                if not _is_retryable_status(exc):
                    # A 4xx is our fault; retrying cannot fix it.
                    raise LLMTransportError(
                        f"vLLM rejected the request ({exc.status_code}): {exc}"
                    ) from exc
                raise

        try:
            return _call()
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            raise LLMTransportError(f"vLLM call failed after retries: {exc}") from exc

    # -- health ------------------------------------------------------------
    def health(self) -> bool:
        """True if the vLLM server answers. Used by integration tests."""
        url = self._vllm.base_url.rstrip("/").removesuffix("/v1") + "/health"
        try:
            return httpx.get(url, timeout=5.0).status_code == 200
        except httpx.HTTPError:
            return False


def _usage_details(completion: ChatCompletion) -> dict[str, int]:
    """Token counts in the shape Langfuse expects."""
    usage = completion.usage
    if usage is None:
        return {}
    return {
        "input": usage.prompt_tokens,
        "output": usage.completion_tokens,
        "total": usage.total_tokens,
    }
