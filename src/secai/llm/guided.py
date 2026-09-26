"""Guided decoding: Pydantic schema in, validated object out.

CLAUDE.md §2.4 is absolute about this — every extraction call is constrained by
a JSON schema and the result is parsed by Pydantic. Free-text parsing of model
output is not allowed anywhere in this codebase.

vLLM enforces the schema during sampling, so the token stream *cannot* leave
the grammar. That makes the model's output structurally correct by
construction; validation here is the second line of defence, catching a server
that ignored the constraint or a schema the model satisfied vacuously.

The constraint is expressed with OpenAI's standard ``response_format``
``json_schema``, not vLLM's older ``guided_json`` extra-body parameter. vLLM
0.30 renamed guided decoding to structured outputs and now **silently ignores**
``guided_json`` — the request succeeds and the model returns free prose, which
is the worst possible failure mode for a guardrail. ``response_format`` is also
what every other OpenAI-compatible server speaks.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from secai.llm.errors import LLMEmptyResponseError, LLMSchemaError

ModelT = TypeVar("ModelT", bound=BaseModel)


def json_schema_for(schema: type[BaseModel]) -> dict[str, Any]:
    """The JSON schema vLLM should constrain generation to."""
    return schema.model_json_schema()


def response_format_for(schema: type[BaseModel]) -> dict[str, Any]:
    """The ``response_format`` that constrains sampling to ``schema``.

    Verified against vLLM 0.30.0: this shape constrains the sampler, while the
    older ``guided_json`` extra-body parameter is accepted and ignored.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": json_schema_for(schema),
            "strict": True,
        },
    }


def parse_structured(content: str | None, schema: type[ModelT]) -> ModelT:
    """Validate raw model output against ``schema``.

    Raises:
        LLMEmptyResponseError: the model returned nothing.
        LLMSchemaError: the output was not JSON, or did not satisfy the schema.
    """
    if content is None or not content.strip():
        raise LLMEmptyResponseError("the model returned an empty response")

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        # Guided decoding should make this impossible; if it happens, the
        # constraint was not applied and the result cannot be trusted.
        raise LLMSchemaError(f"response was not valid JSON: {exc}", raw_output=content) from exc

    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        raise LLMSchemaError(
            f"response did not satisfy {schema.__name__}: {exc}", raw_output=content
        ) from exc
